# -*- coding: utf-8 -*-
from pty import slave_open
import time
import pickle
import numpy as np
import nltk
import evaluate
import collections
import wandb
import os
import time
import matplotlib.pyplot as plt
import modeling

# from symbol import parameters
dir_path = os.path.dirname(os.path.realpath(__file__))

import torch
# Load model directly
from transformers import (AutoModelForSeq2SeqLM, DefaultDataCollator, get_scheduler, DataCollatorForSeq2Seq,
                            )
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
# import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modeling import Create_MoE_Model, save_model
from torch.utils.data.distributed import DistributedSampler
from fmoe.distributed import DistributedGroupedDataParallel as DDP

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'




# train bert
def train_Bert_MoE(**kwargs):
    device = kwargs['device']
    model = kwargs['model']
    tokenizer = kwargs['tokenizer']
    train_batch_size = kwargs['train_batch_size']
    eval_batch_size = kwargs['eval_batch_size']
    log_interval = kwargs['log_interval']
    eval_interval = kwargs['eval_interval']
    num_epochs = kwargs['num_epochs']
    logger = kwargs['logger']
    use_wandb = kwargs['use_wandb']
    world_size = kwargs['world_size']
    local_rank = kwargs['local_rank']
    global_rank = kwargs['global_rank']
    moe_sync_group = kwargs['moe_sync_group']
    dist = kwargs['dist']

    # def rename_files_in_directory(directory, old_suffix, new_suffix):
    # # 遍歷指定目錄中的所有文件
    # for filename in os.listdir(directory):
    #     # 確保文件是文件而不是目錄
    #     if os.path.isfile(os.path.join(directory, filename)):
    #         # 確保文件名稱以舊後綴結尾
    #         if filename.endswith(old_suffix):
    #             # 構建新的文件名
    #             new_filename = filename.replace(old_suffix, new_suffix)
    #             # 重命名文件
    #             os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    #             print(f"已將 {filename} 重命名為 {new_filename}")


    def compute_metrics(start_logits, end_logits, features, examples):

        n_best = 20
        max_answer_length = 30

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    max_length = 384
    stride = 128

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    datasets = load_dataset("squad")
    # raw_datasets  = raw_datasets.train_test_split(test_size=0.2)
    # raw_datasets  = raw_datasets.rename_column("test", "validation")
    # metric = evaluate.load("squad")
    # tokenized_squad = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset = datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    # eval_dataset = datasets["validation"].map(
    #     preprocess_validation_examples,
    #     batched=True,
    #     remove_columns=datasets["validation"].column_names,
    # )
    # validation_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])

    data_collator = DefaultDataCollator()

    batch_size=train_batch_size
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=batch_size,
        sampler = datasampler
    )
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    # )
    # eval_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size)
    num_epochs = num_epochs
    model_name="bert" # config1[some_args]['model']
    # metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
    if use_wandb is True and local_rank == 0:
        wandb.init(    # set the wandb project where this run will be logged
        project="moe",
        name='moe-bert-gpu-4',
        settings=wandb.Settings(
        _stats_sample_rate_seconds=0.1,
        _stats_samples_to_average=1,
        ),
        # track hyperparameters and run metadata
        config={
        "learning_rate": 5e-05,
        "architecture": "bert",
        "dataset": "squad",
        "epochs": 1,
        }
        )

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=3e-5)
                                # betas=(0.9,0.999),
                                # eps=1e-08)
    # num_epochs = 8
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    
    # ddp
    if dist:
        model = DDP(model, device_ids=[local_rank], moe_sync_group = moe_sync_group)
        model._sync_params()

    # 在訓練之前，獲取模型每一層的初始權重
    initial_weights = {name: p.data.clone() for name, p in model.named_parameters()}
    

    try:
        for epoch in range(num_epochs):
            model.train()
            count = 0
            step = 0
            loss_all = 0
            loss_log = 0
            elapsed_all = 0
            elapsed_log = 0
            throttling_costs = 0
            comm_costs = 0
            losses = []
            gate_grads_0 = []
            layer_grads_all = []

            expert_grads_0_L1_nabs=[]
            expert_grads_1_L1_nabs=[]
            expert_grads_2_L1_nabs=[]
            expert_grads_3_L1_nabs=[]
            expert_grads_4_L1_nabs=[]
            expert_grads_5_L1_nabs=[]
            expert_grads_6_L1_nabs=[]
            expert_grads_7_L1_nabs=[]
            expert_grads_0_L2_nabs=[]
            expert_grads_1_L2_nabs=[]
            expert_grads_2_L2_nabs=[]
            expert_grads_3_L2_nabs=[]
            expert_grads_4_L2_nabs=[]
            expert_grads_5_L2_nabs=[]
            expert_grads_6_L2_nabs=[]
            expert_grads_7_L2_nabs=[]

            expert_grads_0_L1_abs=[]
            expert_grads_1_L1_abs=[]
            expert_grads_2_L1_abs=[]
            expert_grads_3_L1_abs=[]
            expert_grads_4_L1_abs=[]
            expert_grads_5_L1_abs=[]
            expert_grads_6_L1_abs=[]
            expert_grads_7_L1_abs=[]
            expert_grads_0_L2_abs=[]
            expert_grads_1_L2_abs=[]
            expert_grads_2_L2_abs=[]
            expert_grads_3_L2_abs=[]
            expert_grads_4_L2_abs=[]
            expert_grads_5_L2_abs=[]
            expert_grads_6_L2_abs=[]
            expert_grads_7_L2_abs=[]

            expert_grads_0_L1_mean_first_nabs = []
            expert_grads_1_L1_mean_first_nabs = [] 
            expert_grads_2_L1_mean_first_nabs = [] 
            expert_grads_3_L1_mean_first_nabs = [] 
            expert_grads_4_L1_mean_first_nabs = [] 
            expert_grads_5_L1_mean_first_nabs = [] 
            expert_grads_6_L1_mean_first_nabs = [] 
            expert_grads_7_L1_mean_first_nabs = []
            expert_grads_0_L2_mean_first_nabs = []
            expert_grads_1_L2_mean_first_nabs = [] 
            expert_grads_2_L2_mean_first_nabs = [] 
            expert_grads_3_L2_mean_first_nabs = [] 
            expert_grads_4_L2_mean_first_nabs = [] 
            expert_grads_5_L2_mean_first_nabs = [] 
            expert_grads_6_L2_mean_first_nabs = [] 
            expert_grads_7_L2_mean_first_nabs = []
            
            expert_grads_0_L1_mean_first_abs = [] 
            expert_grads_1_L1_mean_first_abs = [] 
            expert_grads_2_L1_mean_first_abs = [] 
            expert_grads_3_L1_mean_first_abs = [] 
            expert_grads_4_L1_mean_first_abs = [] 
            expert_grads_5_L1_mean_first_abs = [] 
            expert_grads_6_L1_mean_first_abs = [] 
            expert_grads_7_L1_mean_first_abs = []  
            expert_grads_0_L2_mean_first_abs = [] 
            expert_grads_1_L2_mean_first_abs = [] 
            expert_grads_2_L2_mean_first_abs = [] 
            expert_grads_3_L2_mean_first_abs = [] 
            expert_grads_4_L2_mean_first_abs = [] 
            expert_grads_5_L2_mean_first_abs = [] 
            expert_grads_6_L2_mean_first_abs = [] 
            expert_grads_7_L2_mean_first_abs = []  

            expert_grads_0_L1_sub_first_nabs = []
            expert_grads_1_L1_sub_first_nabs = [] 
            expert_grads_2_L1_sub_first_nabs = [] 
            expert_grads_3_L1_sub_first_nabs = [] 
            expert_grads_4_L1_sub_first_nabs = [] 
            expert_grads_5_L1_sub_first_nabs = [] 
            expert_grads_6_L1_sub_first_nabs = [] 
            expert_grads_7_L1_sub_first_nabs = []    
            expert_grads_0_L2_sub_first_nabs = []
            expert_grads_1_L2_sub_first_nabs = [] 
            expert_grads_2_L2_sub_first_nabs = [] 
            expert_grads_3_L2_sub_first_nabs = [] 
            expert_grads_4_L2_sub_first_nabs = [] 
            expert_grads_5_L2_sub_first_nabs = [] 
            expert_grads_6_L2_sub_first_nabs = [] 
            expert_grads_7_L2_sub_first_nabs = []
            
            expert_grads_0_L1_sub_first_abs = [] 
            expert_grads_1_L1_sub_first_abs = [] 
            expert_grads_2_L1_sub_first_abs = [] 
            expert_grads_3_L1_sub_first_abs = [] 
            expert_grads_4_L1_sub_first_abs = [] 
            expert_grads_5_L1_sub_first_abs = [] 
            expert_grads_6_L1_sub_first_abs = [] 
            expert_grads_7_L1_sub_first_abs = []
            expert_grads_0_L2_sub_first_abs = [] 
            expert_grads_1_L2_sub_first_abs = [] 
            expert_grads_2_L2_sub_first_abs = [] 
            expert_grads_3_L2_sub_first_abs = [] 
            expert_grads_4_L2_sub_first_abs = [] 
            expert_grads_5_L2_sub_first_abs = [] 
            expert_grads_6_L2_sub_first_abs = [] 
            expert_grads_7_L2_sub_first_abs = []  
            
            loss_all_array = []
            previous_grads1 = 0
            previous_grads2 = 0
            previous_grads3 = 0
            previous_grads4 = 0
            previous_grads5 = 0
            previous_grads6 = 0
            previous_grads7 = 0
            previous_grads8 = 0

            for batch in train_dataloader:
                # print(len(train_dataloader))
                # print(batch)
                # break
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_start = time.time()
                outputs = model(**batch, training_step = step)
                loss = outputs.loss
                loss.backward()

                # if count == len(train_dataloader) - 1:
                if count % 50 == 0:
                    #Single Expert gradient output
                    for name, para in model.named_parameters():
                        if count == 0:
                            previous_grads1 = torch.zeros_like(para.grad.view(-1).cpu())
                            previous_grads2 = torch.zeros_like(para.grad.view(-1).cpu())
                            previous_grads3 = torch.zeros_like(para.grad.view(-1).cpu())
                            previous_grads4 = torch.zeros_like(para.grad.view(-1).cpu())
                            previous_grads5 = torch.zeros_like(para.grad.view(-1).cpu())
                            previous_grads6 = torch.zeros_like(para.grad.view(-1).cpu())
                            previous_grads7 = torch.zeros_like(para.grad.view(-1).cpu())
                            previous_grads8 = torch.zeros_like(para.grad.view(-1).cpu())
                            prev_weight1 = torch.zeros_like(para.view(-1).cpu())
                        for i in range(8):
                            expert_grads_L1_mean_first_nabs = f"expert_grads_{i}_L1_mean_first_nabs"
                            expert_grads_L2_mean_first_nabs = f"expert_grads_{i}_L2_mean_first_nabs"
                            expert_grads_L1_mean_first_abs = f"expert_grads_{i}_L1_mean_first_abs"
                            expert_grads_L2_mean_first_abs = f"expert_grads_{i}_L2_mean_first_abs"
                            expert_grads_L1_sub_first_nabs = f"expert_grads_{i}_L1_sub_first_nabs"
                            expert_grads_L2_sub_first_nabs = f"expert_grads_{i}_L2_sub_first_nabs"
                            expert_grads_L1_sub_first_abs = f"expert_grads_{i}_L1_sub_first_abs"
                            expert_grads_L2_sub_first_abs = f"expert_grads_{i}_L2_sub_first_abs"
                            expert_grads_L1_nabs = f"expert_grads_{i}_L1_nabs"
                            expert_grads_L2_nabs = f"expert_grads_{i}_L2_nabs"
                            expert_grads_L1_abs = f"expert_grads_{i}_L1_abs"
                            expert_grads_L2_abs = f"expert_grads_{i}_L2_abs"

                            
                            #L1_L2_nabs
                            np.set_printoptions(threshold=np.inf)
                            if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".htoh4.weight" in name:
                                # this_grads = para.detach().cpu()#.cpu().detach().mean()
                                this_weight = para.detach().view(-1).cpu()
                                print(this_weight)
                                weight_change = (this_weight - prev_weight1).mean()
                                print("weight_change : ", weight_change)
                                # print(f"expert_grads_{i}_L1_nabs: ", this_grads)
                                eval(expert_grads_L1_nabs).append(weight_change)
                                prev_weight1 = this_weight.clone()

                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".h4toh.weight" in name:
                                # this_grads = para.detach().cpu()#.cpu().detach().mean() 
                                # print(f"expert_grads_{i}_L2_nabs: ", this_grads)
                                # eval(expert_grads_L2_nabs).append(this_grads)

                            # #L1_L2_abs
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".htoh4.weight" in name:
                            #     this_grads = para.grad.view(-1)#.cpu().detach().abs().mean()
                            #     print(f"expert_grads_{i}_L1_abs: ", this_grads)
                            #     eval(expert_grads_L1_abs).append(this_grads)
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".h4toh.weight" in name:
                            #     this_grads = para.grad.view(-1)#.cpu().detach().abs().mean()
                            #     print(f"expert_grads_{i}_L2_abs: ", this_grads)
                            #     eval(expert_grads_L2_abs).append(this_grads)

                            # #_mean_first_nabs
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".htoh4.weight" in name:
                            #     if previous_grads1 is not None:
                            #         this_grads = para.grad.view(-1).cpu()
                            #         print(this_grads) 
                            #         print(previous_grads1) 
                            #         grad_change = (this_grads.detach().mean() - previous_grads1.detach().mean())
                            #         print(f"expert_grads_{i}_L1_mean_first_nabs: ", grad_change)
                            #         eval(expert_grads_L1_mean_first_nabs).append(grad_change)
                            #     previous_grads1 = this_grads.clone()
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".h4toh.weight" in name:
                            #     if previous_grads2 is not None:
                            #         this_grads = para.grad.view(-1).cpu() 
                            #         grad_change = (this_grads.detach().mean() - previous_grads2.detach().mean())
                            #         print(f"expert_grads_{i}_L2_mean_first_nabs: ", grad_change)
                            #         eval(expert_grads_L2_mean_first_nabs).append(grad_change)
                            #     previous_grads2 = this_grads.clone()

                            # #_mean_first_abs
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".htoh4.weight" in name:
                            #     if previous_grads3 is not None:
                            #         this_grads = para.grad.view(-1).cpu() 
                            #         grad_change = (this_grads.detach().mean() - previous_grads3.detach().mean()).abs()
                            #         print(f"expert_grads_{i}_L1_mean_first_abs: ", grad_change)
                            #         eval(expert_grads_L1_mean_first_abs).append(grad_change)
                            #     previous_grads3 = this_grads.clone()
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".h4toh.weight" in name:
                            #     if previous_grads4 is not None:
                            #         this_grads = para.grad.view(-1).cpu() 
                            #         grad_change = (this_grads.detach().mean() - previous_grads4.detach().mean()).abs()
                            #         print(f"expert_grads_{i}_L2_mean_first_abs: ", grad_change)
                            #         eval(expert_grads_L2_mean_first_abs).append(grad_change)
                            #     previous_grads4 = this_grads.clone()

                            # #_sub_first_nabs
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".htoh4.weight" in name:
                            #     if previous_grads5 is not None:
                            #         this_grads = para.grad.view(-1).cpu() 
                            #         grad_change = (this_grads - previous_grads5).detach().mean()
                            #         print(f"expert_grads_{i}_L1_sub_first_nabs: ", grad_change)
                            #         eval(expert_grads_L1_sub_first_nabs).append(grad_change)
                            #     previous_grads5 = this_grads.clone()
                            #     # print(previous_grads5)
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".h4toh.weight" in name:
                            #     if previous_grads6 is not None:
                            #         this_grads = para.grad.view(-1).cpu() 
                            #         grad_change = (this_grads - previous_grads6).detach().mean()
                            #         print(f"expert_grads_{i}_L2_sub_first_nabs: ", grad_change)
                            #         eval(expert_grads_L2_sub_first_nabs).append(grad_change)
                            #     previous_grads6 = this_grads.clone()

                            # #_sub_first_abs
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".htoh4.weight" in name:
                            #     if previous_grads7 is not None:
                            #         this_grads = para.grad.view(-1).cpu() 
                            #         grad_change = (this_grads - previous_grads7).detach().abs().mean()
                            #         print(f"expert_grads_{i}_L1_sub_first_abs: ", grad_change)
                            #         eval(expert_grads_L1_sub_first_abs).append(grad_change)
                            #     previous_grads7 = this_grads.clone()
                            # if "bert.encoder.layer.0.moe_linear.experts." + str(i) + ".h4toh.weight" in name:
                            #     if previous_grads8 is not None:
                            #         this_grads = para.grad.view(-1).cpu() 
                            #         grad_change = (this_grads - previous_grads8).detach().abs().mean()
                            #         print(f"expert_grads_{i}_L2_sub_first_abs: ", grad_change)
                            #         eval(expert_grads_L2_sub_first_abs).append(grad_change)
                            #     previous_grads8 = this_grads.clone()


                            
                    # if "bert.encoder.layer.0.moe_linear.gate.gate.weight" in name:
                    #     # print(para.shape)
                    #     current_gate = para.grad.view(-1).cpu()
                    #     # print(f"layer_0_gate_weight: ", para)
                    #     # print(f"Layer_0_gate_for_exp_{i}: ", current_gate)
                    #     # print("current_gate_grads_exp_i: ", current_gate_grads_exp_i)
                    #     gate_mean_gradients = current_gate.detach().abs().mean()
                    #     # print(gate_mean_gradients)
                    #     gate_grads_0.append(gate_mean_gradients)
                count += 1
                # print(count)

                        


                loss_all += loss.item()
                losses.append(loss.item())
                optimizer.step()
                lr_scheduler.step()

                # # 計算每個epoch後權重的變化
                # weight_changes = {name: (p.data - initial_weights[name]).abs().sum().item() for name, p in model.named_parameters()}

                # # 儲存或打印權重變化數據
                # print(f'Epoch {epoch}:')
                # for name, change in weight_changes.items():
                #     print(f' - {name} weight change: {change}')



                        # if "bert.encoder.layer.0.moe_linear.gate.gate.weight" in name:
                        #     current_gate = para.grad.view(-1).cpu()
                        #     print(f"layer_0_gate_weight: ", para)
                        #     print(f"Layer_0_gate_for_exp_{i}: ", current_gate)
                        #     current_gate_grads_exp_i = current_gate[i]
                        #     print("current_gate_grads_exp_i: ", current_gate_grads_exp_i)
                        #     gate_mean_gradients = current_gate_grads_exp_i.detach().abs().mean()
                        #     print(f"gate_grads_for_exp_{i} : ", gate_mean_gradients)
                        #     eval(gate_grads).append(gate_mean_gradients)

                    

                    # if "bert.encoder.layer.0.moe_linear.layer_norm.weight" in name:
                    #     layer_grads = para.grad.view(-1).cpu()
                    #     # print("Layer: ", para)
                    #     layer_mean_gradients = layer_grads.detach().abs().mean()
                    #     # print(f"Layer_grads : ", layer_mean_gradients)
                    #     layer_grads_all.append(layer_mean_gradients)
                # print("file_size: ", len(layer_grads_0))
                        




                    # 读取最终权重
                    # final_weight = final_gate_weights[i]
                    
                #     # 计算当前权重和最终权重的L2范数差异
                #     current_diff = torch.norm(current_weight - final_weight).item()
                #     # 归一化这个差异
                #     normalized_diff = abs(current_diff) / abs(initial_final_diff[i])
                #     normalized_diffs[i].append(normalized_diff)  # 将归一化差异添加到对应层的列表中

                optimizer.zero_grad()
                elapsed_all += time.time() - batch_start
                step += 1
                if use_wandb is True and local_rank == 0:
                    wandb.log({'batch_loss': loss_all/step})
                    # wandb.log({'batch_loss': loss_all})
                throttling_costs += outputs.total_throttling_costs
                comm_costs += outputs.total_comm_costs
                if local_rank == 0:
                    progress_bar.set_description('Epoch {} | Loss {:.2f} | acc {:.2f} | mean batch time {:.2f}, mean throttling time {:.2f}, mean comm time {:.2f}'.format(
                                                epoch, (loss_all/step), best_acc, (elapsed_all/step)*1000, (throttling_costs/step)*1000, (comm_costs/step)*1000) )
                    loss_all_array.append(loss_all/step)
                    progress_bar.update(1)

                # # 指定要修改的目錄路徑
                # directory_path = '/home/hagoo_file/MoE/examples/distributed'
                # # 指定舊的後綴和新的後綴
                # old_suffix = 'gate_count_layer_*.txt'
                # new_suffix = 'gate_count_layer_*.txt'
                # rename_files_in_directory(directory_path, old_suffix, new_suffix)

            # with open(f"layer_grads_all.txt", 'a') as file:
            #         for item in layer_grads_all:
            #             file.write(str(item) + '\n')

            for i in range(8):
                # with open(f"expert_grads_{i}_L1_abs.txt", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L1_abs"):
                #         file.write(str(item) + '\n')
                # with open(f"expert_grads_{i}_L2_abs.txt", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L2_abs"):
                #         file.write(str(item) + '\n')

                with open(f"expert_grads_{i}_L1_nabs.txt", 'a') as file:
                    for item in eval(f"expert_grads_{i}_L1_nabs"):
                        file.write(str(item) + '\n')
                with open(f"expert_grads_{i}_L2_nabs.txt", 'a') as file:
                    for item in eval(f"expert_grads_{i}_L2_nabs"):


                        file.write(str(item) + '\n')

                # with open(f"expert_grads_{i}_L1_mean_first_nabs", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L1_mean_first_nabs"):
                #         file.write(str(item) + '\n')
                # with open(f"expert_grads_{i}_L2_mean_first_nabs", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L2_mean_first_nabs"):
                #         file.write(str(item) + '\n')

                # with open(f"expert_grads_{i}_L1_mean_first_abs", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L1_mean_first_abs"):
                #         file.write(str(item) + '\n')
                # with open(f"expert_grads_{i}_L2_mean_first_abs", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L2_mean_first_abs"):
                #         file.write(str(item) + '\n')

                # with open(f"expert_grads_{i}_L1_sub_first_nabs", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L1_sub_first_nabs"):
                #         file.write(str(item) + '\n')
                # with open(f"expert_grads_{i}_L2_sub_first_nabs", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L2_sub_first_nabs"):
                #         file.write(str(item) + '\n')

                # with open(f"expert_grads_{i}_L1_sub_first_abs", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L1_sub_first_abs"):
                #         file.write(str(item) + '\n')
                # with open(f"expert_grads_{i}_L2_sub_first_abs", 'a') as file:
                #     for item in eval(f"expert_grads_{i}_L2_sub_first_abs"):
                #         file.write(str(item) + '\n')


            # with open(f"loss_value.txt", 'a') as file:
            #         for item in loss_all_array:
            #             file.write(str(item) + '\n')

            

            # with open("gate_grads_0.txt", 'a') as file:
            #     for item in gate_grads_0:
            #         file.write(str(item) + '\n')
            # dict_router = {}
            # index = 0
                # if step % eval_interval == 0:
                #     model.eval()
                #     # question_answerer = pipeline("question-answering", model=model)
                #     start_logits = []
                #     end_logits = []
                #     # accelerator.print("Evaluation!")
                #     for idx, batch in enumerate(eval_dataloader):
                #         batch = {k: v.to(device) for k, v in batch.items()}
                #         with torch.no_grad():
                #             outputs = model(**batch)

                #         start_logits.append(outputs.start_logits.cpu().numpy())
                #         end_logits.append(outputs.end_logits.cpu().numpy())
                #     start_logits = np.concatenate(start_logits)
                #     end_logits = np.concatenate(end_logits)
                    # start_logits = start_logits[: len(validation_dataset)]
                    # end_logits = end_logits[: len(validation_dataset)]
                    # metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
                    # metrics = compute_metrics(start_logits, end_logits, eval_dataset, datasets["validation"])
                    # {'exact_match': 83.0, 'f1': 88.25}
                    # if use_wandb is True and local_rank == 0:
                    #     wandb.log({'loss': loss_all/step, 'exact_match':metrics['exact_match'],'f1':metrics['f1']}) # 'rouge1': result['rouge1']})
                    # if best_acc < metrics['f1']:
                    #     save_model(model,model_name)
                    #     best_acc = metrics['exact_match']
    except KeyboardInterrupt:
        if use_wandb is True and local_rank == 0:
            wandb.finish()
            logger('Exiting from training early')
    if use_wandb is True and local_rank == 0:
        wandb.finish()
    del model
    del datasets
    del tokenizer