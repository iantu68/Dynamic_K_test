# -*- coding: utf-8 -*-
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
    # ,
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
    try:
        for epoch in range(num_epochs):
            model.train()
            step = 0
            loss_all = 0
            loss_log = 0
            elapsed_all = 0
            elapsed_log = 0
            throttling_costs = 0
            comm_costs = 0
            losses = []
            for batch in train_dataloader:
                # break
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_start = time.time()
                outputs = model(**batch, training_step = step)
                loss = outputs.loss
                loss.backward()
                layer_grads = [] 

                for name, para in model.named_parameters():
                    if "bert.encoder.layer.0.moe_linear.experts.0.htoh4.weight" in name:
                                this_grads = para.grad.view(-1)
                                size = this_grads.size(0)
                                sum_gradients = this_grads[0].item()
                                # print(sum_gradients)
                                mean_gradients = sum_gradients / size
                                # print(mean_gradients)
                                layer_grads.append(mean_gradients)
                
                    # for j in range(8):
                    #     if "bert.encoder.layer.0.moe_linear.experts." + str(j) + ".htoh4.weight" in name:
                    #         parts = name.split(".")
                    #         expert_id = int(parts[6])
                    #         if para.requires_grad and para.grad is not None:
                    #             # print(para.grad.view(-1))
                    #             layer_grads[expert_id].extend(para.grad.view(-1).tolist())

                # print("layer0_grad: ", layer0_grads[0])
                # print("layer0_grad_size: ", layer0_grads[0].size())

                # 输出每个专家的梯度
                # for expert_id, grads in enumerate(layer_grads):
                #     print(f"Expert {expert_id} gradients: {grads}")
                    
                # 绘制损失图
                # for expert_id, grads in enumerate(layer_grads):
                #     plt.imshow(grads, cmap='viridis', interpolation='nearest', label=f"Expert {expert_id}")
                #     plt.colorbar()
                #     plt.title('Gradient Matrix')
                #     plt.savefig(f'gate_count_Layer0_expert_{expert_id}.png', bbox_inches='tight')
                #     plt.close()

                # plt.xlabel("Iteration")
                # plt.ylabel("Gradient Value")
                # plt.title("Gradient Flow for Each Expert")
                # plt.legend()
                # # plt.savefig(f'gate_count_Layer0_expert_{expert_id}.png', bbox_inches='tight')
                # plt.show()
                # print("Layer 1",  mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.weight'])
                # if 'conv1' in name:
                #     layer_grads.append(_parameters[name].grad)
                #     print("conv1_Loss:",_parameters[name].grad)
                loss_all += loss.item()
                losses.append(loss.item())
                optimizer.step()
                lr_scheduler.step()
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
                    progress_bar.update(1)

                # # 指定要修改的目錄路徑
                # directory_path = '/home/hagoo_file/MoE/examples/distributed'
                # # 指定舊的後綴和新的後綴
                # old_suffix = 'gate_count_layer_*.txt'
                # new_suffix = 'gate_count_layer_*.txt'
                # rename_files_in_directory(directory_path, old_suffix, new_suffix)
            for layer_grads in enumerate(layer_grads):
                x_values = range(len(layer_grads))
                plt.plot(x_values, layer_grads, marker='o')

            # 添加标题和标签
            plt.title('Histogram of Mean Gradients')
            plt.xlabel('Batech Step')
            plt.ylabel('Mean Gradient Value')

            # 保存图形为 PNG 文件
            plt.savefig('histogram.png')
            # 显示图形（可选）
            plt.show()        
            # 打开一个txt文件以写入模式
            with open('output.txt', 'w') as file:
                # 遍历数组的每个元素，将其写入文件
                for item in layer_grads:
                    file.write(str(item) + '\n')
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