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
import pandas as pd

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


def save_checkpoint(model, optimizer, epoch, step, file_path):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch}, step {step} to {file_path}")

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

    def compute_metrics(predictions, references):
        metric = evaluate.load('glue', 'mnli')
        return metric.compute(predictions=predictions, references=references)

    def preprocess_training_examples(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=128)

    def preprocess_validation_examples(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=128)



    # def compute_metrics(start_logits, end_logits, features, examples):

    #     n_best = 20
    #     max_answer_length = 30

    #     example_to_features = collections.defaultdict(list)
    #     for idx, feature in enumerate(features):
    #         example_to_features[feature["example_id"]].append(idx)

    #     predicted_answers = []
    #     for example in tqdm(examples):
    #         example_id = example["id"]
    #         context = example["context"]
    #         answers = []

    #         # Loop through all features associated with that example
    #         for feature_index in example_to_features[example_id]:
    #             start_logit = start_logits[feature_index]
    #             end_logit = end_logits[feature_index]
    #             offsets = features[feature_index]["offset_mapping"]

    #             start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
    #             end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
    #             for start_index in start_indexes:
    #                 for end_index in end_indexes:
    #                     # Skip answers that are not fully in the context
    #                     if offsets[start_index] is None or offsets[end_index] is None:
    #                         continue
    #                     # Skip answers with a length that is either < 0 or > max_answer_length
    #                     if (
    #                         end_index < start_index
    #                         or end_index - start_index + 1 > max_answer_length
    #                     ):
    #                         continue

    #                     answer = {
    #                         "text": context[offsets[start_index][0] : offsets[end_index][1]],
    #                         "logit_score": start_logit[start_index] + end_logit[end_index],
    #                     }
    #                     answers.append(answer)

    #         # Select the answer with the best score
    #         if len(answers) > 0:
    #             best_answer = max(answers, key=lambda x: x["logit_score"])
    #             predicted_answers.append(
    #                 {"id": example_id, "prediction_text": best_answer["text"]}
    #             )
    #         else:
    #             predicted_answers.append({"id": example_id, "prediction_text": ""})

    #     theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    #     return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    # max_length = 384
    # stride = 128
    # #------------SQuAD----------------------
    # def preprocess_training_examples(examples):
    #     questions = [q.strip() for q in examples["premise"]]

    #     inputs = tokenizer(
    #         questions,
    #         examples["context"],
    #         max_length=max_length,
    #         truncation="only_second",
    #         stride=stride,
    #         return_overflowing_tokens=True,
    #         return_offsets_mapping=True,
    #         padding="max_length",
    #     )
    #     masks = torch.tensor(inputs['attention_mask'], dtype=torch.bool)
    #     # masks = torch.Tensor(inputs['attention_mask'])
    #     # print("Masks : ", masks)
    #     # model(masks=masks)        #將mask 值外傳
    #     # print("masks : ", masks)
    #     offset_mapping = inputs.pop("offset_mapping")
    #     sample_map = inputs.pop("overflow_to_sample_mapping")
    #     answers = examples["answers"]
    #     start_positions = []
    #     end_positions = []
    #     training_padding_mask = [masks[i] for i in sample_map]
    #     inputs['train_padding_mask'] = training_padding_mask

    #     for i, offset in enumerate(offset_mapping):
    #         sample_idx = sample_map[i]
    #         answer = answers[sample_idx]
    #         start_char = answer["answer_start"][0]
    #         end_char = answer["answer_start"][0] + len(answer["text"][0])
    #         sequence_ids = inputs.sequence_ids(i)

    #         # Find the start and end of the context
    #         idx = 0
    #         while sequence_ids[idx] != 1:
    #             idx += 1
    #         context_start = idx
    #         while sequence_ids[idx] == 1:
    #             idx += 1
    #         context_end = idx - 1

    #         # If the answer is not fully inside the context, label is (0, 0)
    #         if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
    #             start_positions.append(0)
    #             end_positions.append(0)
    #         else:
    #             # Otherwise it's the start and end token positions
    #             idx = context_start
    #             while idx <= context_end and offset[idx][0] <= start_char:
    #                 idx += 1
    #             start_positions.append(idx - 1)

    #             idx = context_end
    #             while idx >= context_start and offset[idx][1] >= end_char:
    #                 idx -= 1
    #             end_positions.append(idx + 1)

    #     inputs["start_positions"] = start_positions
    #     inputs["end_positions"] = end_positions
    #     return inputs

    # def preprocess_validation_examples(examples):
    #     questions = [q.strip() for q in examples["premise"]]
    #     inputs = tokenizer(
    #         questions,
    #         examples["context"],
    #         max_length=max_length,
    #         truncation="only_second",
    #         stride=stride,
    #         return_overflowing_tokens=True,
    #         return_offsets_mapping=True,
    #         padding="max_length",
    #     )

    #     sample_map = inputs.pop("overflow_to_sample_mapping")
    #     example_ids = []

    #     for i in range(len(inputs["input_ids"])):
    #         sample_idx = sample_map[i]
    #         example_ids.append(examples["id"][sample_idx])

    #         sequence_ids = inputs.sequence_ids(i)
    #         offset = inputs["offset_mapping"][i]
    #         inputs["offset_mapping"][i] = [
    #             o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
    #         ]

    #     inputs["example_id"] = example_ids
    #     return inputs
    
    # EMA200计算函数
    def calculate_ema200(data, window=200):
        ema = [sum(data[:window])/window]
        alpha = 2 / (window + 1)
        for price in data[window:]:
            ema.append(ema[-1] + alpha * (price - ema[-1]))
        return ema[-1]

    
    # ---------------------------------------------------------------------------------------
    datasets = load_dataset('glue', 'mnli')
    # datasets = load_dataset("squad")
    print("dataset = ", datasets)
    # raw_datasets  = raw_datasets.train_test_split(test_size=0.2)
    # raw_datasets  = raw_datasets.rename_column("test", "validation")
    metric = evaluate.load('glue', 'mnli')
    # tokenized_squad = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset = datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    # print("train_data : ", train_dataset)
    eval_dataset = datasets["validation_matched"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=datasets["validation_matched"].column_names,
    )
    validation_dataset = eval_dataset#.remove_columns(["example_id", "offset_mapping"])

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
    eval_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size)
    num_epochs = num_epochs
    model_name="bert" # config1[some_args]['model']
    # metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
    # if use_wandb is True and local_rank == 0:
    #     wandb.init(    # set the wandb project where this run will be logged
    #     project="moe",
    #     name='moe-bert-gpu-4',
    #     settings=wandb.Settings(
    #     _stats_sample_rate_seconds=0.1,
    #     _stats_samples_to_average=1,
    #     ),
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 5e-05,
    #     "architecture": "bert",
    #     "dataset": "squad",
    #     "epochs": 1,
    #     }
    #     )

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-05)                        #學習率調整
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
        model = DDP(model, device_ids=[local_rank])#, moe_sync_group = moe_sync_group)
        model._sync_params()

    # 在訓練之前，獲取模型每一層的初始權重
    initial_weights = {name: p.data.clone() for name, p in model.named_parameters()}
    expert_grads_FFN0_Linear0_nabs = [[]for i in range(8)]
    expert_grads_FFN0_Linear1_nabs = [[]for i in range(8)]
    expert_grads_FFN1_Linear0_nabs = [[]for i in range(8)]
    expert_grads_FFN1_Linear1_nabs = [[]for i in range(8)]
    expert_grads_FFN2_Linear0_nabs = [[]for i in range(8)]
    expert_grads_FFN2_Linear1_nabs = [[]for i in range(8)]
    expert_grads_FFN3_Linear0_nabs = [[]for i in range(8)]
    expert_grads_FFN3_Linear1_nabs = [[]for i in range(8)]
    expert_grads_FFN4_Linear0_nabs = [[]for i in range(8)]
    expert_grads_FFN4_Linear1_nabs = [[]for i in range(8)]
    expert_grads_FFN5_Linear0_nabs = [[]for i in range(8)]
    expert_grads_FFN5_Linear1_nabs = [[]for i in range(8)]
    expert_grads_FFN6_Linear0_nabs = [[]for i in range(8)]
    expert_grads_FFN6_Linear1_nabs = [[]for i in range(8)]
    expert_grads_FFN7_Linear0_nabs = [[]for i in range(8)]
    expert_grads_FFN7_Linear1_nabs = [[]for i in range(8)]


    expert_grads_FFN0_Avg = [[]for i in range(8)]
    expert_grads_FFN1_Avg = [[]for i in range(8)]
    expert_grads_FFN2_Avg = [[]for i in range(8)]
    expert_grads_FFN3_Avg = [[]for i in range(8)]
    expert_grads_FFN4_Avg = [[]for i in range(8)]
    expert_grads_FFN5_Avg = [[]for i in range(8)]
    expert_grads_FFN6_Avg = [[]for i in range(8)]
    expert_grads_FFN7_Avg = [[]for i in range(8)]

    ema_comparison_masks = [[1] * 8 for _ in range(2)]

    count = 0
    step = 0
    loss_all = 0
    loss_log = 0
    elapsed_all = 0
    elapsed_log = 0
    throttling_costs = 0
    comm_costs = 0
    accuracy = 0
    losses = []
    acc = []
    gate_grads_0 = []
    avg_grads1 = None
    avg_grads2 = None
    previous_grad1 = None
    previous_grad2 = None
    try:
        for epoch in range(num_epochs):
            model.train()
            start_time = time.time()

            # if count % 30 == 0:
            #     save_checkpoint(model, optimizer, epoch, count, f"checkpoint_epoch_{epoch}.pth")
            # count += 1
            for batch in train_dataloader:
                # print(len(train_dataloader))
                # print("batch = ", batch)
                # print(batch['input_ids'])
                # break
                batch = {k: v.to(device) for k, v in batch.items()}
                # batch_padding_mask = batch['train_padding_mask']
                # print("batch_padding_mask=", batch_padding_mask)
                # batch.pop('train_padding_mask', None)  # 移除 'train_padding_mask' 键
                # print(batch_padding_mask)

                last_elements_FFN0 = [sub_arr[-1] for sub_arr in expert_grads_FFN0_Avg if sub_arr]
                last_elements_FFN1 = [sub_arr[-1] for sub_arr in expert_grads_FFN1_Avg if sub_arr]
                last_elements_FFN2 = [sub_arr[-1] for sub_arr in expert_grads_FFN2_Avg if sub_arr]
                last_elements_FFN3 = [sub_arr[-1] for sub_arr in expert_grads_FFN3_Avg if sub_arr]
                last_elements_FFN4 = [sub_arr[-1] for sub_arr in expert_grads_FFN4_Avg if sub_arr]
                last_elements_FFN5 = [sub_arr[-1] for sub_arr in expert_grads_FFN5_Avg if sub_arr]
                last_elements_FFN6 = [sub_arr[-1] for sub_arr in expert_grads_FFN6_Avg if sub_arr]
                last_elements_FFN7 = [sub_arr[-1] for sub_arr in expert_grads_FFN7_Avg if sub_arr]

                batch_start = time.time()
                # print("Here!!!")
                #定初始化定義向前傳播
                # print("="*10 + "Training.py" + "="*10)
                outputs = model(**batch)
                                # , training_step = step,)
                                # batch_padding_mask = batch_padding_mask,
                                # last_elements_FFN0 = last_elements_FFN0, last_elements_FFN1 = last_elements_FFN1,
                                # last_elements_FFN2 = last_elements_FFN2, last_elements_FFN3 = last_elements_FFN3,
                                # last_elements_FFN4 = last_elements_FFN4, last_elements_FFN5 = last_elements_FFN5,
                                # last_elements_FFN6 = last_elements_FFN6, last_elements_FFN7 = last_elements_FFN7,)
                                # ema_comparison_masks = ema_comparison_masks)
                                # last_elements_FFN0 = last_elements_FFN0, last_elements_FFN1 = last_elements_FFN1,
                                # last_elements_FFN0 = last_elements_FFN0, last_elements_FFN1 = last_elements_FFN1,)
                loss = outputs.loss
                if loss is None:
                    logger("Loss is None. Skipping this batch.")
                    continue
                loss.backward()
                loss_all += loss.item()
                losses.append(loss.item())
                optimizer.step()
                lr_scheduler.step()
                # if count == len(train_dataloader) - 1:
                # print(count)
                # if count % len(train_dataloader) == 0:
                print ("===============catch===============")
                #Single Expert gradient output
                for name, para in model.named_parameters():
                    for i in range(2):
                        # print("Layer = ", i)
                        for j in range(8):
                            if f"bert.encoder.layer.{i}.moe_linear.experts.{j}.htoh4.weight" in name:
                                # count += 1
                                # 获取上一次的梯度值，如果列表为空则设为0
                                previous_grads1_list = eval(f"expert_grads_FFN{i}_Linear0_nabs[{j}]")
                                if previous_grads1_list:
                                    previous_grad1 = previous_grads1_list[-1]
                                else:
                                    previous_grad1 = 0.0

                                # 获取当前梯度值
                                this_grads1 = para.grad.detach().norm().view(-1).cpu().numpy()
                                avg_grads1 = (previous_grad1 * len(previous_grads1_list) + this_grads1) / (len(previous_grads1_list) + 1)

                                # 更新梯度列表
                                eval(f"expert_grads_FFN{i}_Linear0_nabs[{j}]").append(avg_grads1)

                            if f"bert.encoder.layer.{i}.moe_linear.experts.{j}.h4toh.weight" in name:
                                # 获取上一次的梯度值，如果列表为空则设为0
                                previous_grads2_list = eval(f"expert_grads_FFN{i}_Linear1_nabs[{j}]")
                                if previous_grads2_list:
                                    previous_grad2 = previous_grads2_list[-1]
                                else:
                                    previous_grad2 = 0.0

                                # 获取当前梯度值
                                this_grads2 = para.grad.detach().norm().view(-1).cpu().numpy()
                                avg_grads2 = (previous_grad2 * len(previous_grads2_list) + this_grads2) / (len(previous_grads2_list) + 1)

                                # 更新梯度列表
                                eval(f"expert_grads_FFN{i}_Linear1_nabs[{j}]").append(avg_grads2)

                            if avg_grads1 is not None and avg_grads2 is not None:
                                # 计算两个梯度的平均值
                                linear_avg_grads = (avg_grads1 + avg_grads2) * 0.5

                                # 确保平均值是标量
                                if isinstance(linear_avg_grads, np.ndarray) or isinstance(linear_avg_grads, torch.Tensor):
                                    avg_grads_value = linear_avg_grads.item()  # 提取标量值
                                else:
                                    avg_grads_value = linear_avg_grads  # 如果已经是标量，直接赋值

                                # 更新平均梯度列表
                                eval(f"expert_grads_FFN{i}_Avg[{j}]").append(avg_grads_value)
                                # print(f"expert_grads_FFN0_Avg[{j}] = ", len(expert_grads_FFN0_Avg[j]))
                                # 清除临时变量
                                avg_grads1 = None
                                avg_grads2 = None

                            # # 计算EMA200
                            # if len(eval(f"expert_grads_FFN{i}_Avg[{j}]")) >= 200:
                            #     ema200 = calculate_ema200(eval(f"expert_grads_FFN{i}_Avg[{j}]"))
                            #     # 比较当前移动平均梯度和EMA200
                            #     if avg_grads_value > ema200:
                            #         ema_comparison_masks[i][j] = 1
                            #     else:
                            #         ema_comparison_masks[i][j] = 0

                                # print(f"ema_comparison_masks{i} = ", ema_comparison_masks[i])

                            
                            # 输出当前平均梯度值
                            # print(f"expert_grads_FFN{i}_Avg[{j}] = ", avg_grads_value)
                        

                optimizer.zero_grad()
                # torch.cuda.empty_cache()  # 清理缓存
                elapsed_all += time.time() - batch_start
                step += 1
                # if use_wandb is True and local_rank == 0:
                #     wandb.log({'batch_loss': loss_all/step})
                    # wandb.log({'batch_loss': loss_all})
                throttling_costs += outputs.total_throttling_costs
                comm_costs += outputs.total_comm_costs

                end_time = time.time() - start_time
                print(f"Epoch {epoch} | Loss {loss_all/step:.2f} | acc {accuracy:.2f} | time {end_time:.2f} |")


                # print(f"Epoch {epoch} | Loss {loss_all/step:.2f} | acc {best_acc:.2f}")

                # if local_rank == 0:
                #     progress_bar.set_description('Epoch {} | Loss {:.2f} | acc {:.2f} | mean batch time {:.2f}, mean throttling time {:.2f}, mean comm time {:.2f}'.format(
                #                                 epoch, (loss_all/step), best_acc, (elapsed_all/step)*1000, (throttling_costs/step)*1000, (comm_costs/step)*1000) )
                #     # loss_all_array.append(loss_all/step)
                #     progress_bar.update(1)

            model.eval()
            is_eval = True
            # question_answerer = pipeline("question-answering", model=model)
            start_logits = []
            end_logits = []
            # accelerator.print("Evaluation!")
            for idx, batch in enumerate(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)

                start_logits.append(outputs.start_logits.cpu().numpy())
                end_logits.append(outputs.end_logits.cpu().numpy())
            start_logits = np.concatenate(start_logits)
            end_logits = np.concatenate(end_logits)
            start_logits = start_logits[: len(validation_dataset)]
            end_logits = end_logits[: len(validation_dataset)]
            # metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
            metrics = compute_metrics(eval_dataset, datasets["validation_matched"])
            accuracy = metrics['exact_match']
            print("accuracy: ", accuracy)
            acc.append(accuracy)
            print(f"Epoch {epoch} | Loss {loss_all/step:.2f} | acc {accuracy:.2f} |=========================================")
                # {'exact_match': 83.0, 'f1': 88.25}
                # # if use_wandb is True and local_rank == 0:
                #     # wandb.log({'loss': loss_all/step, 'exact_match':metrics['exact_match'],'f1':metrics['f1']}) # 'rouge1': result['rouge1']})
                
                # if local_rank == 0:
                #     print(f'Eval | Loss: {loss_all/step:.6f} | Acc:{metrics["exact_match"]} | f1: {metrics["f1"]}')
                
                # if best_acc < metrics['f1']:
                #     save_model(model,model_name)
                #     best_acc = metrics['exact_match']

                
        np.save('losses.npy', losses)
        np.save('acc.npy', acc)
        # np.save('FFN0_grads_Avg.npy', expert_grads_FFN0_Avg)
        # np.save('FFN1_grads_Avg.npy', expert_grads_FFN1_Avg)
        for i in range(8):
            np.save(f"FFN0_grads_Avg_{i}.npy", expert_grads_FFN0_Avg[i])
            np.save(f"FFN1_grads_Avg_{i}.npy", expert_grads_FFN1_Avg[i])
            # np.save(f"FFN2_grads_Avg_{i}.npy", expert_grads_FFN2_Avg[i])
            # np.save(f"FFN3_grads_Avg_{i}.npy", expert_grads_FFN3_Avg[i])
            # np.save(f"FFN4_grads_Avg_{i}.npy", expert_grads_FFN4_Avg[i])
            # np.save(f"FFN5_grads_Avg_{i}.npy", expert_grads_FFN5_Avg[i])
            # np.save(f"FFN6_grads_Avg_{i}.npy", expert_grads_FFN6_Avg[i])
            # np.save(f"FFN7_grads_Avg_{i}.npy", expert_grads_FFN7_Avg[i])
            # np.save(f"expert_grads_FFN0_Linear0_{i}_nabs.npy", expert_grads_FFN0_Linear0_nabs[i])
            # np.save(f"expert_grads_FFN0_Linear1_{i}_nabs.npy", expert_grads_FFN0_Linear1_nabs[i])
            # np.save(f"expert_grads_FFN1_Linear0_{i}_nabs.npy", expert_grads_FFN1_Linear0_nabs[i])
            # np.save(f"expert_grads_FFN1_Linear1_{i}_nabs.npy", expert_grads_FFN1_Linear1_nabs[i])
            # np.save(f"expert_grads_FFN2_Linear0_{i}_nabs.npy", expert_grads_FFN2_Linear0_nabs[i])
            # np.save(f"expert_grads_FFN2_Linear1_{i}_nabs.npy", expert_grads_FFN2_Linear1_nabs[i])
            # np.save(f"expert_grads_FFN3_Linear0_{i}_nabs.npy", expert_grads_FFN3_Linear0_nabs[i])
            # np.save(f"expert_grads_FFN3_Linear1_{i}_nabs.npy", expert_grads_FFN3_Linear1_nabs[i])

            # dict_router = {}
            # index = 0
            
    except KeyboardInterrupt:
            # wandb.finish()
            logger('Exiting from training early')
    # if use_wandb is True and local_rank == 0:
    #     wandb.finish()
    del model
    del datasets
    del tokenizer