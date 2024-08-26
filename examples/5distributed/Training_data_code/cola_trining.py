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

dir_path = os.path.dirname(os.path.realpath(__file__))

import torch
from transformers import (AutoModelForSeq2SeqLM, DefaultDataCollator, get_scheduler, DataCollatorForSeq2Seq)
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
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

    def calculate_ema200(data, window=200):
        ema = [sum(data[:window]) / window]
        alpha = 2 / (window + 1)
        for price in data[window:]:
            ema.append(ema[-1] + alpha * (price - ema[-1]))
        return ema[-1]

    # ---------------------------------------------------------------------------------------
    datasets = load_dataset('glue', 'cola')
    print("data = ", datasets)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

    train_dataset = datasets['train'].map(preprocess_function, batched=True)
    eval_dataset = datasets['validation'].map(preprocess_function, batched=True)

    train_dataset = train_dataset.remove_columns(['idx', 'sentence'])
    eval_dataset = eval_dataset.remove_columns(['idx', 'sentence'])

    train_dataset.set_format('torch')
    eval_dataset.set_format('torch')

    data_collator = DefaultDataCollator()

    batch_size = train_batch_size
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=batch_size,
        sampler=datasampler
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
    num_epochs = num_epochs
    model_name = "bert"

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=5e-05)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0

    if dist:
        model = DDP(model, device_ids=[local_rank], moe_sync_group=moe_sync_group)
        model._sync_params()

    initial_weights = {name: p.data.clone() for name, p in model.named_parameters()}
    expert_grads_FFN0_Linear0_nabs = [[] for i in range(8)]
    expert_grads_FFN0_Linear1_nabs = [[] for i in range(8)]
    expert_grads_FFN1_Linear0_nabs = [[] for i in range(8)]
    expert_grads_FFN1_Linear1_nabs = [[] for i in range(8)]
    expert_grads_FFN2_Linear0_nabs = [[] for i in range(8)]
    expert_grads_FFN2_Linear1_nabs = [[] for i in range(8)]
    expert_grads_FFN3_Linear0_nabs = [[] for i in range(8)]
    expert_grads_FFN3_Linear1_nabs = [[] for i in range(8)]
    expert_grads_FFN4_Linear0_nabs = [[] for i in range(8)]
    expert_grads_FFN4_Linear1_nabs = [[] for i in range(8)]
    expert_grads_FFN5_Linear0_nabs = [[] for i in range(8)]
    expert_grads_FFN5_Linear1_nabs = [[] for i in range(8)]
    expert_grads_FFN6_Linear0_nabs = [[] for i in range(8)]
    expert_grads_FFN6_Linear1_nabs = [[] for i in range(8)]
    expert_grads_FFN7_Linear0_nabs = [[] for i in range(8)]
    expert_grads_FFN7_Linear1_nabs = [[] for i in range(8)]

    expert_grads_FFN0_Avg = [[] for i in range(8)]
    expert_grads_FFN1_Avg = [[] for i in range(8)]
    expert_grads_FFN2_Avg = [[] for i in range(8)]
    expert_grads_FFN3_Avg = [[] for i in range(8)]
    expert_grads_FFN4_Avg = [[] for i in range(8)]
    expert_grads_FFN5_Avg = [[] for i in range(8)]
    expert_grads_FFN6_Avg = [[] for i in range(8)]
    expert_grads_FFN7_Avg = [[] for i in range(8)]

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

            if count % 30 == 0:
                save_checkpoint(model, optimizer, epoch, count, f"checkpoint_epoch_{epoch}.pth")
            count += 1
            for batch in train_dataloader:
                step += 1
                batch = {k: v.to(device) for k, v in batch.items()}

                last_elements_FFN0 = [sub_arr[-1] for sub_arr in expert_grads_FFN0_Avg if sub_arr]
                last_elements_FFN1 = [sub_arr[-1] for sub_arr in expert_grads_FFN1_Avg if sub_arr]
                last_elements_FFN2 = [sub_arr[-1] for sub_arr in expert_grads_FFN2_Avg if sub_arr]
                last_elements_FFN3 = [sub_arr[-1] for sub_arr in expert_grads_FFN3_Avg if sub_arr]
                last_elements_FFN4 = [sub_arr[-1] for sub_arr in expert_grads_FFN4_Avg if sub_arr]
                last_elements_FFN5 = [sub_arr[-1] for sub_arr in expert_grads_FFN5_Avg if sub_arr]
                last_elements_FFN6 = [sub_arr[-1] for sub_arr in expert_grads_FFN6_Avg if sub_arr]
                last_elements_FFN7 = [sub_arr[-1] for sub_arr in expert_grads_FFN7_Avg if sub_arr]

                batch_start = time.time()
                outputs = model(**batch,
                                last_elements_FFN0=last_elements_FFN0, last_elements_FFN1=last_elements_FFN1)
                
                loss = outputs.loss
                loss.backward()
                loss_all += loss.item()
                losses.append(loss.item())
                optimizer.step()
                lr_scheduler.step()

                for name, para in model.named_parameters():
                    for i in range(8):
                        for j in range(8):
                            if f"bert.encoder.layer.{i}.moe_linear.experts.{j}.htoh4.weight" in name:
                                previous_grads1_list = eval(f"expert_grads_FFN{i}_Linear0_nabs[{j}]")
                                if previous_grads1_list:
                                    previous_grad1 = previous_grads1_list[-1]
                                else:
                                    previous_grad1 = 0.0

                                this_grads1 = para.grad.detach().norm().view(-1).cpu().numpy()
                                avg_grads1 = (previous_grad1 * len(previous_grads1_list) + this_grads1) / (
                                            len(previous_grads1_list) + 1)

                                eval(f"expert_grads_FFN{i}_Linear0_nabs[{j}]").append(avg_grads1)

                            if f"bert.encoder.layer.{i}.moe_linear.experts.{j}.h4toh.weight" in name:
                                previous_grads2_list = eval(f"expert_grads_FFN{i}_Linear1_nabs[{j}]")
                                if previous_grads2_list:
                                    previous_grad2 = previous_grads2_list[-1]
                                else:
                                    previous_grad2 = 0.0

                                this_grads2 = para.grad.detach().norm().view(-1).cpu().numpy()
                                avg_grads2 = (previous_grad2 * len(previous_grads2_list) + this_grads2) / (
                                            len(previous_grads2_list) + 1)

                                eval(f"expert_grads_FFN{i}_Linear1_nabs[{j}]").append(avg_grads2)

                            if avg_grads1 is not None and avg_grads2 is not None:
                                linear_avg_grads = (avg_grads1 + avg_grads2) * 0.5

                                if isinstance(linear_avg_grads, np.ndarray) or isinstance(linear_avg_grads, torch.Tensor):
                                    avg_grads_value = linear_avg_grads.item()
                                else:
                                    avg_grads_value = linear_avg_grads

                                eval(f"expert_grads_FFN{i}_Avg[{j}]").append(avg_grads_value)

                                avg_grads1 = None
                                avg_grads2 = None

                optimizer.zero_grad()
                end_time = time.time() - start_time
                step += 1
                print(f"Epoch {epoch} | Loss {loss_all / step:.2f} | acc {accuracy:.2f} | time {end_time:.2f} |")

                model.eval()
                eval_acc = 0
                eval_count = 0

                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    eval_acc += (predictions == batch['labels']).sum().item()
                    eval_count += len(batch['labels'])

                accuracy = (eval_acc / eval_count) * 100
                acc.append(accuracy)

        np.save('losses.npy', losses)
        np.save('acc.npy', acc)

        for i in range(8):
            np.save(f"FFN0_grads_Avg_{i}.npy", expert_grads_FFN0_Avg[i])
            np.save(f"FFN1_grads_Avg_{i}.npy", expert_grads_FFN1_Avg[i])
            np.save(f"FFN2_grads_Avg_{i}.npy", expert_grads_FFN2_Avg[i])
            np.save(f"FFN3_grads_Avg_{i}.npy", expert_grads_FFN3_Avg[i])
            np.save(f"FFN4_grads_Avg_{i}.npy", expert_grads_FFN4_Avg[i])
            np.save(f"FFN5_grads_Avg_{i}.npy", expert_grads_FFN5_Avg[i])
            np.save(f"FFN6_grads_Avg_{i}.npy", expert_grads_FFN6_Avg[i])
            np.save(f"FFN7_grads_Avg_{i}.npy", expert_grads_FFN7_Avg[i])

    except KeyboardInterrupt:
        logger('Exiting from training early')

    del model
    del datasets
    del tokenizer
