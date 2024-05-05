import time
import pickle
import numpy as np
import nltk
import evaluate
import collections
import wandb
import os
import time
dir_path = os.path.dirname(os.path.realpath(__file__))
import pandas as pd

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
import torch

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
    loss_values = []  # 空列表來儲存每步的損失值

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
        
        # for key, value in inputs.items():
        #     print(f"{key}: {value}")       
        # for i in range(len(inputs["input_ids"])):
        #     print(inputs["input_ids"])
        # print(inputs)
        # for i, ids in enumerate(inputs.input_ids):
        #     for j, token_id in enumerate(ids):
        #       print(f"token: {j}: {token_id}")

        #     # for id in ids:
        #     #     print(f"Input {id}")
        # print(questions)
        # questions_tokens = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
        # example_tokens = tokenizer(examples["context"], padding=True, truncation=True, return_tensors="pt")

        # print(questions_tokens)
        # print(examples)
        # print(example_tokens)
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
    metric = evaluate.load("squad")
    # tokenized_squad = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset = datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    # eval_subset = datasets["validation"].select(range(3,4))
    # eval_subset = datasets["validation"].select(range(501))
    eval_subset = datasets["validation"].select(range(1))
    num_eval_subset = len(eval_subset)
    
    eval_dataset = eval_subset.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )
    validation_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])

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
    
    # initial_weights = {name: p.data.clone() for name, p in model.named_parameters()}
    
    get_final_gate = False
    
    if get_final_gate == False:
        def load_final_gate_weights(num_layers, device='cpu'):
            final_gate_weights = {}
            for i in range(num_layers):
                filename = f"normalized_diff/layer_50_gate_weight/gate_layer_{i}_weights_epoch_49.txt"
                # 使用numpy读取数据，假设权重是以高精度格式保存的
                with open(filename, 'r') as f:
                    weight_array = np.loadtxt(f, dtype=np.float32)  # 加载为float64以保持精度
                    weight_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device)  # 转换为Tensor，使用float64
                final_gate_weights[i] = weight_tensor
                
                # 打印权重，保持原始精度
                # print(np.array2string(weight_tensor.numpy(), formatter={'float_kind':'{:.18e}'.format}))
            return final_gate_weights
        
        # 在训练开始之前，记录每一层gate的初始权重状态
        initial_gate_weights = [layer.moe_linear.gate.gate.weight.data.clone().cpu() for layer in model.bert.encoder.layer]
        final_gate_weights = load_final_gate_weights(len(model.bert.encoder.layer))
        initial_final_diff = {i: torch.norm(initial_gate_weights[i] - final_gate_weights[i]).item() for i in range(len(model.bert.encoder.layer))}    # 存储权重变化数据的列表
        print(f"initial_final_diff={initial_final_diff}\n")
    
    count = 0  # 初始化计数器
    write_count = 0
    # 初始化一个字典来存储每一层的归一化差异
    normalized_diffs = {i: [] for i in range(len(model.bert.encoder.layer))}
    current_diffs = {i: [] for i in range(len(model.bert.encoder.layer))}
    current_weights = {i: [] for i in range(len(model.bert.encoder.layer))}
    
    reciprocal ={i: [] for i in range(len(model.bert.encoder.layer))}
    exponential_decay = {i: [] for i in range(len(model.bert.encoder.layer))}

    # 初始化权重跟踪和收敛率记录
    layer_weights_previous_step = {i: None for i in range(len(model.bert.encoder.layer))}
    layer_convergence_rates = {i: [] for i in range(len(model.bert.encoder.layer))}

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

            for batch in train_dataloader:
                # break
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_start = time.time()
                # outputs = model(**batch, training_step = step, is_eval = False)
                outputs = model(**batch, training_step = step, is_zero = 0,is_eval=False)
                loss = outputs.loss
                loss.backward()
                loss_all += loss.item()
                optimizer.step()
                lr_scheduler.step()
                
                
                for i, layer in enumerate(model.bert.encoder.layer):
                    # 获取当前权重并转移到CPU
                    current_weight = layer.moe_linear.gate.gate.weight.data.cpu() # dtype=float32
                    current_weights[i] = current_weight
                    # 读取最终权重
                    if get_final_gate == False:

                        final_weight = final_gate_weights[i]
                        
                        # 计算当前权重和最终权重的L2范数差异
                        current_diff = torch.norm(current_weight - final_weight).item()
                        current_diffs[i].append(current_diff)
                        # 归一化这个差异
                        normalized_diff = current_diff / (initial_final_diff[i])
                        normalized_diffs[i].append(normalized_diff)  # 将归一化差异添加到对应层的列表中
                        
                        # 反比例轉換
                        reciprocal[i].append(1.0 / (1.0 + current_diff))

                        # 指數負比例轉換
                        exponential_decay[i].append(np.exp(-current_diff))
                        
                        if layer_weights_previous_step[i] is not None:
                            prev_weight = layer_weights_previous_step[i]
                            norm_diff_t1 = torch.norm(prev_weight - final_weight).item()
                            norm_diff_t2 = torch.norm(current_weight - final_weight).item()
                            # 定义2.1收敛率计算
                            if norm_diff_t1 != 0:  # Avoid division by zero
                                convergence_rate = (norm_diff_t1 - norm_diff_t2) / norm_diff_t1
                                layer_convergence_rates[i].append(convergence_rate)
                    # 更新当前权重为下一步的"前一步权重"
                    layer_weights_previous_step[i] = current_weight.clone()   
               
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
                    loss_values.append(loss_all/step)
                    progress_bar.update(1)
            # dict_router = {}
            # index = 0
                
                if step % eval_interval == 0:
                    # 將normalized diff 寫入txt
                    
                            
                    # 在每个epoch结束后，为每一层创建新的文件并输出当前权重
                    # for i, layer in enumerate(model.bert.encoder.layer):
                    #     # current_weight = layer.moe_linear.gate.gate.weight.data.cpu().numpy()  # 转换Tensor到NumPy数组并移动到CPU
                    #     current_weight = layer.moe_linear.gate.gate.weight.data.cpu()  # 移动到CPU
                    #     print(f'Epoch {epoch} - Layer {i} current_weight {current_weight}')
                    #     filename = f"l2_tinybert/l2_2/gate_layer_{i}_weights_{count}.txt"  # 创建文件名
                    #     with open(filename, 'w') as file:
                    #         np.savetxt(file, current_weight, fmt='%.18e')  # 写入权重到文件
                    #     print(f'Epoch {epoch} - Layer {i} current_weight written to {filename}')
                    # count += 1  # 更新文件计数器
                    model.eval()
                    is_eval = True
                    # question_answerer = pipeline("question-answering", model=model)
                    start_logits = []
                    end_logits = []
                    # accelerator.print("Evaluation!")
                    for idx, batch in enumerate(eval_dataloader):
                        
                        # print(f"idx={idx}\n batch={batch}")
                        batch = {k: v.to(device) for k, v in batch.items()}

                        for eval_idx, sequence in enumerate(batch['input_ids']):
                            eval_num=(len(batch['input_ids']))
                            # print(f"{j}\n{eval_num}")
                            # print(f"j={j} token={sequence}\n")
                            for k,token_id in enumerate(sequence):
                                
                                if token_id==102:
                                    # num_zero = len(eval_subset)//(train_batch_size)
                                    
                                    is_zero = k
                                    with torch.no_grad():
                                        outputs = model(**batch, training_step = step, is_zero=is_zero, is_eval=is_eval,epoch=epoch,eval_idx=eval_idx,eval_num=eval_num,num_eval_subset=num_eval_subset,train_batch_size=train_batch_size)

                                    # print(is_zero)
                        with torch.no_grad():
                            # outputs = model(**batch, is_eval = True)
                            outputs = model(**batch, training_step = step, is_zero=is_zero, is_eval=is_eval,epoch=epoch,)
                            
                        start_logits.append(outputs.start_logits.cpu().numpy())
                        end_logits.append(outputs.end_logits.cpu().numpy())
                    start_logits = np.concatenate(start_logits)
                    end_logits = np.concatenate(end_logits)
                    start_logits = start_logits[: len(validation_dataset)]
                    end_logits = end_logits[: len(validation_dataset)]
                    # metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
                    metrics = compute_metrics(start_logits, end_logits, eval_dataset, datasets["validation"])
                    # {'exact_match': 83.0, 'f1': 88.25}
                    if use_wandb is True and local_rank == 0:
                        wandb.log({'loss': loss_all/step, 'exact_match':metrics['exact_match'],'f1':metrics['f1']}) # 'rouge1': result['rouge1']})
                    if best_acc < metrics['f1']:
                        save_model(model,model_name)
                        best_acc = metrics['exact_match']
            
            # 按epoch存final gate weight
            # for i, layer in enumerate(model.bert.encoder.layer):
            #     # current_weight = layer.moe_linear.gate.gate.weight.data.cpu().numpy()  # 转换Tensor到NumPy数组并移动到CPU
            #     current_weight = layer.moe_linear.gate.gate.weight.data.float().cpu()  # 移动到CPU
            #     # print(f'Epoch {epoch} - Layer {i} current_weight {current_weight}')
            #     filename = f"l4_smallbert/gate_layer_{i}_weights_epoch_{epoch}.txt"  # 创建文件名
            #     with open(filename, 'w') as file:
            #         np.savetxt(file, current_weight, fmt='%.18e')  # 写入权重到文件
            #     print(f'Epoch {epoch} - Layer {i} current_weight written to {filename}')
            
            
            #  存每個epoch的final gate weight
            # for i, layer in enumerate(model.bert.encoder.layer):
            #     filename = f"normalized_diff/layer_50_gate_weight1/gate_layer_{i}_weights_epoch_{epoch}.txt"  # 创建文件名
            #     with open(filename, 'w') as file:
            #         np.savetxt(file, current_weights[i], fmt='%.18e')  # 写入权重到文件
            #     print(f'Epoch {epoch} - Layer {i} current_weight written to {filename}')
            if get_final_gate == False:
                # if epoch % 2 == 0:
                for i, layer in enumerate(model.bert.encoder.layer):
                    with open(f'normalized_diff/L2_*50_tiny_bert/layer_{i}_{epoch}_normalized_diff.txt', 'w') as f:
                        f.write(f"{normalized_diffs[i]}") 
                        print(f"save the normalzied diff")
                    with open(f'normalized_diff/L2_*50_tiny_bert_current_diff/layer_{i}_{epoch}_current_diff.txt', 'w') as f:
                        f.write(f"{current_diffs[i]}") 
                        print(f"save the current diff")
       
        # file_loss = f"loss_curve.txt"
        # with open(file_loss, 'w') as file:
        #     # 将专家计数结果追加到文件中
        #     file.write(f"{loss_values}\n")      
        for i, layer in enumerate(model.bert.encoder.layer):
            with open(f'normalized_diff/L2_*50_tiny_bert_current_diff/layer_{i}_{epoch}_reciprocal.txt', 'w') as f:
                f.write(f"{reciprocal[i]}") 
                print(f"save the reciprocal")
            with open(f'normalized_diff/L2_*50_tiny_bert_current_diff/layer_{i}_{epoch}_exponential_decay.txt', 'w') as f:
                f.write(f"{exponential_decay[i]}") 
                print(f"save the exponential_decay")
            with open(f'normalized_diff/L2_*50_tiny_bert_current_diff/layer_{i}_{epoch}_layer_convergence_rates.txt', 'w') as f:
                f.write(f"{layer_convergence_rates[i]}") 
                print(f"save the layer_convergence_rates")
        
            
    except KeyboardInterrupt:
        if use_wandb is True and local_rank == 0:
            wandb.finish()
            logger('Exiting from training early')
    if use_wandb is True and local_rank == 0:
        wandb.finish()
    del model
    del datasets
    del tokenizer

    
        
    # # 训练结束后，将每层的gate权重变化保存到文件
    # for layer_idx, changes in gate_weight_changes.items():
    #     file_path = f"{layer_idx}_gate_weight_changes.txt"
    #     with open(file_path, 'w') as file:
    #         for change in changes:
    #             file.write(f"{change}\n")
                                
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
    from transformers import DataCollatorWithPadding
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    dataset = load_dataset("samsum")
    metric = evaluate.load("rouge")

    def preprocess_function(examples):
        # inputs = [doc for doc in examples['dialogue']]
        model_inputs = tokenizer(examples['dialogue'], padding="max_length", max_length=1024, truncation=True)
        
        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples["summary"], padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    
    batch_size=train_batch_size
    # dataset = load_dataset("yelp_review_full")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = dataset.map(preprocess_function,  batched=True)
    tokenized_datasets.set_format("torch")
    # tokenized_datasets = tokenized_datasets.remove_columns(["valid"])
    tokenized_datasets = tokenized_datasets.remove_columns(["dialogue"])
    tokenized_datasets = tokenized_datasets.remove_columns(["id"])
    tokenized_datasets = tokenized_datasets.remove_columns(["summary"])
    
    train_dataset = tokenized_datasets["train"].shuffle(seed=42) # .select(range(1000))
    eval_dataset = tokenized_datasets["test"]# .shuffle(seed=42) # .select(range(1000))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=None,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=batch_size,
        sampler = datasampler
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-05,
                                betas=(0.9,0.999),
                                eps=1e-08)
    num_epochs = num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # model.train()

    if local_rank == 0:
        progress_bar = tqdm(range(num_training_steps))
    best_acc = 0
    model_name='gpt'

    if use_wandb is True and local_rank == 0:
        wandb.init(    # set the wandb project where this run will be logged
        project="moe",
        name='moe-gpt2-samsum-gpu-4',
        settings=wandb.Settings(
        _stats_sample_rate_seconds=0.1,
        _stats_samples_to_average=2,
        ),
        # track hyperparameters and run metadata
        config={
        "learning_rate": 5e-05,
        "architecture": "gpt2",
        "dataset": "samsum",
        "epochs": 1,
        }
        )

    # ddp
    if dist:
        model = DDP(model, device_ids=[local_rank], moe_sync_group = moe_sync_group)
        model._sync_params()
    try:
        for epoch in range(num_epochs):
            model.train()
            step = 0
            loss_all = 0
            elapsed_all = 0
            loss_log = 0
            elapsed_log = 0
            throttling_costs = 0
            comm_costs = 0
            for batch in train_dataloader:
                # break
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_start = time.time()
                outputs = model(**batch, training_step = step)
                loss = outputs.loss
                loss.backward()
                loss_all += loss.item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                elapsed_all += time.time() - batch_start
                step += 1
                if use_wandb is True and local_rank == 0:
                    wandb.log({'batch_loss': loss_all/step})
                    # wandb.log({'batch_loss': loss_all})
                # break
                throttling_costs += outputs.total_throttling_costs
                comm_costs += outputs.total_comm_costs
                if local_rank == 0:
                    progress_bar.set_description('Epoch {} | Loss {:.2f} | acc {:.2f} | mean batch time {:.2f}, mean throttling time {:.2f}, mean comm time {:.2f}'.format(
                                                epoch, (loss_all/step), best_acc, (elapsed_all/step)*1000, (throttling_costs/step)*1000, (comm_costs/step)*1000) )
                    progress_bar.update(1)
                torch.cuda.empty_cache()
            # dict_router = {}
            # index = 0
                if step % eval_interval == 0:
                    model.eval()
                    for idx, batch in enumerate(eval_dataloader):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        with torch.no_grad():
                            if dist:
                                outputs = model.module.generate(batch['input_ids'])# (**batch)
                            else:
                                outputs = model.generate(batch['input_ids'])# (**batch)
                            # outputs = model(**batch)
                        # logits = outputs.logits
                        # predictions = torch.argmax(logits, dim=-1)
                        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                        if idx >= 10:
                            break
                    result = metric.compute()

                    if use_wandb is True and local_rank == 0:
                        wandb.log({'loss': loss_all/step, 'rouge1': result['rouge1']})
                    if best_acc < result['rouge1']:
                        save_model(model,model_name)
                        best_acc = result['rouge1']
    except KeyboardInterrupt:
        if use_wandb is True and local_rank == 0:
            wandb.finish()
            logger('Exiting from training early')

    if use_wandb is True and local_rank == 0:
        wandb.finish()
    del model
    del dataset
    del tokenizer
