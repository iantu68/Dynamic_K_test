import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm

# 加载 SQuAD 数据集
print("Loading SQuAD dataset...")
dataset = load_dataset('squad')
print("Done!")

# 加载 prajjwal1/bert-tiny 和分词器
print("Loading tokenizer and model...")
model_name = 'prajjwal1/bert-tiny'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)
print("Done!")

# 预处理数据
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is out of the span (in the case of truncation)
            if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx < len(offsets) and offsets[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= 0 and offsets[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# 应用预处理函数
print("Preprocessing training data...")
train_dataset = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
print("Done!")
print("Preprocessing validation data...")
validation_dataset = dataset["validation"].map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)
print("Done!")

# 将数据集转换为 torch tensor
def convert_to_features(example_batch):
    return {
        "input_ids": torch.tensor(example_batch["input_ids"]),
        "attention_mask": torch.tensor(example_batch["attention_mask"]),
        "start_positions": torch.tensor(example_batch["start_positions"]),
        "end_positions": torch.tensor(example_batch["end_positions"]),
    }

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])
validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions"])

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # 我们将手动循环进行100次训练
    weight_decay=0.01,
    save_strategy="no",
    save_total_limit=1,
)

# 创建 Trainer 对象
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset
)
print("Done!")

# 初始化一个数组来存储每个epoch的精度
accuracies = []

# 开始训练和评估100个epoch
for epoch in tqdm(range(100), desc="Training Epochs"):
    print(f"Start training epoch {epoch + 1}...")
    trainer.train()
    print("Training done!")
    
    print(f"Start evaluation epoch {epoch + 1}...")
    eval_result = trainer.evaluate()
    print("Evaluation done!")
    
    accuracies.append(eval_result['eval_exact_match'])

# 保存精度到文件
print("Saving accuracies to accuracy.npy...")
np.save('accuracy.npy', accuracies)
print("Done!")

# 打印最大精度
max_accuracy = max(accuracies)
print(f"Max accuracy: {max_accuracy}")
