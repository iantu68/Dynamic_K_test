import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm

# 加载 SST-2 数据集
print("Loading SST-2 dataset...")
dataset = load_dataset('glue', 'sst2')
print("Done!")

# 加载 prajjwal1/bert-tiny 和分词器
print("Loading tokenizer and model...")
model_name = 'prajjwal1/bert-tiny'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("Done!")

# 预处理数据
def preprocess_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

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
        "labels": torch.tensor(example_batch["label"]),
    }

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
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
    
    accuracies.append(eval_result['eval_accuracy'])

# 保存精度到文件
print("Saving accuracies to accuracy.npy...")
np.save('accuracy.npy', accuracies)
print("Done!")

# 打印最大精度
max_accuracy = max(accuracies)
print(f"Max accuracy: {max_accuracy}")
