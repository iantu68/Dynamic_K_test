# import libraries
import torch
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# from tensorflow.keras.callbacks import ModelCheckpoint
from torchvision import datasets
import torchvision.transforms as transforms

# set environment
# device = torch.device('cuda' if args.cuda else 'cpu')
# device = torch.device('cuda:0')

# number of subprocesses to use for data loading
num_workers = 0
batch_size = 200

# 定义transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载完整的MNIST数据集
full_train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# 筛选训练集中标签为0和1的样本
def filter_01(dataset):
    targets = torch.tensor(dataset.targets)
    mask = (targets == 0) | (targets == 1)
    dataset.targets = targets[mask]
    dataset.data = dataset.data[mask]
    return dataset

filtered_train_data = filter_01(full_train_data)

# 获取完整训练集大小
full_train_size = len(full_train_data)

# 获取过滤后的训练集大小
filtered_train_size = len(filtered_train_data)

# 划分训练集和验证集
# 训练集只包含标签为0和1的样本
train_data = filtered_train_data

# 验证集包含所有样本
val_data = full_train_data

# 定义DataLoader
batch_size = 64
num_workers = 2

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# 验证训练数据集中是否只包含标签0和1
print(f"Labels in training batch: {labels.numpy()}")

# # plot the images in the batch, along with the corresponding labels
# fig = plt.figure(figsize=(25, 4))
# for idx in np.arange(20):
#     ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
#     ax.imshow(np.squeeze(images[idx]), cmap='gray')
#     # print out the correct label for each image
#     # .item() gets the value contained in a Tensor
#     ax.set_title(str(labels[idx].item()))

## TODO: Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear layer (784 -> 1 hidden node)
        self.fc0 = nn.Linear(28 * 28, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    #有\ dropout
    # def forward(self, x):
    #     # flatten image input
    #     x = x.view(-1, 28 * 28)
    #     # add hidden layer, with relu activation function
    #     x = self.dropout(F.relu(self.fc1(x)))
    #     x = self.dropout(F.relu(self.fc2(x)))
    #     x = self.fc3(x)
    #     return x
    
# initialize the NN
model = Net()
# 加载检查点文件
# checkpoint = torch.load("checkpoint_epoch_1000.pt")
# 加载模型参数
# model.load_state_dict(checkpoint)
print(model)

# # 獲取每一層的初始權重
# for name, param in model.named_parameters():
#     if 'weight' in name:
#         print(f'層名稱: {name}, 初始權重:\n{param.data}')


## TODO: Specify loss and optimization functions
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# number of epochs to train the model
n_epoch = 0
end_epoch = 30
checkpoint_interval = 500

# 初始化记录训练过程的数组
train_losses = []
val_losses = []
accuracies = []
norm_gradients = []
Layer_grad = [[] for i in range(3)]
total_parameters = 0
count = 0

for epoch in range(n_epoch, end_epoch):
    model.train() 
    train_loss = 0.0
    ###################
    # train the model #s
    ###################
    # print("here")
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # if epoch % 50 == 0:
    for name, param in model.named_parameters():
            # print(count)
            # time.sleep(1)
            for i in range(3):
                if "fc" + str(i) + ".weight" in name:
                    this_grads = param.grad.data.detach().norm().view(-1).cpu().numpy()
                    # print(f"grad_L{i} : ", this_grads)
                    eval(f"Layer_grad[{i}]").extend(this_grads)
    
    ###################
    #  valid the model #
    ###################
    model.eval() # prep model for *evaluation*
    val_loss = 0.0

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()*data.size(0)

    # 计算平均验证损失
    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    # print(f"Epoch: {n_epochs+1}/{end_epoch} | Train Loss: {train_loss:.6f} | Grads Value: {avg_gradient:.6f} | Val Loss: {val_loss:.6f}", end="\n")
    print(f"Epoch: {epoch+1}/{end_epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}", end="\n")

    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"  # 检查点文件名
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

# existing_train_losses = np.load("Train_Loss.npy")
# existing_valid_losses = np.load("Valid_Loss.npy")
# existing_Layer0_grads = np.load("Layer0_grads.npy")
# existing_Layer1_grads = np.load("Layer1_grads.npy")
# existing_Layer2_grads = np.load("Layer2_grads.npy")

# existing_train_losses = np.append(existing_train_losses, train_losses)
# existing_valid_losses = np.append(existing_valid_losses, val_losses)
# existing_Layer0_grads = np.append(existing_Layer0_grads, Layer_grad[0])
# existing_Layer1_grads = np.append(existing_Layer1_grads,  Layer_grad[1])
# existing_Layer2_grads = np.append(existing_Layer2_grads,  Layer_grad[2])


for i in range(3):
    np.save(f"Layer{i}_grads.npy", Layer_grad[i])  

np.save(f"Valid_Loss.npy", val_losses)
np.save(f"Train_Loss.npy", train_losses)


# # 準確度
# model.eval() # prep model for *evaluation*
# test_loss = 0.0
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))

# for data, target in test_loader:
#     output = model(data)
#     loss = criterion(output, target)
#     test_loss += loss.item()*data.size(0)
#     _, pred = torch.max(output, 1)
#     correct = np.squeeze(pred.eq(target.data.view_as(pred)))
#     for i in range(batch_size):
#         label = target.data[i]
#         class_correct[label] += correct[i].item()
#         class_total[label] += 1

# # calculate and print avg test loss
# test_loss = test_loss/len(test_loader.dataset)
# print('Test Loss: {:.6f}\n'.format(test_loss))

# for i in range(100):
#     if class_total[i] > 0:
#         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
#             str(i), 100 * class_correct[i] / class_total[i],
#             np.sum(class_correct[i]), np.sum(class_total[i])))
#     else:
#         print('Test Accuracy of %5s: N/A (no training examples)' % (str(i)), end=" | ")
