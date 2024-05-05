import numpy as np
import torch
def load_final_gate_weights(num_layers, device='cpu'):
        final_gate_weights = {}
        for i in range(num_layers):
            filename = f"normalized_diff/layer_10_gate_weight/gate_layer_{i}_weights_epoch_9.txt"
            # 使用numpy读取数据，假设权重是以高精度格式保存的
            with open(filename, 'r') as f:
                weight_array = np.loadtxt(f, dtype=np.float32)  # 加载为float64以保持精度
                weight_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device)  # 转换为Tensor，使用float64
            final_gate_weights[i] = weight_tensor
            
            # 打印权重，保持原始精度
            # print(np.array2string(weight_tensor.numpy(), formatter={'float_kind':'{:.18e}'.format}))
        return final_gate_weights
def load_final_gate_weights1(num_layers, device='cpu'):
        final_gate_weights = {}
        for i in range(num_layers):
            filename = f"normalized_diff/layer_10_gate_weight1/gate_layer_{i}_weights_epoch_9.txt"
            # 使用numpy读取数据，假设权重是以高精度格式保存的
            with open(filename, 'r') as f:
                weight_array = np.loadtxt(f, dtype=np.float32)  # 加载为float64以保持精度
                weight_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device)  # 转换为Tensor，使用float64
            final_gate_weights[i] = weight_tensor
            
            # 打印权重，保持原始精度
            # print(np.array2string(weight_tensor.numpy(), formatter={'float_kind':'{:.18e}'.format}))
        return final_gate_weights
    
    # 在训练开始之前，记录每一层gate的初始权重状态
    
final_gate_weights = load_final_gate_weights(2)
final_gate_weights1 = load_final_gate_weights1(2)

final_diff = {i: torch.norm(final_gate_weights[i] - final_gate_weights1[i]).item() for i in range(2)}    # 存储权重变化数据的列表
print(final_diff)