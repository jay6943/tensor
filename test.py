import torch

list_data = [[10, 20], [30, 40]]

tensor1 = torch.Tensor(list_data)

print(tensor1)
print(f'type: {type(tensor1)}')
print(f'shape: {tensor1.shape}')
print(f'dtype: {tensor1.dtype}')
print(f'device: {tensor1.device}')
