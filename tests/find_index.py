import torch

# Assuming your PyTorch tensor is named 'tensor'
tensor = torch.tensor([[1, 2, 3, 4, 2, 5], 
                       [2, 2, 3, 4, 5, 5], 
                       [1, 1, 2, 5, 5, 5]])

# Find indices where value is equal to 2
# print(tensor.shape)
# indices = (tensor!= 5).sum(-1)  - 1

# if len(indices) > 0:
#     breakpoint()
#     print(indices)
#     first_index = indices[0].tolist()
#     print("First index:", first_index)
# else:
#     print("Element 2 not found in the tensor.")

tensor = torch.tensor([[1, 2, 3, 4, 2, 5], 
                       [2, 2, 3, 4, 5, 5], 
                       [1, 1, 2, 5, 5, 5]])
cols = [3, 4, 2]
res = tensor[:, cols]
print(res)
