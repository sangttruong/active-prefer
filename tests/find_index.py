import torch

# Assuming batch['input_ids'] is your PyTorch tensor
batch_input_ids = torch.tensor([[1, 2, 3, 4, 2, 5],
                                [2, 2, 3, 4, 5, 5],
                                [1, 1, 2, 5, 5, 5]])

# Calculate the tensor expression
result_tensor = (batch_input_ids != 2).sum(-1) - 1

# Convert the tensor to a list
result_list = result_tensor.tolist()

print(result_list)
