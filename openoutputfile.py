import torch

# Load the .pt file
data = torch.load("/home/shruti/Documents/FYP/mc_uav/outputs/op_coop20/km/attention_rollout_3agents_20240228T115151/epoch-0.pt")

# Now you can access the data stored in the file
# For example, if the file contains a tensor, you can access it like this:
tensor_data = data['agent_0']  # Replace 'tensor_key' with the key used to store the tensor in the file

# You can also directly load a model like this:
# model = torch.load("your_model.pt")

# Remember to replace "your_file.pt" with the path to your .pt file.
