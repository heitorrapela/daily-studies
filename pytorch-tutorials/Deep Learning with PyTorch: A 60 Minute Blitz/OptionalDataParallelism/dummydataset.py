import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

class RandomDataset(Dataset):

	def __init__(self, size, length):
		self.len = length
		self.data = torch.randn(length, size)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return self.len

class Model(nn.Module):

	def __init__(self, input_size, output_size):
		super(Model,self).__init__()

	def forward(self, input):
		output = self.fc(input)
		print(" In Model: input size", input.size(), "output size", output.size())
		return output


rand_loader = DataLoader(dataset=RandomDataset(input_size, 100), 
						batch_size=batch_size, shuffle=True)

model = Model(input_size, output_size)

if torch.cuda.device_count() > 1:
	print("Lets use", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)

if torch.cuda.is_available():
	model.cuda()
for data in rand_loader:
	if torch.cuda.is_available():
		input_var = Variable(data.cuda())
	else:
		input_var = Variable(data)
	output = model(input_var)
	print("Outside: input size", input_var.size(),"output_size", output.size())