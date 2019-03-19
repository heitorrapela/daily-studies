import torch.nn as nn

class DataParallelModel(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.block1 = nn.Linear(10,20)

		# block 2 in dataparallel
		self.block2 = nn.Linear(20,20)
		self.block2 = nn.DataParallel(self.block2)

		self.block2 = nn.Linear(20,20)

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		return x

class DistributedModel(nn.Module):

	def __init__(self):
		super().__init__(
			embedding=nn.Embedding(1000,10),
			rnn=nn.Linear(10,10).cuda(0),
		)

	def forward(self, x):
		# compute embedding on CPU
		x = self.embedding(x)

		# transfer to GPU
		x = x.cuda(0)

		# compute RNN ON GPU
		x = self.rnn(x)
		return x