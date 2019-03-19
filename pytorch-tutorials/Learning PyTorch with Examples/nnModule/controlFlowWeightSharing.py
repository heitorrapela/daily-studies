import random
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

class DynamicNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(DynamicNet,self).__init__()
		self.input_linear = torch.nn.Linear(D_in, H)
		self.middle_linear = torch.nn.Linear(H, H)
		self.output_linear = torch.nn.Linear(H, D_out)

	def forward(self, x):
		h_relu = self.input_linear(x).clamp(min=0)
		for _ in range(random.randint(0,3)):
			h_relu = self.middle_linear(h_relu).clamp(min=0)
		y_pred = self.output_linear(h_relu)
		return y_pred

# N is the batch size; D_in is the input dimension
# H is the hidden dimension; D_out is the output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output
x = Variable(torch.randn(N, D_in).type(dtype))
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Construct model
model = DynamicNet(D_in, H, D_out)

# Criterion and optimizer
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(500):
	y_pred = model(x)

	loss = criterion(y_pred, y)
	print(t, loss.data[0])

	# Zero grad, update backward and step
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()