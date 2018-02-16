# Custom nn Modules
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

class TwoLayerNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(TwoLayerNet, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear2 = torch.nn.Linear(H, D_out)

	def forward(self, x):
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		return y_pred

# N is the batch size; D_in is the input dimension
# H is the hidden dimension; D_out is the output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors data
x = Variable(torch.randn(N, D_in).type(dtype))
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Construct our model by instantiating the class defines above
model = TwoLayerNet(D_in, H, D_out)

# Optim
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for t in range(500):
	y_pred = model(x)

	# Compute and print loss
	loss = criterion(y_pred, y)
	print(t, loss.data[0])

	# Zero gradients, perform backward pass, and update weights
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()