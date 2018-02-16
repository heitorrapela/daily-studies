# Neural Net using nn module
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

# N is batch size; D_in is input dimension
# H is the hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors tohold inputs and outpus
x = Variable(torch.randn(N, D_in).type(dtype))
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# nn.Sequential = sequential of layers
model = torch.nn.Sequential(
	torch.nn.Linear(D_in,H),
	torch.nn.ReLU(),
	torch.nn.Linear(H, D_out),
)

# nn package has definition of loss functions too
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4

for t in range(500):
	y_pred = model(x)

	loss = loss_fn(y_pred,y)

	# zero grads
	model.zero_grad()

	loss.backward()

	for param in model.parameters():
		param.data -= learning_rate * param.grad.data
