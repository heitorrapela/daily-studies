# Neural Network with PyTorch
import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
# Run in GPU
# dtype = torch.cuda.FloatTensor

# N is the batch size: D_in is input dimension
# H is hidden dimensio: D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random tensors to hold input and output
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random weights
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6

for t in range(500):
	y_pred = x.mm(w1).clamp(min=0).mm(w2)
	loss = (y_pred - y).pow(2).sum()
	print(t, loss.data[0])

	loss.backward()

	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data

	# Manually zero the gradientes after updating weights
	w1.grad.data.zero_()
	w2.grad.data.zero_()
