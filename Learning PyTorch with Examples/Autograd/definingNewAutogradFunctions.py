import torch
from torch.autograd import Variable

class MyReLU(torch.autograd.Function):
	'''
	Custom autograd functions subclassing torch.autograd.Function
	and implement forward and backward pass
	'''

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return input.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		return grad_input

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor

# N is the batch size; D_in is the input dimension
# H is the hidden dimension; D_out is the output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random tensors
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# Create random weights
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
	relu = MyReLU.apply

	# Using custom relu function
	y_pred = relu(x.mm(w1)).mm(w2)

	# Compute and print Loss
	loss = (y_pred - y).pow(2).sum()
	print(t,loss.data[0])

	# Use autograd to compute backward pass
	loss.backward()

	# Update weights using gradient descnet
	w1.data -= learning_rate * w1.grad.data
	w2.data -= learning_rate * w2.grad.data

	# Manually zero the gradients after updating weights
	w1.grad.data.zero_()
	w2.grad.data.zero_()