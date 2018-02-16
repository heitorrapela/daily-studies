import torch

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor

# N is the batch size; D_in is the input dimension
# H is hidden dimension; D_out is the output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N,D_in).type(dtype)
y = torch.randn(N,D_out).type(dtype)

# Create random weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
	# Forward pass: compute predict y
	h = x.mm(w1)
	h_relu = h.clamp(min=0)
	y_pred = h_relu.mm(w2)

	# Compute and print loss
	loss = (y_pred - y).pow(2).sum()
	print(t,loss)

	# Backprop to compute gradients of w1 and w2 with respective loss
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.t().mm(grad_y_pred)
	grad_h_relu = grad_y_pred.mm(w2.t())
	grad_h = grad_h_relu.clone()
	grad_h[h < 0] = 0
	grad_w1 = x.t().mm(grad_h)

	# Update weights using gradient descent
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2