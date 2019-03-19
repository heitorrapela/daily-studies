import torch
from torch.autograd import Variable

dtype = torch.FloatTensors

# N is the batch size; D_in is the input dimension
# H is the hidden dimension; D_out is the output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and output
x = Variable(torch.randn(N, D_in).type(dtype))
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

# nn package
model = torch.nn.Sequential(
	torch.nn.Linear(D_in, H),
	torch.nn.ReLu(),
	torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=False)

## optm
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
	y_pred = model(x)

	loss = loss_fn(y_pred,y)
	print(t, loss.data[0])

	optimizer.zero_grad()

	loss.backward()

	optimizer.step()