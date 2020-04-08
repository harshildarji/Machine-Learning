import torch

class MyCell(torch.nn.Module):
	def __init__(self):
		super(MyCell, self).__init__()

	def forward(self, x, h):
		new_h = torch.tanh(x + h)
		return new_h, new_h

my_cell = MyCell()
x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))