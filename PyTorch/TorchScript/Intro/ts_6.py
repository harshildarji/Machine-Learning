import torch

class MyDecisionGate(torch.nn.Module):
	def forward(self, x):
		if x.sum() > 0:
			return x
		else:
			return -x

scripted_gate = torch.jit.script(MyDecisionGate())

class MyCell(torch.nn.Module):
	def __init__(self, dg):
		super(MyCell, self).__init__()
		self.dg = dg
		self.linear = torch.nn.Linear(4, 4)

	def forward(self, x, h):
		new_h = torch.tanh(self.dg(self.linear(x)) + h)
		return new_h, new_h

my_cell = MyCell(scripted_gate)
traced_cell = torch.jit.script(my_cell)
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell(x, h)

class MyRNNLoop(torch.nn.Module):
	def __init__(self):
		super(MyRNNLoop, self).__init__()
		self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

	def forward(self, xs):
		h, y = torch.zeros(3, 4), torch.zeros(3, 4)
		for i in range(xs.size(0)):
			y, h = self.cell(xs[i], h)
		return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)

class WrapRNN(torch.nn.Module):
	def __init__(self):
		super(WrapRNN, self).__init__()
		self.loop = torch.jit.script(MyRNNLoop())

	def forward(self, xs):
		y, h = self.loop(xs)
		return torch.relu(y)

traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)

traced.save('wrapped_rnn.zip')

loaded = torch.jit.load('wrapped_rnn.zip')

print(loaded)
print(loaded.code)