
from functions import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


# class for reshaping within sequential net
class Reshape(nn.Module):
	def __init__(self, *shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, input):
		return input.view(self.shape)


if __name__ == '__main__':
	torch.set_printoptions(threshold=10000, linewidth=1000, precision=4)

	# stockfish = chess.uci.popen_engine('/usr/bin/stockfish')
	# stockfish.uci()

	net = nn.Sequential(
		nn.BatchNorm1d(12),
		Reshape(-1, 768),
		nn.Linear(768, 7680),
		nn.ReLU(),
		nn.Linear(7680, 3),
		nn.Softmax(dim=0)
	).cuda()

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

	predictions = []
	labels = []
	losses = []
	for game, x0, x1, y0, y1 in fen_generator('sources/ccrl.pgn', 1000):
		# training
		y_pred = net(x1)
		loss = loss_fn(y_pred, y0)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())

		if len(losses) % 100 == 0:
			x_axis = range(len(losses))
			plt.scatter(x_axis, losses, s=1)

			# calc the trend line
			z = np.polyfit(x_axis, losses, 1)
			p = np.poly1d(z)
			plt.plot(x_axis, p(x_axis), "r--")
			plt.show()

			print(game, 'games, slope:', z[0])
