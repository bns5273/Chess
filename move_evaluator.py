"""
the engine evaluations of a chess position are best described as a rate of change. if the evaluation is positive,
the board generally will look even more favourable as the game progresses.

so instead of fitting to the engines' evaluations of each position,
I will instead use the evaluations as a loss function.

"""

from functions import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


def loss_fn(b, c):
	a = torch.FloatTensor([.346, .257, .397])  # from CCRL
	a = b.sub(a)    # b - a
	b = c.sub(a)    # c - a


if __name__ == '__main__':
	torch.set_printoptions(threshold=10000, linewidth=1000, precision=4)

	# stockfish = chess.uci.popen_engine('/usr/bin/stockfish')
	# stockfish.uci()

	net = nn.Sequential(
		Reshape(-1, 768),
		nn.Linear(768, 7680),
		nn.ReLU(),
		nn.Linear(7680, 768),
		nn.ReLU(),
		nn.Linear(768, 3),
		nn.Softmax(dim=0)
	)

	if torch.cuda.is_available():
		net = net.cuda()

	optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

	predictions = []
	labels = []
	losses = []

	for game, x, y0, y1 in fen_generator('sources/ccrl.pgn', 1000):
		# training
		y_pred = net(x)
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
