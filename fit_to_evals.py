import chess
from functions import *
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
	torch.set_printoptions(threshold=10000, linewidth=1000, precision=4)

	# stockfish = chess.uci.popen_engine('/usr/bin/stockfish')
	# stockfish.uci()

	net = nn.Sequential(
		# nn.BatchNorm1d(12),
		Reshape(-1, 768),
		nn.Linear(768, 768),
		nn.ReLU(),
		nn.Linear(768, 768),
		nn.ReLU(),
		nn.Linear(768, 3),
		nn.Softmax(dim=1)
	)

	if torch.cuda.is_available():
		net = net.cuda()

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

	predictions = []
	labels = []
	losses = []

	for game, x, y0, y1 in fen_generator('sources/ccrl.pgn', 100):
		# training
		y_pred = net(x)
		loss = loss_fn(y_pred, y0)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print(y0)

		# graphing
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

	# testing with starting position !!
	board = chess.Board()
	batch = get_x(board).unsqueeze(0)
	output = net(batch)
	print(output.data[0])

	# checkmate for black
	board.push_san('f4')
	board.push_san('e5')
	board.push_san('g4')
	batch = get_x(board).unsqueeze(0)
	output = net(batch)
	print(output.data[0])
