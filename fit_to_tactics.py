"""
maybe only train on the last move of each tactic? having multiple layers is messing up the evaluations
white to move vs black is causing issues
"""
import chess
from chess import pgn
from functions import *
import torch
from torch import nn
import matplotlib.pyplot as plt

torch.set_printoptions(threshold=10000, linewidth=1000, precision=4)
file = open('sources/ccrl.pgn', 'r', encoding='Latin-1')

net = nn.Sequential(
	Reshape(-1, 768),
	nn.Linear(768, 1152),
	nn.ReLU(),
	nn.Linear(1152, 1152),
	nn.ReLU(),
	nn.Linear(1152, 3),
	nn.Softmax(dim=1)
)

if torch.cuda.is_available():
	net = net.cuda()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters())

colors_src = ['blue', 'orange', 'green']

colors = []
losses = []
for game in range(1, 50000):
	root = pgn.read_game(file)
	node = root.end()

	x = get_x(node.board())
	y = get_result_mse(root.headers)

	# last position for white
	if not node.board_cached.turn:
		node.board_cached.pop()

	# training
	y_pred = net(x)
	loss = loss_fn(y_pred, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	losses.append(loss.item())
	colors.append(colors_src[get_result_simple(root.headers)])

	# graphing
	if game % 2000 == 0:
		x_axis = range(len(losses))
		plt.scatter(x_axis, losses, s=1, c=colors)

		# calc the trend line
		z = np.polyfit(x_axis, losses, 1)
		p = np.poly1d(z)
		plt.plot(x_axis, p(x_axis), "r--")
		print(p.coefficients)

		plt.title(game)
		plt.show()


# testing with starting position !!
board = chess.Board()
batch = get_x(board)
output = net(batch)
print(output.data[0])

# checkmate for black
board.push_san('f4')
board.push_san('e5')
board.push_san('g4')
board.push_san('Qh4')
batch = get_x(board)
output = net(batch)
print(output.data[0])
