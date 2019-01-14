"""
I would like to simulate thousands of games at once using batching
"""

from functions import *
import chess.pgn


net = nn.Sequential(
	Reshape(-1, 768),
	nn.Linear(768, 768),
	nn.ReLU(),
	nn.Linear(768, 768),
	nn.ReLU(),
	nn.Softmax(dim=1)
)

if torch.cuda.is_available():
	net = net.cuda()

loss_fn = nn.CrossEntropyLoss()     # uneven dataset, use weight!
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

if __name__ == '__main__':
	for g in range(10):
		board = chess.Board()

		while not board.is_game_over():
			# get legal moves
			children = []
			moves = list(board.generate_legal_moves())
			for move in moves:
				child = chess.Board(board.fen())
				child.push(move)
				children.append(get_x(child))

			# evaluate moves
			x = torch.stack(children)
			y_pred = net(x)

			to_move = board.turn * 2    # side to move. 0,1 -> 0,2
			scores = [float(x[to_move]) for x in y_pred.data]
			# print(scores)

			# move
			choice = scores.index(max(scores))
			board.push(moves[choice])
			# print(board)
		game = chess.pgn.Game()
		game = game.from_board(board)
		print(game)
