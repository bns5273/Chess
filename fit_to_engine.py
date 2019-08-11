# some issues with fitting to engine evaluations from the ccrl pgn:
# 1. positions are all too complex to properly train from
# 2. actual model decisions are not judged at all
# 3. only 1 move is judged from each position

# this model plays against itself, and after every move fits to the stockfish evaluations
# for each legal move

import chess
import chess.pgn
import chess.engine
import torch
import torch.nn
from functions import get_x, get_result_mse, Reshape
import matplotlib.pyplot as plt
from joblib import load
import numpy as np


def select_move(y, turn):
	factor = [[-1, 0, 1], [1, 0, -1]]
	matrix = np.dot(y.data, factor[turn])
	return np.argmax(matrix)


if __name__ == '__main__':
	model = torch.nn.Sequential(
		Reshape(-1, 768),
		torch.nn.Linear(768, 768),
		torch.nn.ReLU(),
		torch.nn.Linear(768, 768),
		torch.nn.ReLU(),
		torch.nn.Linear(768, 3),
		torch.nn.Softmax(dim=1)
	)
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters())
	stockfish = chess.engine.SimpleEngine.popen_uci('sources/stockfish.exe')
	stockfish_normalizer = load('sources/new_cp_to_prob_model.joblib')

	losses = []
	for game in range(10050):
		board = chess.Board()
		while not board.is_game_over():
			moves = list(board.generate_legal_moves())
			response = stockfish.analyse(board, limit=chess.engine.Limit(depth=5), multipv=len(moves))
			
			# create stack tensors
			x = []
			y = []
			for move in response:
				pv = move['pv'][0]	# 'pv' is a single element list
				score = move['score'].white().score(mate_score=327) / 100.
				predict = stockfish_normalizer.predict_proba([[len(board.move_stack), score]])	# should probs due this with 1 call
				y.append(torch.Tensor(predict))

				board.push(pv)
				x.append(get_x(board))
				board.pop()
			x = torch.stack(x)
			y = torch.stack(y)

			# model makes move
			y_pred = model(x)
			move_index = select_move(y_pred, board.turn)
			board.push(response[move_index]['pv'][0])
			# print(board, end='\n\n')

			# back propogation
			loss = loss_fn(y_pred, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# graphing
			losses.append(loss.item())
			if len(losses) % 100 == 0:
				x_axis = range(len(losses))
				plt.scatter(x_axis, losses, s=1)

				# calc the trend line
				plt.plot(np.convolve(losses, np.ones((50,))/50, mode='valid'), color='--r');
				# plt.show()	# main thread is not in main loop error

		game = chess.pgn.Game.from_board(board)
		print(game, file=open('sources/games.pgn', 'a'), end='\n\n')

	plt.show()

	stockfish.close()
	exit()
