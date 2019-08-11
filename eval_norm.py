from functions import *
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from chess import pgn
import re
from chess.engine import Cp, Mate


# converts centipawns to [white, draw, black] probabilities
class EvalNorm:
	def __init__(self):
		self.clf = load('sources/new_cp_to_prob_model.joblib')

	def get_y(self, x):
		y = self.clf.predict_proba(x)
		return torch.FloatTensor(y[0])


def pgn_generator():
	with open('sources/ccrl_stockfish.pgn', 'r', encoding='Latin-1') as file:
		game = pgn.read_game(file)
		while game is not None:
			node = game.variations[0]
			move = 1
			if 'Stockfish' in game.headers['Black']:
				node = node.variations[0]
				move += 1
			while not node.is_end():
				comment = node.comment
				
				score = re.findall(' [-+]\d*.\d*', comment)
				depth = re.findall('/\d*', comment)
				result = get_result_simple(game.headers['Result'])

				if score and depth:
					yield move, float(score[0]), result

				# we are skipping every other move 
				node = node.variations[0]
				if node.is_end():
					break
				else:
					node = node.variations[0]
				move += 2
			game = pgn.read_game(file)


if __name__ == '__main__':
	x = []
	y = []
	for turn, score, result in pgn_generator():
		x.append([turn, score])
		y.append(result)

	print(len(x))

	model = LogisticRegression()
	model = model.fit(x, y)

	print(model.predict_proba([[55, -.125], [20, .55], [20, -.55]]))
	dump(model, 'sources/new_cp_to_prob_model.joblib')
