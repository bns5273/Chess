import re
import torch
from torch import nn
import numpy as np
#import math
import chess.pgn as pgn
from eval_norm import EvalNorm
import math


# class for reshaping within sequential net
class Reshape(nn.Module):
	def __init__(self, *shape):
		super(Reshape, self).__init__()
		self.shape = shape

	def forward(self, input):
		return input.view(self.shape)


# for use training eval_norm
def get_result_simple(h):
	if h == '1-0':
		return 1
	elif h == '0-1':
		return -1
	else:
		return 0

# for cross entropy loss function
def get_result_ce(h):
	if h['Result'] == '1-0':
		return [2]
	elif h['Result'] == '0-1':
		return [0]
	else:
		return [1]


# for mse loss function
def get_result_mse(h):
	if h['Result'] == '1-0':
		return torch.FloatTensor([0, 0, 1])
	elif h['Result'] == '0-1':
		return torch.FloatTensor([1, 0, 0])
	else:
		return torch.FloatTensor([0, 1, 0])


# creates an input tensor
def get_x(board):
#	board = node.board()                # 768 bits, sparse, 2x6x8x8 -> 12x64 (team-piece x rank-file)
	board_teams = board.occupied_co     #
	turn = board.turn                   # 1 bit
	castling = board.castling_rights    # 4 bits
	enpassant = board.ep_square         # 6 bits OR 64 bits ??
	halfmove = board.halfmove_clock     #
	fullmove = board.fullmove_number    #

	board_vars = vars(board)
	pieces = ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings']

	team_pieces = []
	rank_files = []
	for team_piece in range(12):
		team = board_teams[team_piece % 2]           # bit mask (int)
		piece = board_vars[pieces[team_piece // 2]]  # int

		bit_mask = 1
		while bit_mask < piece:                      # remaining pieces?
			if piece & team & bit_mask > 0:          # piece?
				rank_files.append(math.log2(piece & bit_mask))
				team_pieces.append(team_piece)
			bit_mask = bit_mask << 1

	# if torch.cuda.is_available():
	# i = torch.cuda.LongTensor([team_pieces, rank_files])
	# v = torch.cuda.FloatTensor(np.ones(len(rank_files)))
	# return torch.cuda.sparse.FloatTensor(i, v, torch.Size([12, 64])).to_dense()
	# else:
	i = torch.LongTensor([team_pieces, rank_files])
	v = torch.FloatTensor(np.ones(len(rank_files)))
	return torch.sparse.FloatTensor(i, v, torch.Size([12, 64])).to_dense()


# game #, move #, x (node), y0 (eval), y1 (result)
# [loss, draw, win]
def fen_generator(filename, limit):
	get_y = EvalNorm().get_y        # sklearn model: centi-pawn to result probability vector
	with open(filename, 'r', encoding='Latin-1') as file:
		position = 0
		game = 0
		for i in range(limit):
			x0 = []
			y0 = []
			y1 = []

			move = 1
			old_node = pgn.read_game(file)
			result = get_result_ce(old_node.headers)
			while not old_node.is_end():
				node = old_node.variations[0]
				# only white to move positions. some engines (as black) evaluate from perspective!
				if move % 1 == 0:
					eval = re.findall('[-+]\d*\.\d*', node.comment)
					if eval:
						x0.append(get_x(node.board()))
						y0.append(get_y(float(eval[0])))
						y1.append(result)
						position += 1
					elif re.findall('[-]', node.comment):       # black mate in ...
						x0.append(get_x(node.board()))
						# cuda
						eval = torch.FloatTensor([1, 0, 0])
						y0.append(eval)
						y1.append(result)
						position += 1
					elif re.findall('[+]', node.comment):       # white mate in ...
						x0.append(get_x(node.board()))
						# cuda
						eval = torch.FloatTensor([0, 0, 1])
						y0.append(eval)
						y1.append(result)
						position += 1
				old_node = node
				move += .5
			game += 1

			# cuda
			yield game, torch.stack(x0), torch.stack(y0), torch.FloatTensor(y1)

		print(game, 'games', position, 'positions')


# test case for get_x
if __name__ == '__main__':
	with open('sources/ccrl.pgn', 'r', encoding='Latin-1') as file:
		old_node = pgn.read_game(file)
		x = get_x(old_node)
		print(x)
