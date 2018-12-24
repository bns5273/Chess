import re
import torch
import numpy as np
import math
import chess.pgn as pgn
from eval_norm import EvalNorm


def get_result(h):
	if h['Result'] == '1-0':
		out = [0, 0, 1]
	elif h['Result'] == '0-1':
		out = [1, 0, 0]
	else:
		out = [0, 1, 0]

	if torch.cuda.is_available():
		return torch.cuda.FloatTensor(out)
	else:
		return torch.FloatTensor(out)


# creates an input tensor
def get_x(node):
	board = node.board()                # 768 bits, sparse, 2x6x8x8 -> 12x64 (team-piece x rank-file)
	board_teams = board.occupied_co     #
	turn = board.turn                   # 1 bit
	castling = board.castling_rights    # 4 bits
	enpassant = board.ep_square         # 64 bits, sparse, 1x64
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

	if torch.cuda.is_available():
		i = torch.cuda.LongTensor([team_pieces, rank_files])
		v = torch.cuda.FloatTensor(np.ones(len(rank_files)))
		return torch.cuda.sparse.FloatTensor(i, v, torch.Size([12, 64])).to_dense()
	else:
		i = torch.LongTensor([team_pieces, rank_files])
		v = torch.FloatTensor(np.ones(len(rank_files)))
		return torch.sparse.FloatTensor(i, v, torch.Size([12, 64])).to_dense()


# game #, move #, x (node), y0 (eval), y1 (result)
# [loss, draw, win]
def fen_generator(filename, limit):
	get_y = EvalNorm().get_y
	with open(filename, 'r', encoding='Latin-1') as file:
		position = 0
		game = 0
		for i in range(limit):
			xs = []
			ys = []

			move = 1
			old_node = pgn.read_game(file)
			result = get_result(old_node.headers)
			while not old_node.is_end():
				node = old_node.variations[0]
				# only white to move positions. some engines (as black) evaluate from perspective!
				if move % 1 == 0:
					eval = re.findall('[-+]\d*\.\d*', node.comment)
					if eval:
						xs.append(get_x(node))
						ys.append(get_y(float(eval[0])))
						position += 1
					elif re.findall('[-]', node.comment):
						xs.append(get_x(node))
						eval = torch.FloatTensor([1, 0, 0])
						ys.append(eval)
						position += 1
					elif re.findall('[+]', node.comment):
						xs.append(get_x(node))
						eval = torch.FloatTensor([0, 0, 1])
						ys.append(eval)
						position += 1
				old_node = node
				move += .5
			game += 1
			yield game, torch.stack(xs), torch.stack(ys), result

		print(game, 'games', position, 'positions')


# test case for get_x
if __name__ == '__main__':
	with open('sources/ccrl.pgn', 'r', encoding='Latin-1') as file:
		old_node = pgn.read_game(file)
		x = get_x(old_node)
		print(x)
