import chess
from chess import uci

engine = uci.popen_engine("sources/stockfish")
engine.uci()
print(engine.name)

engine.setoption({'Skill Level': 16})
board = chess.Board()


def tree(board):
	# engine.position(board)
	# move, ponder = engine.go(movetime=100)
	if board.fullmove_number > 1:
		return
	else:
		moves = board.generate_legal_moves()
		for m in moves:
			b = chess.Board(board.fen())
			b.push(m)
			print(b, '\n')
			tree(b)


tree(board)
