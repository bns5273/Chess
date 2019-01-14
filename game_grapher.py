from functions import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
	start = [.346, .257, .397]   # from CCRL

	for game, x, y0, y1 in fen_generator('sources/ccrl.pgn', 10):
		gy = []

		for i in list(y0.data):
			gy.append(tuple(i))
		plt.hlines(start, 0, len(y0), linestyles='--', colors=['blue', 'orange', 'green'])
		plt.plot(range(len(y0)), gy)
		plt.title(game)
		# plt.savefig('graphs/{0}'.format(game))
		plt.show()
