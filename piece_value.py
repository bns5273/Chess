from functions import *
import matplotlib.pyplot as plt

next_game = 1
x_values = []
y_values = []
for game, move, x, y0, y1 in fen_generator('sources/ccrl.pgn', 10):
    if game == next_game:
        plt.title(game)
        plt.plot(x_values, y_values)
        plt.show()
        next_game += 1
        x_values = []
        y_values = []
    x_values.append(move)
    y_values.append([y0[0], y0[1], y0[2]])
