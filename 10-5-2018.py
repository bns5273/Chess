"""
TODO:
generators
black as negative will not work!
dimensionality reduction
automated rating estimation using stockfish binary?
"""

import chess
import chess.pgn as pgn
import chess.uci
from functions import *
import re
from numpy import mean, corrcoef
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def mygenerator(filename):
    file = open(filename, 'r', encoding='Latin-1')

    position = 0
    game = 0
    for offset, header in pgn.scan_headers(file):
        white = header['White']
        black = header['Black']
        result = get_result(header)

        move = 1
        x = []
        y = []
        old_node = pgn.read_game(file)
        while not old_node.is_end():
            node = old_node.variations[0]
            eval = re.findall('[-+]?\d*\.\d+', node.comment)
            # evaluation is empty for mating and book opening moves
            if eval:
                # only white to move positions. some engines (as black) evaluate from perspective!
                if move % 1 == 0:
                    yield get_x(node), float(eval[0])
                    position += 1
            old_node = node
            move += .5
        game += 1

        if game == 10:
            break

        if game % 1000 == 0:
            print(game / 916663)

    print(game, 'games', position, 'positions')


if __name__ == '__main__':

    # stockfish = chess.uci.popen_engine('/usr/bin/stockfish')
    # stockfish.uci()

    for x, y in mygenerator('sources/ccrl_stockfish.pgn'):
        print(y)



    # plt.plot(range(len(w_eval)), w_eval)
    # plt.plot(range(len(b_eval)), b_eval)
    # plt.title(result)
    # plt.ylim(-10, 10)
    # plt.show()


