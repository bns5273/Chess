import chess
import chess.pgn as pgn
import chess.uci
from functions import *
import re
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


if __name__ == '__main__':

    stockfish = chess.uci.popen_engine('stockfish_9_x64.exe')
    stockfish.uci()

    file = open('ccrl_stockfish.pgn', 'r', encoding='Latin-1')

    position = 0
    game = 0
    for offset, header in pgn.scan_headers(file):
        white = header['White']
        black = header['Black']
        result = header['Result']

        move = 1
        x = []
        y = []
        # b_eval = []
        node = pgn.read_game(file)
        while not node.is_end():
            next_node = node.variations[0]
            evaluation = re.findall('[-+]?\d*\.\d+', next_node.comment)
            if evaluation:          # evaluation is empty for mating and book opening moves
                if move % 1 == 0:   # only white to move positions. some engines (as black) evaluate from perspective!
                    x.append(get_x(next_node))
                    y.append(winning_chances(float(evaluation[0])))
                    position += 1

                print(next_node.board().fen(), y[-1])
            node = next_node
            move += .5
        # print(g, result, '\n')
        game += 1

        if game == 1:
            break

    print(game)
    print(position)

    # plt.plot(range(len(w_eval)), w_eval)
    # plt.plot(range(len(b_eval)), b_eval)
    # plt.title(result)
    # plt.ylim(-10, 10)
    # plt.show()

    if game % 1000 == 0:
        print(game / 916663)
