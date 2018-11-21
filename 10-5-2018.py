"""
TODO:
black as negative will not work! at least for pawns
add 'mate in x' float values
add result of matches as target
dimensionality reduction. LSI?
batching
automated rating estimation using stockfish binary?

instead of fitting to engine evaluations:
    value of material
    value of material by square
    convert centipawns to percentage
"""

import chess
import chess.pgn as pgn
import chess.uci
from functions import *
import re
from numpy import mean, corrcoef
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math


def fen_generator(filename):
    with open(filename, 'r', encoding='Latin-1') as file:
        position = 0
        game = 0
        for i in range(100):
            move = 1
            old_node = pgn.read_game(file)
            while not old_node.is_end():
                node = old_node.variations[0]
                eval = re.findall('[-+]?\d*\.\d+', node.comment)
                # evaluation is empty for mating and book opening moves
                if eval:
                    # only white to move positions. some engines (as black) evaluate from perspective!
                    eval = float(eval[0])
                    if -5 < eval < 5 and move % 1 == 0:
                        yield game, move, get_x(node), torch.FloatTensor([eval])
                        position += 1
                old_node = node
                move += .5
            game += 1

            # if game % 1000 == 0:
            #     print(game / 916663)

        print(game, 'games', position, 'positions')


# class for reshaping within sequential net
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


if __name__ == '__main__':

    # stockfish = chess.uci.popen_engine('/usr/bin/stockfish')
    # stockfish.uci()

    net = nn.Sequential(
        View(-1),
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.ReLU(),
        nn.ReLU(),
        nn.ReLU(),
        nn.ReLU(),
        nn.ReLU(),
        nn.Linear(384, 1)
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.00001)

    performance = []
    predictions = []
    labels = []
    for game, move, x, y in fen_generator('sources/ccrl_stockfish.pgn'):
        y_pred = net(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if move == 20:
            print(game, y.item(), y_pred.item(), loss.item())
        performance.append(loss.item())
        predictions.append(y_pred.item())
        labels.append(y.item())

    print(corrcoef(predictions, labels)[0][1])

    plt.plot(performance)
    plt.show()
