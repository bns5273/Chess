"""
TODO:
black as negative will not work! at least for pawns
add result of matches as target
dimensionality reduction. LSI?
batching
automated rating estimation using stockfish binary?

instead of fitting to engine evaluations:
    value of material
    value of material by square
    guess the turn # from fen
"""

from functions import *
from numpy import mean, corrcoef
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# class for reshaping within sequential net
class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


if __name__ == '__main__':

    # stockfish = chess.uci.popen_engine('/usr/bin/stockfish')
    # stockfish.uci()

    net = nn.Sequential(
        Reshape(-1),
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.Linear(384, 384),
        nn.ReLU(),
        nn.Linear(384, 3),
        nn.Softmax(dim=0)
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    predictions = []
    labels = []
    for game, move, x, y in fen_generator('sources/ccrl.pgn'):
        y_pred = net(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # predictions.append(y_pred.item())
        # labels.append(y.item())

        # if game != 0 and game % 1 == 0 and move == 20:
        if move == 20:
            print(y)
