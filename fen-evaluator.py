"""
TODO:
black as negative will not work! at least for pawns
add result of matches as target
dimensionality reduction. LSI?
batching
automated rating estimation using stockfish binary?

ideas:
    fit to engine evaluations
    value of material
    value of material by square, turn, in-play pieces
    guess the turn # from fen
    personalized training tool
"""

from functions import *
import torch
import torch.nn as nn
from numpy import corrcoef
import matplotlib.pyplot as plt


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
    losses = []
    for game, move, x, y0, y1 in fen_generator('sources/ccrl.pgn', 1000):
        y_pred = net(x)
        loss = loss_fn(y_pred, y0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss.item())
        losses.append(loss.item())

        # if move == 15:
        #     print(game, y_pred.data[2], y0.data[2])
        if len(losses) % 500 == 0:
            plt.plot(range(len(losses)), losses)
            plt.show()
