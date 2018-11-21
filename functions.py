import math
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_result(h):
    if h['Result'] == '1-0':
        return 1
    elif h['Result'] == '0-1':
        return -1
    else:
        return 0


def winning_chances(centipawns):
    """
    Takes an evaluation in centipawns and returns an integer value estimating
    the chance the player to move will win the game
    winning chances = 50 + 50 * (2 / (1 + e^(-0.004 * centipawns)) - 1)
    """
    return 2 / (1 + math.exp(-0.004 * centipawns)) - 1


# creates an input tensor
def get_x(node):
    board = node.board()
    turn = board.turn
    castling = board.castling_rights
    enpassant = board.ep_square
    halfmove = board.halfmove_clock
    fullmove = board.fullmove_number
    white, black = board.occupied_co

    board_vars = vars(board)
    pieces = ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings']
    coordinates = [[], []]
    piece_colors = []
    for piece_counter in range(6):
        piece_value = board_vars[pieces[piece_counter]]             # integer
        square_counter = 0
        piece_locations = []
        bit_mask = 1
        while bit_mask < piece_value:
            if piece_value & bit_mask > 0:
                piece_locations.append(square_counter)
                if (piece_value & white & bit_mask) > 0:            # white's piece?
                    piece_colors.append(1)                                # white
                else:
                    piece_colors.append(-1)                               # black
            square_counter += 1
            bit_mask = bit_mask << 1
        coordinates[0] += [piece_counter]*len(piece_locations)      # row
        coordinates[1] += piece_locations                           # column
    # print(coordinates)
    # print(piece_colors)
    i = torch.LongTensor(coordinates)
    v = torch.FloatTensor(piece_colors)
    return torch.sparse.FloatTensor(i, v, torch.Size([6, 64])).to_dense()
