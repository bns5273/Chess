import math
import re
import pprint
import numpy as np


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


# this converts an integer to a bit array for qualitative tensors
# pretty fast
def to_bin_array(p, w, b):
    # convert integers to length 64 strings. mask pieces by team
    w = bin(w & p)[2:]
    w = '0' * (64 - len(w)) + w
    b = bin(b & p)[2:]
    b = '0' * (64 - len(b)) + b

    # conjoin. white + black -
    out = []
    for white, black, in zip(w, b):
        out.append(int(white) - int(black))
    return out


# creates an input tensor
def get_x(node):
    board = node.board()
    turn = board.turn
    castling = board.castling_rights
    enpassant = board.ep_square
    halfmove = board.halfmove_clock
    fullmove = board.fullmove_number
    white, black = board.occupied_co

    out = [to_bin_array(board.pawns, white, black),
           to_bin_array(board.knights, white, black),
           to_bin_array(board.bishops, white, black),
           to_bin_array(board.rooks, white, black),
           to_bin_array(board.queens, white, black),
           to_bin_array(board.kings, white, black)]

    # print(out[0])

    return out
