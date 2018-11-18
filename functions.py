import math
import re
import pprint
import numpy as np


def result_bool(h):
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


def get_x(node):
    """
    sample input:
    8/8/p3k3/P2p4/1p1p3p/1P1Pq2P/5N2/5R1K b - - 10 58
    """
    board_dict = {
        '1': [[0, 0, 0, 0, 0, 0]],
        '2': [[0, 0, 0, 0, 0, 0] for _ in range(2)],
        '3': [[0, 0, 0, 0, 0, 0] for _ in range(3)],
        '4': [[0, 0, 0, 0, 0, 0] for _ in range(4)],
        '5': [[0, 0, 0, 0, 0, 0] for _ in range(5)],
        '6': [[0, 0, 0, 0, 0, 0] for _ in range(6)],
        '7': [[0, 0, 0, 0, 0, 0] for _ in range(7)],
        '8': [[0, 0, 0, 0, 0, 0] for _ in range(8)],
        'P': [[1, 0, 0, 0, 0, 0]],
        'N': [[0, 1, 0, 0, 0, 0]],
        'B': [[0, 0, 1, 0, 0, 0]],
        'R': [[0, 0, 0, 1, 0, 0]],
        'Q': [[0, 0, 0, 0, 1, 0]],
        'K': [[0, 0, 0, 0, 0, 1]],
        'p': [[-1, 0, 0, 0, 0, 0]],
        'n': [[0, -1, 0, 0, 0, 0]],
        'b': [[0, 0, -1, 0, 0, 0]],
        'r': [[0, 0, 0, -1, 0, 0]],
        'q': [[0, 0, 0, 0, -1, 0]],
        'k': [[0, 0, 0, 0, 0, -1]],
    }
    castle_dict = {
        'k': [[]],
        'q': [[]],
        'K': [[]],
        'Q': [[]]
    }

    fen = node.board().fen()
    board, turn, castle, enpassant, halfmove, fullmove = str(fen).split()

    board = board.split('/')
    # print(board)
    output = []
    for row in board:
        for square in row:
            output += board_dict[square]

    # output = np.reshape(output, (8, 8, 6))  # 64x6 -> 8x8x6
    # if turn == 'b':                         # if black to move, flip board vertically
    #     output = np.flip(output, 0)

    # pprint.pprint(output)
    return output
