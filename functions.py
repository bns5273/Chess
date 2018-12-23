import re
import torch
import chess.pgn as pgn
from joblib import load


def get_result(h):
    if h['Result'] == '1-0':
        return 1
    elif h['Result'] == '0-1':
        return -1
    else:
        return 0


# converts centipawns to [loss, draw, win] probabilities
def get_y(centipawns, turn):
    clf = load('sources/clf_model.joblib')
    y = clf.predict_proba([[centipawns, turn]])
    return torch.FloatTensor(y[0])


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


# game #, move #, x (node), y0 (eval), y1 (result)
# [loss, draw, win]
def fen_generator(filename, limit):
    with open(filename, 'r', encoding='Latin-1') as file:
        position = 0
        game = 0
        for i in range(limit):
            move = 1
            old_node = pgn.read_game(file)
            result = get_result(old_node.headers)
            while not old_node.is_end():
                node = old_node.variations[0]
                # only white to move positions. some engines (as black) evaluate from perspective!
                if move % 1 == 0:  # and -3 < eval < 3:
                    eval = re.findall('[-+]\d*\.\d*', node.comment)
                    if eval:
                        eval = get_y(float(eval[0]), move)
                        yield game, move, get_x(node), eval, result
                        position += 1
                    elif re.findall('[-]', node.comment):
                        eval = [1, 0, 0]
                        yield game, move, get_x(node), eval, result
                        position += 1
                    elif re.findall('[+]', node.comment):
                        eval = [0, 0, 1]
                        yield game, move, get_x(node), eval, result
                        position += 1
                old_node = node
                move += .5
            game += 1

        print(game, 'games', position, 'positions')
