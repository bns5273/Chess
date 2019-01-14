import chess
from functions import *
import chess
from chess import uci
import numpy as np


class MiniMax:
	# print utility value of root node (assuming it is max)
	# print names of all nodes visited during search
	def __init__(self, game_tree):
		self.game_tree = game_tree  # GameTree
		self.root = game_tree.root  # GameNode
		self.currentNode = None     # GameNode
		self.successors = []        # List of GameNodes
		return

	def minimax(self, node):
		# first, find the max value
		best_val = self.max_value(node)  # should be root node of tree

		# second, find the node which HAS that max value
		#  --> means we need to propagate the values back up the
		#      tree as part of our minimax algorithm
		successors = self.getSuccessors(node)
		print("MiniMax:  Utility Value of Root Node: = " + str(best_val))
		print(node)
		# find the node with our best move
		best_move = None
		for elem in successors:   # ---> Need to propagate values up tree for this to work
			if self.getUtility(elem) == best_val:
				best_move = elem
				break

		# return that best value that we've found
		return best_move

	def max_value(self, node):
		# print(node, '\n')
		if self.isTerminal(node):
			return self.getUtility(node)

		infinity = float('inf')
		max_value = -infinity

		successors_states = self.getSuccessors(node)
		for state in successors_states:
			max_value = max(max_value, self.min_value(state))
		return max_value

	def min_value(self, node):
		# print(node, '\n')
		if self.isTerminal(node):
			return self.getUtility(node)

		infinity = float('inf')
		min_value = infinity

		successor_states = self.getSuccessors(node)
		for state in successor_states:
			min_value = min(min_value, self.max_value(state))
		return min_value

	#                     #
	#   UTILITY METHODS   #
	#                     #

	# successor states in a game tree are the child nodes...
	def getSuccessors(self, node):
		assert node is not None

		successors = []
		mvs = node.generate_legal_moves()
		for mv in mvs:
			c = chess.Board(node.fen())
			c.push(mv)
			successors.append(c)
		print(len(successors))

		return successors

	# return true if the node has NO children (successor states)
	# return false if the node has children (successor states)
	def isTerminal(self, node):
		assert node is not None
		return node.is_game_over() or node.fullmove_number > 3

	def getUtility(self, node):
		assert node is not None
		return node.occupied_co[0]      # white pawn integer


if __name__ == '__main__':
	board = chess.Board()
	board.push_san('f4')
	board.push_san('e5')
	board.push_san('g4')

	mm = MiniMax(board)
	mm.minimax(board)

	# batch = get_x(board)
	# batch.unsqueeze(0)
	# output = net(batch)

