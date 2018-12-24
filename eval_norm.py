from functions import *
import torch
from sklearn import tree
from joblib import dump, load


# converts centipawns to [loss, draw, win] probabilities
class EvalNorm:
	def __init__(self):
		self.clf = load('sources/clf_model.joblib')

	def get_y(self, centipawns, turn):
		y = self.clf.predict_proba([[centipawns, turn]])
		return torch.FloatTensor(y[0])


if __name__ == '__main__':
	evaluations = []
	results = []
	for game, x, y0, y1 in fen_generator('sources/ccrl.pgn', 1):
		print(y0.data)
		print(y1.data)

	# clf = tree.DecisionTreeClassifier()
	# clf.fit(evaluations, results)
	# dump(clf, 'clf_model.joblib')
