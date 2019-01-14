from functions import *
import numpy as np
import torch
from sklearn import tree
from joblib import dump, load


# converts centipawns to [loss, draw, win] probabilities
class EvalNorm:
	def __init__(self):
		self.clf = load('sources/eval_norm_model.joblib')

	def get_y(self, centipawns):
		y = self.clf.predict_proba([[centipawns]])
		if torch.cuda.is_available():
			return torch.cuda.FloatTensor(y[0])
		else:
			return torch.FloatTensor(y[0])


# no longer will work as the fen_generator is now built on this trained model!
if __name__ == '__main__':
	x = []
	y = []
	for game, x0, x1, y0, y1 in fen_generator('sources/ccrl.pgn', 10000):
		temp = list(y1.data).index(1) - 1
		for i in range(len(x1)):
			x.append(x1[i])
			y.append(temp)
		# print(x)
		# print(y)
	x = np.reshape(x, (-1, 1))
	y = np.reshape(y, (-1, 1))

	clf = tree.DecisionTreeClassifier()
	clf.fit(x, y)

	print(clf.predict_proba([[-.5], [0], [.5], [.611]]))

	dump(clf, 'sources/eval_norm_model.joblib')
