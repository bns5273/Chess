from functions import *
from sklearn import tree
from joblib import dump

evaluations = []
results = []
wins = []
losses = []
draws = []
for game, move, x, y0, y1 in fen_generator('sources/ccrl.pgn', 100):
    evaluations.append([y0.item(), move])
    results.append(y1.item())

clf = tree.DecisionTreeClassifier()
clf.fit(evaluations, results)
dump(clf, 'clf_model.joblib')
