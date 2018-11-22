from functions import *
from matplotlib import pyplot as plt
from numpy import mean, corrcoef
from sklearn import tree
from joblib import dump, load

evaluations = []
results = []
wins = []
losses = []
draws = []
for game, move, x, y0, y1 in fen_generator('sources/ccrl.pgn'):
    evaluations.append([y0.item(), move])
    results.append(y1.item())

# plt.hist([wins, losses, draws], 50, label=['wins', 'losses', 'draws'], stacked=True)
# plt.legend()
# plt.title('all')
# plt.xlim([-3, 3])
# plt.show()

# print(corrcoef(evaluations, results)[0][1])
# print(mean(wins), mean(losses), mean(draws))

clf = tree.DecisionTreeClassifier()
clf.fit(evaluations, results)
dump(clf, 'clf_model.joblib')
