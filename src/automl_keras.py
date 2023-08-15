import autokeras as ak
import numpy as np
from sklearn.model_selection import train_test_split

# npz = np.load('data/data_matrix.npz')
npz = np.load('data/data_text.npz', allow_pickle=True)
x_train, x_valid, y_train, y_valid = train_test_split(npz['x'], npz['y'])

# clf = ak.StructuredDataClassifier()  # --for data_matrix, acc: 0.685
clf = ak.TextClassifier()  # for data_text, acc: 0.7309
clf.fit(x_train, y_train, verbose=2)
print(clf.evaluate(x_valid, y_valid, verbose=2))
# y_test = clf.predict(x_test, verbose=2)
