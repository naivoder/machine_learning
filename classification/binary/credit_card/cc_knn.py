import numpy as np
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier as knn

# adapted from scikit-learn.org
# included for reference
def test_knn(debug=False):
    x = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    model = knn(n_neighbors=3)
    model.fit(x, y)
    if debug:
        print('should predict "[0]"...')
    print(model.predict([[1.1]])) if debug else model.predict([[1.1]])
    return model

def load_data(data, header=True, debug=False):
    if header:
        with open(data, 'r') as file:
            feats = [x for x in file.readline().rstrip().split('\t')]
    recs  = np.loadtxt(data, skiprows=1)
    targs = recs[:, -1]
    recs  = recs[:,:-1]
    recs, targs = shuffle(recs, targs)
    train_recs  = recs[:554, :]
    test_recs   = recs[554:, :]
    train_targs = targs[:554]
    test_targs  = targs[554:]
    if debug:
        print('Sanity Samples...')
        print('Features:', feats[:3])
        print(len(feats))
        print('Records:', recs[0, :3])
        print(len(recs), len(recs[0]))
        print('Targets:', targs[:3, ])
        print(len(targs), 1)
        print('Test/Train Split:', len(train_recs), len(test_recs))
    return feats, train_recs, train_targs, test_recs, test_targs

def cc_knn(recs, targs, k, alg='auto'):
    model = knn(n_neighbors=k, algorithm=alg)
    model.fit(recs, targs)
    return model

def make_predictions(model, recs):
    preds = model.predict(recs)
    return preds

def calculate_accuracy(preds, targs, debug=False):
    if debug:
        print(preds)
        print(targs)
    correct = sum([1 for i in range(len(preds)) if preds[i] == targs[i]])
    total = len(preds)
    acc = float(correct / total)
    print('Model Accuracy:', acc)
    return acc

if __name__=='__main__':
    test_model = test_knn(debug=False)
    features, train_records, train_targets, test_records, test_targets = load_data('credit_card_data-headers.txt', debug=False)
    knn = cc_knn(train_records, train_targets, k=11)
    predictions = make_predictions(knn, test_records)
    accuracy = calculate_accuracy(predictions, test_targets, debug=False)