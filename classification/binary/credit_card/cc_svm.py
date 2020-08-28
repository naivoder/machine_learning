import numpy as np
from sklearn import svm
from sklearn.utils import shuffle

# adapted from scikit-learn.org
# included for reference
def test_svm(debug=False):
    x = [[0, 0], [1, 1]]
    y = [0, 1]
    model = svm.SVC()
    model.fit(x, y)
    if debug:
        print('should predict "[1]"...')
    print(model.predict([[2., 2.]])) if debug else model.predict([[2., 2.]])
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

def cc_svm(recs, targs, kern=None):
    if kern is not None:
        model = svm.SVC(kernel=kern)
    else:
        model = svm.SVC()
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
    test_model = test_svm(debug=False)
    features, train_records, train_targets, test_records, test_targets = load_data('credit_card_data-headers.txt', debug=False)
    svm = cc_svm(train_records, train_targets)
    predictions = make_predictions(svm, test_records)
    accuracy = calculate_accuracy(predictions, test_targets, debug=False)