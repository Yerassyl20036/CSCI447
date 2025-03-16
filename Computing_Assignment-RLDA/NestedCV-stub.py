import pylab as pl
import numpy as np
import scipy as sp
from numpy.linalg import inv
from scipy.io import loadmat
import pdb

def load_data(fname):
    # load the data
    data = loadmat(fname)
    # extract images and labels
    X = data['X']
    Y = data['Y']
    # collapse the time-electrode dimensions
    X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    # transform the labels to (-1, 1)
    Y = np.where(Y[0, :] > 0, 1, -1)

    # Randomly pick 500 trials
    n_samples = X.shape[1]
    perm = np.random.permutation(n_samples)
    idx_500 = perm[:500]
    X = X[:, idx_500]
    Y = Y[idx_500]

    # Standardize each feature (zero mean, unit variance)
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    X_std[X_std == 0] = 1.0  # avoid division by zero
    X = (X - X_mean) / X_std

    print(X.shape)
    return X, Y

def crossvalidate_nested(X, Y, f, gammas):
    """
    Optimize shrinkage parameter for generalization performance via nested CV.
    Input:
      X: data (dims x samples)
      Y: labels (samples,)
      f: number of folds
      gammas: array of possible shrinkage parameters
    """
    # the next two lines reshape vector of indices in to a matrix:
    # number of rows = # of folds
    # number of columns = # of total data-points / # folds
    N = f * int(np.floor(X.shape[-1] / f))
    idx = np.reshape(np.arange(N), (f, int(np.floor(N / f))))
    acc_test = np.zeros(f)
    testgamma = np.zeros((len(gammas), f))

    # loop over folds:
    # select one row of 'idx' for testing, all other rows for training
    # call variables (indices) for training and testing 'train' and 'test'
    for ifold in range(f):
        test_index = idx[ifold, :]
        train_index = np.hstack(idx[np.arange(f) != ifold, :])
        
        X_train, X_test = X[:, train_index], X[:, test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # loop over gammas
        inner_accuracies = np.zeros(len(gammas))
        for igamma, gamma_val in enumerate(gammas):
            # each gamma is fed into the inner CV via the function 'crossvalidate_lda'
            # the resulting variable is called 'testgamma'
            inner_accuracies[igamma] = crossvalidate_lda(X_train, Y_train, f - 1, gamma_val)
            # find the the highest accuracy of gammas for a given fold and use it to train an LDA on the training data
        
        best_gamma_idx = np.argmax(inner_accuracies)
        best_gamma = gammas[best_gamma_idx]
        testgamma[:, ifold] = inner_accuracies

        w, b = train_lda(X_train, Y_train, best_gamma)

        # calculate the accuracy for this LDA classifier on the test data
        predictions = np.sign(X_test.T @ w - b)
        acc_test[ifold] = np.mean(predictions == Y_test)

    # do some plotting
    pl.figure()
    pl.boxplot(testgamma.T)
    pl.xticks(np.arange(len(gammas)) + 1, gammas)
    pl.xlabel('$\\gamma$')
    pl.ylabel('Accuracy')
    pl.savefig('cv_nested-boxplot.pdf')

    return acc_test, testgamma

def crossvalidate_lda(X, Y, f, gamma):
    ''' 
    Test generalization performance of shrinkage lda
    Input:	X	data (dims-by-samples)
                Y	labels (1-by-samples)
                f	number of cross-validation folds
                trainfunction 	trains linear classifier, returns weight vector and bias term
    '''
    N = f * int(np.floor(X.shape[-1] / f))
    idx = np.reshape(np.arange(N), (f, int(np.floor(N / f))))
    acc_test = np.zeros(f)

    # loop over folds
    # select one row of idx for testing, all others for training
    # call variables (indices) for training and testing 'train' and 'test'
    for ifold in range(f):
        test_index = idx[ifold, :]
        train_index = np.hstack(idx[np.arange(f) != ifold, :])
        
        X_train, X_test = X[:, train_index], X[:, test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # train LDA classifier with training data and given gamma:
        w, b = train_lda(X_train, Y_train, gamma)
        # test classifier on test data:
        predictions = np.sign(X_test.T @ w - b)
        acc_test[ifold] = np.mean(predictions == Y_test)

    return acc_test.mean()

def train_lda(X, Y, gamma):
    """
    Train a regularized LDA classifier.
    We compute the weight vector using the standard LDA solution:
       w = inv(Shrink) * (mu+ - mu-)
    and set the offset as b = (w^T mu+ + w^T mu-)/2.
    """
    # class means
    mupos = np.mean(X[:, Y > 0], axis=1)
    muneg = np.mean(X[:, Y < 0], axis=1)

    # inter and intra class covariance matrices
    Sintra = np.cov(X[:, Y > 0]) + np.cov(X[:, Y < 0])

    # shrink covariance matrix estimate
    nu = np.mean(np.diag(Sintra))
    Shrink = (1 - gamma) * Sintra + gamma * nu * np.eye(Sintra.shape[0])

    # weight vector
    w = inv(Shrink) @ (mupos - muneg)

    # offset
    b = (np.dot(w, mupos) + np.dot(w, muneg)) / 2.
    return w, b

def main():
    X, Y = load_data('bcidata.mat')
    gammas = np.array([0, 0.005, 0.05, 0.5, 1])
    a, b = crossvalidate_nested(X, Y, 10, gammas)
    print(a)
    print(b)

if __name__ == "__main__":
    main()
