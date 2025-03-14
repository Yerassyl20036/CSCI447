import pylab as pl
import scipy as sp
import numpy as np
from scipy.linalg import eig
from scipy.io import loadmat
import pdb

def load_data(fname):
    # load the data
    data = loadmat(fname)
    X, Y = data['X'], data['Y']
    # Collapse the time-electrode dimensions
    X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    # transform
    Y = np.sign((Y[0, :] > 0) - 0.5)
    return X, Y

def train_ncc(X, Y):
    """
    Train a nearest centroid classifier
    """
    target_idx = np.where(Y == 1)[0]
    non_target_idx = np.where(Y == -1)[0]

    mu_target = np.mean(X[:, target_idx], axis=1)
    mu_non_target = np.mean(X[:, non_target_idx], axis=1)

    w = mu_target - mu_non_target
    b = -0.5 * (np.dot(w, mu_target) + np.dot(w, mu_non_target))
    return w, b

def train_lda(X, Y):
    """
    Train a linear discriminant analysis classifier
    """
    target_idx = np.where(Y == 1)[0]
    non_target_idx = np.where(Y == -1)[0]
    
    mu_target = np.mean(X[:, target_idx], axis=1, keepdims=True)
    mu_non_target = np.mean(X[:, non_target_idx], axis=1, keepdims=True)

    # Between-class scatter
    SB = (mu_target - mu_non_target) @ (mu_target - mu_non_target).T

    # Within-class scatter
    SW = np.zeros((X.shape[0], X.shape[0]))
    for i in target_idx:
        SW += np.outer(X[:, i] - mu_target[:, 0], X[:, i] - mu_target[:, 0])
    for j in non_target_idx:
        SW += np.outer(X[:, j] - mu_non_target[:, 0], X[:, j] - mu_non_target[:, 0])

    # Regularization to stabilize the inverse
    regularization_term = 0.01 * np.eye(SW.shape[0])
    SW += regularization_term

    eigvals, eigvecs = eig(SB, SW)
    w = eigvecs[:, np.argmax(eigvals.real)].real

    b = -0.5 * (np.dot(w, mu_target[:, 0]) + np.dot(w, mu_non_target[:, 0]))

    return w, b

def compare_classifiers():
    """
    compares nearest centroid classifier and linear discriminant analysis
    """
    fname = 'bcidata.mat'
    X, Y = load_data(fname)

    # Shuffle and split
    permidx = np.random.permutation(np.arange(X.shape[-1]))
    trainpercent = 70.
    stopat = int(np.floor(Y.shape[-1] * trainpercent / 100.))

    X_train = X[:, permidx[:stopat]]
    Y_train = Y[permidx[:stopat]]
    X_test = X[:, permidx[stopat:]]
    Y_test = Y[permidx[stopat:]]

    # Train both classifiers
    w_ncc, b_ncc = train_ncc(X_train, Y_train)
    w_lda, b_lda = train_lda(X_train, Y_train)

    # Evaluate accuracy: sign(w^T x + b)
    preds_ncc = np.sign(w_ncc.dot(X_test) + b_ncc)
    preds_lda = np.sign(w_lda.dot(X_test) + b_lda)
    acc_ncc = 100.0 * np.mean(preds_ncc == Y_test)
    acc_lda = 100.0 * np.mean(preds_lda == Y_test)

    # Plot histograms for raw projections w^T X_test
    fig = pl.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(w_ncc.dot(X_test[:, Y_test < 0]))
    ax1.hist(w_ncc.dot(X_test[:, Y_test > 0]))
    ax1.set_xlabel('$w_{NCC}^T X$')
    ax1.legend(('non-target', 'target'))
    ax1.set_title(f"NCC Acc: {acc_ncc:.2f}%")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(w_lda.dot(X_test[:, Y_test < 0]))
    ax2.hist(w_lda.dot(X_test[:, Y_test > 0]))
    ax2.set_xlabel('$w_{LDA}^T X$')
    ax2.legend(('non-target', 'target'))
    ax2.set_title(f"LDA Acc: {acc_lda:.2f}%")

    pl.savefig('ncc-lda-comparison.pdf')
    pl.show()

def crossvalidate(X, Y, f=10, trainfunction=train_lda):
    """
    Test generalization performance of a linear classifier
    Input:  X   data (dims-by-samples)
            Y   labels (1-by-samples) assuming labels are {1, -1}
            f   number of cross-validation folds
            trainfunction  trains linear classifier
    """
    num_samples = X.shape[1]
    fold_size = num_samples // f
    indices = np.random.permutation(num_samples)

    acc_train = np.zeros(f)
    acc_test = np.zeros(f)

    for ifold in range(f):
        test_indices = indices[ifold * fold_size : (ifold + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, Y_train = X[:, train_indices], Y[train_indices]
        X_test, Y_test = X[:, test_indices], Y[test_indices]

        w, b = trainfunction(X_train, Y_train)

        preds_train = np.sign(w.dot(X_train) + b)
        preds_test = np.sign(w.dot(X_test) + b)

        acc_train[ifold] = np.mean(preds_train == Y_train)
        acc_test[ifold] = np.mean(preds_test == Y_test)

    return acc_train, acc_test

if __name__ == "__main__":
    # Load the data
    X, Y = load_data('bcidata.mat')

    # Compare NCC vs. LDA with a 70/30 split
    compare_classifiers()

    # Cross-validation for LDA
    acc_train, acc_test = crossvalidate(X, Y, f=10, trainfunction=train_lda)
    print('Accuracy on training data (per fold):', acc_train)
    print('Accuracy on test data (per fold):   ', acc_test)

    # plot of cross-validation results
    fig = pl.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.boxplot([acc_train, acc_test], labels=['train block', 'test block'])
    ax.set_title('Cross-Validation Accuracy for LDA')
    ax.set_ylabel('Accuracy')

    # set y range
    ax.set_ylim([0.945, 0.985])

    pl.savefig('cross_validation_accuracy.pdf')
    pl.show()
