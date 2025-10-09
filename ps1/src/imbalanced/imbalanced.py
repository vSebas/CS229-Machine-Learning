import numpy as np
from src.imbalanced import util
from random import random
import os

### NOTE : You need to complete logreg implementation first!

from src.linearclass.logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt() as a 1D numpy array

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)
    y_pred_label = (y_pred >= 0.5).astype(int)

    # confusion matrix
    tp = np.sum((y_valid == 1) & (y_pred_label == 1))
    tn = np.sum((y_valid == 0) & (y_pred_label == 0))
    fp = np.sum((y_valid == 0) & (y_pred_label == 1))
    fn = np.sum((y_valid == 1) & (y_pred_label == 0))

    A = (tp + tn) / (tp + tn + fp + fn)
    A0 = tp / (tp + fn)     # minority class
    A1 = tn / (tn + fp)     # majority class
    BA = 0.5 * (A0 + A1)

    print(f'Accuracy: {A}, Balanced Accuracy: {BA}, A0: {A0}, A1: {A1}')    

    np.savetxt(output_path_naive, y_pred)
    util.plot(x_valid, y_valid, clf.theta, output_path_naive.replace('.txt', '.png'))

    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt() as a 1D numpy array
    # Repeat minority examples 1 / kappa times

    pos_samples = (y_train == 1)
    neg_samples = (y_train == 0)
    x_train_upsampled = np.vstack([x_train[neg_samples], np.repeat(x_train[pos_samples], int(1 / kappa), axis=0)])
    y_train_upsampled = np.concatenate([y_train[neg_samples], np.repeat(y_train[pos_samples], int(1 / kappa))])

    clf = LogisticRegression()
    clf.fit(x_train_upsampled, y_train_upsampled)
    y_pred = clf.predict(x_valid)
    y_pred_label = (y_pred >= 0.5).astype(int)

    # confusion matrix
    tp = np.sum((y_valid == 1) & (y_pred_label == 1))
    tn = np.sum((y_valid == 0) & (y_pred_label == 0))
    fp = np.sum((y_valid == 0) & (y_pred_label == 1))
    fn = np.sum((y_valid == 1) & (y_pred_label == 0))

    A = (tp + tn) / (tp + tn + fp + fn)
    A0 = tp / (tp + fn)    # minority class
    A1 = tn / (tn + fp)    # majority class
    BA = 0.5 * (A0 + A1)

    print(f'Accuracy: {A}, Balanced Accuracy: {BA}, A0: {A0}, A1: {A1}')    

    np.savetxt(output_path_upsampling, y_pred)
    util.plot(x_valid, y_valid, clf.theta, output_path_upsampling.replace('.txt', '.png'))

    # *** END CODE HERE

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(train_path=os.path.join(script_dir, 'train.csv'),
         validation_path=os.path.join(script_dir, 'validation.csv'),
         save_path=os.path.join(script_dir, 'imbalanced_X_pred.txt'))
