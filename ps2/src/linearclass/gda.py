import numpy as np
import util



def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***

    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    clf = GDA()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)

    np.savetxt(save_path, y_pred)

    util.plot(x_valid, y_valid, clf.theta, save_path.replace('.txt', '.png'))

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = x[:, 1:]
        n_examples, dim = x.shape
        # print(x, y)
        # print(x.shape, y.shape)
        # print(y == 0)
        # print(x[y == 0]) # select rows where y == 0 by boolean indexing

        if self.theta is None:
            self.theta = np.zeros(dim)  # dim is the number of features (x1, x2, x3, ... , xd)
        
        sigma = np.zeros((dim, dim))

        # Find phi, mu_0, mu_1, and sigma
        phi = np.mean(y)   # proportion of positive examples sum(y) / n_examples
        mu_0 = np.mean(x[y == 0], axis=0)  # mean of negative examples, axis=/rows means compute the mean for each column
        mu_1 = np.mean(x[y == 1], axis=0)  # mean of positive examples

        for i in range(n_examples):
            if y[i] == 0:
                diff = (x[i] - mu_0)
            else:
                diff = (x[i] - mu_1)
            sigma += np.outer(diff, diff)
        sigma /= n_examples

        # Write theta in terms of the parameters
        self.theta =  np.linalg.inv(sigma) @ (mu_1 - mu_0) # theta is a vector of shape (dim,)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        g = lambda z: 1 / (1 + np.exp(-z))

        return g(x @ self.theta)
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
