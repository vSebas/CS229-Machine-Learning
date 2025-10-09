import numpy as np
from src.linearclass import util
import os

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path as a 1D numpy array

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)

    np.savetxt(save_path, y_pred)

    util.plot(x_valid, y_valid, clf.theta, save_path.replace('.txt', '.png'))
    
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        g = lambda z: 1 / (1 + np.exp(-z))

        n_examples, dim = x.shape
        
        if self.theta is None:
            self.theta = np.zeros(dim)  # dim is the number of features (x1, x2, x3, ... , xd)
            self.prev_theta = np.zeros(dim)

        for i in range(self.max_iter):
            self.prev_theta = self.theta.copy()

            eta = x @ self.theta
            gradient = x.T @ (g(eta) - y)       # dim x 1
            M = np.diag(g(eta) * (1 - g(eta)))  # diagonal matrix with g(eta) * (1 - g(eta)) on the diagonal, dim x dim
            H = x.T @ (M @ x)
            self.theta -= np.linalg.solve(H, gradient)
            loss = -(y * np.log(g(eta) + self.eps) + (1 - y) * np.log(1 - g(eta) + self.eps)).sum()/n_examples
            
            # if self.verbose and i % 1 == 0:
            #     print(f'Iteration {i}, loss: {loss}')

            if(np.linalg.norm(self.theta - self.prev_theta) < self.eps):
                print(f'Converged at iteration {i}, loss: {loss}')
                break

            if self.verbose and i == self.max_iter - 1:
                print(f'Max iterations reached, loss: {loss}')

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        g = lambda z: 1 / (1 + np.exp(-z))

        return g(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    main(train_path=os.path.join(script_dir, 'ds1_train.csv'),
         valid_path=os.path.join(script_dir, 'ds1_valid.csv'),
         save_path=os.path.join(script_dir, 'logreg_pred_1.txt'))

    main(train_path=os.path.join(script_dir, 'ds2_train.csv'),
         valid_path=os.path.join(script_dir, 'ds2_valid.csv'),
         save_path=os.path.join(script_dir, 'logreg_pred_2.txt'))