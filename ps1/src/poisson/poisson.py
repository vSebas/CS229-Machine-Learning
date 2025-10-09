import numpy as np
from src.poisson import util
import matplotlib.pyplot as plt
import os


def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path as a 1D numpy array

    # Plot the training data and the fitted Poisson regression curve
    clf = PoissonRegression(step_size=lr, verbose=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)

    # plt.scatter([:, 1], y_train, label='Training data')
    plt.scatter(y_eval, y_pred)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Poisson Regression: True vs Predicted')
    plt.legend()
    plt.show()
    
    np.savetxt(save_path, y_pred, fmt="%.6f")

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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

        self.prev_theta = None

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Initialize self.theta to zero vector if it is None
        # Implement gradient ascent loop with convergence check

        n_examples, dim = x.shape
        
        if self.theta is None:
            self.theta = np.zeros(dim)  # dim is the number of features
            self.prev_theta = np.zeros(dim)

        # Compute the gradient of the log-likelihood for Poisson regression 
        # with batch gradient ascent
        theta = 0

        for i in range(self.max_iter):
            gradient = np.zeros(dim) # same dimension as theta
            e = np.exp(x @ self.theta) # n_examples x 1
            error = y - e
            gradient = x.T @ error # dim x n_examples

            self.prev_theta = self.theta.copy() # copy because self.theta will be updated
            self.theta += self.step_size * gradient

            if self.verbose and i % 1000 == 0:
                print(f'Iteration {i}')

            if(np.linalg.norm(self.theta - self.prev_theta) < self.eps):
                print(f'Converged at iteration {i}')
                break

            if self.verbose and i == self.max_iter - 1:
                print(f'Max iterations reached')

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Return the prediction according to the Poisson regression model
        return np.exp(x @ self.theta)

        # *** END CODE HERE ***

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(lr=1e-5,
        train_path=os.path.join(script_dir, 'train.csv'),
        eval_path=os.path.join(script_dir, 'valid.csv'),
        save_path=os.path.join(script_dir, 'poisson_pred.txt'))
