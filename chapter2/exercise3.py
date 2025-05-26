"""
Verify if the discriminative and generative linear regression models make the same predictions for the same data
Fit a line to three data points using both methods and see if
the result is the same.

Solution:
    Fix a dataset.
    Calculate the closed form versions to highlight any differences
"""

import numpy
import scipy.linalg

def calculate_loss_least_squares(y, y_hat):
    return numpy.sum( numpy.square(numpy.subtract(y, y_hat)), axis=0)

dataset1 = {
    "x": numpy.array([0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90]),
    "y": numpy.array([0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54, 1.55, 1.68, 1.73, 1.6 ]),
}

data = dataset1

x = data['x']
y = data['y']

X_augmented = numpy.column_stack((numpy.ones_like(x), x))
Y_augmented = numpy.column_stack((numpy.ones_like(x), y))

phi = numpy.polyfit(x,y,deg=1)[::-1] # returns parameters in reverse order
y_hat = X_augmented @ phi # [1 x]*phi
print("Loss Inference:", calculate_loss_least_squares(y, y_hat))

theta = numpy.polyfit(y,x,deg=1)[::-1]
x_hat = Y_augmented @ theta # [1 y]*phi
X_hat_augmented = numpy.column_stack((numpy.ones_like(x), x_hat))
print("Loss Generative:", calculate_loss_least_squares(x, x_hat))
y_hat_g = (x - theta[0]) / theta[1]

def calculate_generative_inference_parameters(phi):
    """
    Given x = g[phi, y] for linear regression model,
    y = x*phi^T(phi*phi^T)^-1

    Doesn't compute the inverse directly, instead use psudoinverse
    Z = phi^T(phi*phi^T)^-1
    NOTE: linalg.pinv works for both under- and over-determined cases
        in which dim(phi) != dim(x)
    """
    out = numpy.linalg.pinv(phi)
    return out

# Recover z via pseudoinverse
theta_inv = calculate_generative_inference_parameters(theta.reshape(-1,1))
print("Estimated generative inverse parameters:", theta_inv)
y_hat_g_inverse = theta_inv @ X_hat_augmented.T

# These are supposed to be really close if the data is linear
print("Won't all match unless linear ------ ")
print("Disciminative y_hat outputs:", y_hat)
print("Inverse generative (algebraic):", y_hat_g)
print("Inverse generative (pseodoinv):", y_hat_g_inverse)