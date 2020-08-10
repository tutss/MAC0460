import numpy as np
import matplotlib.pyplot as plt


def logistic_fit(input_data, output_data, w=None, batch_size=None, learning_rate=1e-2,
                 num_iterations=1000, return_history=False):
    """
    Logistic regression implementation.
    **Parameters:
    X input      - 2D array, size N x d
    y labels     - 1D array, size N (values +1 and -1)
    w weights    - 1D array, size d + 1
    batch_size   - integer specifying batch size for training (batch_size < N)
    learning_rate - real value, gradient descent parameter
    num_iterations - integer specifying the number of iterations in X
    return_history - boolean
    **Outputs:
    weights - weight vector
    if return_history is True, return the values of the function at each update
    """
    assert input_data.shape[0] == output_data.shape[0], 'X and y with different sizes'
    size, dim = input_data.shape
    input_data = np.c_[np.ones(size), input_data]
    weights = np.random.rand(1, dim+1) if w is None else w

    # Make format 1 x (d+1)
    if len(weights.shape) == 1:
        weights = weights[np.newaxis]
    batch = size if batch_size is None or batch_size > size else batch_size
    i = 0
    history = np.zeros(dim+1)
    while num_iterations:
        if batch*(i+1) >= size:
            i = 0

        error_sum = sum(output_data[batch*i:batch*(i+1)] * input_data[batch*i:batch*(i+1)] /
                        (1 + np.exp(output_data[batch*i:batch*(i+1)] *
                                    np.dot(input_data[batch*i:batch*(i+1)], weights.T))))

        gradient = -error_sum/batch
        vt_direction = -gradient
        weights = weights + learning_rate * vt_direction

        if return_history:
            history = np.vstack((history, weights))

        num_iterations -= 1
        i += 1

    if return_history:
        return weights, history
    return weights


def gradient_descent(input_data, output, weight_vector, batch_size):
    """
    Gradient descent method for logistic regression (
    used Learning from Data Mostafa version)

    :param input_data: input values
    :param output: output values
    :param weight_vector: transposed weights
    :param batch_size: integer batch size
    :return: correspondent gradient
    """
    error_sum = sum(output * input_data /
                    (1 + np.exp(output * np.dot(input_data, weight_vector.T))))
    return -error_sum/batch_size


def update_weights(weights, learn_rate, gradient_direction):
    """
    Update rule for the weights

    :param weights: weights vector
    :param learn_rate: learning rate
    :param gradient_direction: gradient direction
    :return: updated weights
    """
    return weights + learn_rate * gradient_direction


def shapes(*args):
    """
    Auxliar function for checking shapes
    :param args: matrices
    """
    for m in args:
        print("\t\tMatrix shape: ", m.shape)


def logistic_predict(data, weights):
    """
    Predict y given input data X and weights w.
    **Parameters:
    X input      - 2D array, size N x d
    w weights    - 1D array, size d + 1
    **Output:
    prediction - predicted value given the weight and the input
    """
    data = np.c_[np.ones(len(data)), data]
    pred = 1 / (1 + np.exp(-np.dot(data, weights.T)))
    return pred


def sigmoid(z_parameter):
    """
    Sigmoid function

    :param z_parameter: array of values
    :return: sigmoid of z
    """
    return 1 / (1 + np.exp(-z_parameter))


def sigmoid_derivative(sig):
    """
    Derivative of the sigmoid function
    :param sig: real value output of the sigmoid function
    :return: value of derivative of the sigmoid function
    """
    return sig * (1 - sig)


def aux_print(arr_x, arr_y):
    """
    Print function for debugging
    """
    print(arr_x)
    print('\n\n\n\n############################################\n\n\n\n')
    print(arr_y)
    print('\n\n\n\n############################################\n\n\n\n')


def plot(X, y, y_hat):
    """
    Auxiliar function for plotting data
    :param X: 2D array, N x 2
    :param y: 1D array, N x 1
    """
    plt.title("Logistic Regression (yellow and purple x's are predictions)")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.scatter(X[:, 0], X[:, 1], c=['blue' if i > 0 else 'red' for i in y])
    plt.scatter(X[:, 0], X[:, 1], c=['yellow' if i > 0.5 else 'purple' for i in y_hat], marker='x')
    plt.show()


def generate_dataset(mean_pos, mean_neg, cov, N):
    """
    Generate dataset given means and covariance. Seen on PACA Moodle.

    :param mean_pos: mean vector for positive Normal
    :param mean_neg: mean vector for negative Normal
    :param cov: covariance matrix
    :param N: size
    :return: data, output - X and y
    """
    x_position = np.random.multivariate_normal(mean_pos, cov, size=N // 2)
    y_position = np.zeros(N // 2) + 1

    x_negative = np.random.multivariate_normal(mean_neg, cov, size=N - N // 2)
    y_neg = np.zeros(N - N // 2) - 1

    data = np.concatenate([x_position, x_negative], axis=0)
    output = np.concatenate([y_position, y_neg], axis=0)

    perm = np.random.permutation(N)
    data = data[perm]
    output = output[perm]

    return data, output


def main():
    """
    Test function.
    """
    N = 1000

    mean_pos = [0, 0]
    mean_neg = [3, 0]
    cov = [
        [1, 0],
        [0, 1]
    ]

    X, y = generate_dataset(mean_pos, mean_neg, cov, N)
    y = y[np.newaxis].T
    # shapes(X, y)
    w = logistic_fit(X, y, num_iterations=10000)
    prediction = logistic_predict(X, w)
    # print(prediction)
    plot(X, y, prediction)


if __name__ == "__main__":
    main()

# def format_input(x, y, n_elem_1, n_elem_2):
#     """
#     Given x and y points, joins them in a single matrix, for further usage in
#     predicting.
#
#     Parameters:
#     x - 1D array, x points
#     y - 1D array, y points
#     n_elem_1 - number of elements of the first set
#     n_elem_2 - number of elements of the second set
#
#     Output:
#     X - 2D array, dataset
#     output - 1D array, labels for each tuple in the dataset
#     """
#     assert len(x) == len(y), 'Missing values'
#     size = len(x)
#     X = np.zeros((size, 2))
#     for i in range(size):
#         X[i][0] = x[i]
#     for i in range(size):
#         X[i][1] = y[i]
#     output = np.ones((n_elem_1 + n_elem_2, 1))
#     output[n_elem_1:n_elem_1 + n_elem_2] *= -1
#     return X, output


# def join_regions(mean_1, cov_1, n1, mean_2, cov_2, n2):
#     """
#     Auxiliar function to create a dataset defined by to regions. These
#     regions follow a Multivariate Normal distribution.
#
#     Parameters:
#     mean_1 - array that defines the mean of the first set of points
#     mean_2 - array that defines the mean of the second set of points
#     cov_1 - covariance of the first set of points
#     cov_2 - covariance of the second set of points
#     n1 - number of points of the first set
#     n2 - number of points of the second set
#
#     Output:
#     x - 1D array, x axis values
#     y - 1D array, y axis values
#     """
#     x1, y1 = np.random.multivariate_normal(mean_1, cov_1, n1).T
#     x2, y2 = np.random.multivariate_normal(mean_2, cov_2, n2).T
#     x = np.append(x1, x2)
#     y = np.append(y1, y2)
#     return x, y


# def generate_points(w, dx):
#     x = []
#     y = []
#     i = 5
#     x1 = -5
#     while i >= -5:
#         x2 = w[0][0] + w[0][1] * x1 + w[0][2] * np.power(x1, 2)
#         x.append(x1)
#         y.append(x2)
#         x1 += dx
#         i -= dx
#     return x, y

