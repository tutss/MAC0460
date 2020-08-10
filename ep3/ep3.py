#########################################################
# Nome: Artur Magalhaes Rodrigues dos Santos            #
# NUSP: 10297734                                        #
#                                                       #
# Na versão do código com todas as funções,             #
# logistic_fit e logistic_predict tinham sido           #
# modularizadas. Uma função retornava o gradiente,      #
# outra o vetor de pesos (para logistic_fit). Em        #
# logistic_predict, a sigmóide era uma função a parte.  #
# Condensei ambas para adequar o envio (apenas          #
# logistic fit e prediction).                           #
#                                                       #
# Requisitos: Numpy                                     #
#########################################################
import numpy as np


def logistic_fit(X, y, w=None, batch_size=None, learning_rate=1e-2,
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
    assert X.shape[0] == y.shape[0], 'X and y with different sizes'
    size, dim = X.shape
    X = np.c_[np.ones(size), X]
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

        error_sum = sum(y[batch*i:batch*(i+1)] * X[batch*i:batch*(i+1)] /
                        (1 + np.exp(y[batch*i:batch*(i+1)] *
                                    np.dot(X[batch*i:batch*(i+1)], weights.T))))

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
