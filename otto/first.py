import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


def create_X_y_files(input_filename):
    with open(input_filename) as the_file:
        df = pd.read_csv(the_file)

        #print df.head(3)

        df['category'] = 0

        for i in range(1,10):
            df.loc[df['target'] == 'Class_{}'.format(i), 'category'] = i

        #print df['category']

        X = df.ix[:,1:94] # zero indexed
        y = df['category']

        #write to CSV with headers
        X.to_csv('X_with_headers.csv', sep=',', header=True, index=False)
        y.to_csv('y_with_headers.csv', sep=',', header=True, index=False)
        #write to CSV without headers
        X.to_csv('X.csv', sep=',', header=False, index=False)
        y.to_csv('y.csv', sep=',', header=False, index=False)

        #print X.head()

create_X_y_files('sample.csv')

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_function(theta, X, y, n, m, lamb=1.0):

    ones_column = np.ones((m,1))

    #append a column of ones
    X = np.append(ones_column, X, axis=1)

    J=0
    grad = np.zeros(theta.shape) # (n+1, 1)

    sigmoid_h = sigmoid(X.dot(theta))

    J = np.multiply((1.0/m), np.sum(np.multiply(-y, np.log(sigmoid_h)) - np.multiply((1-y), np.log(1 - sigmoid_h)))) \
        + (lamb/(2.0*m)) * np.sum(np.square(theta[1:,0]))

    grad_non_reg = (1.0/m) * (X.transpose().dot(sigmoid_h - y))

    grad = grad_non_reg + (lamb/(m)) * theta

    grad[0,:] = grad_non_reg[0,:]

    return (J, grad)


def learn_training_set(X_filename, y_filename):
    X = np.loadtxt(X_filename, dtype=float, delimiter=',')
    y = np.loadtxt(y_filename, dtype=float, delimiter=',')
    m = y.shape[0]
    n = X.shape[1]

    init_theta = np.zeros((n+1,1))

    J, grad = cost_function(theta=init_theta, X=X, y=y, n=n, m=m, lamb=1.0)

    print J
    print grad

learn_training_set('X.csv', 'y.csv')





