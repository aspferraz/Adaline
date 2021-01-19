
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
from adaline import Adaline


def run():
    # load training dataset
    df = pd.read_csv('res/data.csv', header=None)
    print(df.head())

    X = df.iloc[:35, [1, 2, 3, 4]].values
    y = df.iloc[:35, 5].values

    print(X)
    print("Y:", y)

    # plotting errors

    names = ['T1', 'T2', 'T3', 'T4', 'T5']

    step = 1
    plt.figure(figsize=(14, 7))
    for name in names:
        ax = plt.subplot(2, 3, step)
        clf = Adaline().fit(X, y)
        ax.plot(range(len(clf.error_)), clf.error_)
        ax.set_ylabel('Sum-squared-error')
        ax.set_xlabel('Epochs')
        ax.set_title(name)

        step += 1

    plt.show()


    # load test dataset
    df = pd.read_csv('res/test.csv',header=None)
    print(df.head())

    clf = Adaline().fit(X, y)

    X = df.iloc[:15, [1, 2, 3, 4]].values
    for i in range(X.shape[0]):
        pattern = X[i,:]
        print(pattern)
        print(str(clf.predict(pattern)))


if __name__ == '__main__':
    run()

