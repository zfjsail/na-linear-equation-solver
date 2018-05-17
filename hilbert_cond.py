import math
from scipy.linalg import hilbert
from numpy.linalg import cond
import matplotlib.pyplot as plt


def get_hilbert_cond2(n):
    x = []
    y = []
    for i in range(2, n):
        cur_mat = hilbert(i)
        cur_cond = cond(cur_mat, 2)
        x.append(i)
        y.append(math.log(cur_cond))
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    get_hilbert_cond2(10)  # exp
