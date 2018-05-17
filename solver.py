import numpy as np
from scipy.linalg import hilbert


class Solver:
    def __init__(self, A, b):
        self.A = A  # square
        self.b = b
        self.n = A.shape[0]
        self.x = np.empty(self.n)

    def gaussian_elimination(self):
        a_extended = np.concatenate((self.A, self.b), axis=1)

        for i in range(self.n-1):
            col_max_idx = np.argmax(abs(a_extended[i:, i]))
            if i != col_max_idx and a_extended[i, i] < a_extended[col_max_idx, i]:  # swap
                a_extended[[i, col_max_idx]] = a_extended[[col_max_idx, i]]
            for j in range(i+1, self.n):
                if a_extended[j, i] != 0:
                    a_extended[j, :] = a_extended[j, :] - a_extended[i, :] * (a_extended[j, i]/a_extended[i, i])

        for i in range(self.n-1, -1, -1):
            for j in range(self.n-1, i, -1):
                a_extended[i, -1] -= a_extended[i, j] * self.x[j]
            self.x[i] = a_extended[i, -1]/a_extended[i, i]
        print('--------gaussian elimination--------')
        print(self.x)


if __name__ == '__main__':
    d = 4
    a = hilbert(d)
    x = np.ones((d, 1))
    solver = Solver(a, np.dot(a, x))
    solver.gaussian_elimination()
