import numpy as np
from scipy.linalg import hilbert


class Solver:
    def __init__(self, A, b):
        self.A = A  # square
        print(self.A)
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

    def jacobi_iter(self, N=25, x=None):
        if x is None:
            x = np.zeros((self.n, 1))

        D = np.diag(self.A)
        D_1 = np.diagflat(1 / D)
        L_plus_U = np.diagflat(D) - self.A

        for i in range(N):
            x = np.dot(D_1, np.dot(L_plus_U, x)) + np.dot(D_1, self.b)

        print('------jacobi--------')
        print(x)

    def gauss_seidel(self, N=25, x=None):
        if x is None:
            x = np.zeros((self.n, 1))

        # U = np.triu(self.A, 1)
        # D_minus_L = np.tril(self.A)

        for i in range(N):
            for j in range(self.n):
                r = self.b[j]
                for k in range(j):
                    r -= self.A[j, k] * x[k]
                for k in range(j+1, self.n):
                    r -= self.A[j, k] * x[k]
                x[j] = r / self.A[j, j]
            print('remain', np.dot(self.A, x) - self.b)
        print('--------gauss seidel-------')
        print(x)


if __name__ == '__main__':
    d = 2
    a = hilbert(d)
    x = np.ones((d, 1))
    solver = Solver(a, np.dot(a, x))
    # solver.gaussian_elimination()
    # solver.jacobi_iter()
    solver.gauss_seidel()
