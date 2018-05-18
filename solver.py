import numpy as np
from scipy.linalg import hilbert
from numpy.linalg import norm


class Solver:
    def __init__(self, A, b):
        self.A = A  # square
        # print(self.A)
        self.b = b
        assert np.any(self.b)
        self.n = A.shape[0]
        self.x = np.empty(self.n)

    def gaussian_elimination(self):
        a_extended = np.concatenate((self.A, self.b), axis=1)

        for i in range(self.n-1):
            col_max_idx = np.argmax(abs(a_extended[i:, i])) + i  # take care for index
            if i != col_max_idx and abs(a_extended[i, i]) < abs(a_extended[col_max_idx, i]):  # swap
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

    def jacobi_iter(self, N=1000, x=None):
        if x is None:
            x = np.zeros((self.n, 1))

        D = np.diag(self.A)
        D_1 = np.diagflat(1 / D)
        L_plus_U = np.diagflat(D) - self.A

        for i in range(N):
            x = np.dot(D_1, np.dot(L_plus_U, x)) + np.dot(D_1, self.b)

        print('------jacobi--------')
        print(x)

    def gauss_seidel(self, N=1000, x=None):
        if x is None:
            x = np.zeros(self.n)

        # U = np.triu(self.A, 1)
        # D_minus_L = np.tril(self.A)

        for _ in range(N):
            for i in range(self.n):
                s = sum(-self.A[i, j] * x[j] for j in range(self.n) if i != j)
                x[i] = (s + self.b[i, 0]) / self.A[i, i]
        # access element not sub-matrix (for example: b is two dim)
        print('--------gauss seidel-------')
        print(x)

    def sor(self, N=1000, w=1.8, x=None):
        if x is None:
            x = np.zeros(self.n)
        x_old = np.empty_like(x)
        x_old[:] = x

        for _ in range(N):
            for i in range(self.n):
                s = sum(-self.A[i, j] * x[j] for j in range(self.n) if i != j)
                x[i] = (s + self.b[i, 0]) / self.A[i, i]
            x = w * x + (1 - w) * x_old
            x_old = x

        print('--------SOR w={}-------'.format(w))
        print(x)

    def conj_grad(self, N=1000, x=None, tol=1e-10):
        if x is None:
            x = np.zeros(self.n)
        a_diag = np.diag(self.A)
        M = np.diagflat(a_diag)
        r = self.b.reshape(self.n) - np.dot(self.A, x)
        r_old = np.empty_like(r)
        r_old[:] = r
        M_inv = np.diagflat(1/a_diag)
        z = np.dot(M_inv, r)
        z_old = np.empty_like(z)
        z_old[:] = z
        p = z
        for k in range(N):
            a_dot_p = np.dot(self.A, p)
            alpha = np.dot(z, r) / np.dot(p, a_dot_p)
            x = x + alpha * p
            r = r_old - alpha * a_dot_p
            err = norm(r) / norm(self.b)
            if err <= tol:
                print('final', x)
                break
            z = np.dot(M_inv, r)
            belta = np.dot(z, r) / np.dot(z_old, r_old)
            p = z + belta * p
            r_old, z_old = r, z
        print('----------conj grad------------')
        print(x)


if __name__ == '__main__':
    d = 100
    a = hilbert(d)
    x = np.ones((d, 1))
    solver = Solver(a, np.dot(a, x))
    # solver.gaussian_elimination()
    # solver.jacobi_iter()
    # solver.gauss_seidel()
    # solver.sor()
    solver.conj_grad()
