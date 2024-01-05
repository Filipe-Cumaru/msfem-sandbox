import numpy as np
from scipy.sparse import csc_matrix


class BaseTwoLevelASPreconditioner(object):
    def __init__(self, A, Phi, N, n, k) -> None:
        self.A = A
        self.Phi = Phi
        self.N = N
        self.n = n
        self.k = k

        self.m = self.N * self.n + 1
        self.H = 1 / self.N
        self.h = 1 / self.n
        self.P = None

    def assemble(self):
        raise NotImplementedError()


class TwoLevelASPreconditioner(BaseTwoLevelASPreconditioner):
    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)

    def assemble(self):
        return super().assemble()


class TwoLevelRASPreconditioner(BaseTwoLevelASPreconditioner):
    def __init__(self, A, Phi, N, n, k) -> None:
        super().__init__(A, Phi, N, n, k)
