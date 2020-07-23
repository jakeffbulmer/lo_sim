import numpy as np 
from scipy.linalg import block_diag
import abc
from itertools import chain

class OpticalElement(abc.ABC):

    @abc.abstractmethod
    def global_unitary(self, N):
        # this method must be implemented by all
        # subclasses
        pass

    @property
    def acting_modes(self):
        return set(range(self.offset, self.offset+self.n))

class OpticalUnitary(OpticalElement):
    def __init__(self, U, offset=None, label=''):
        self.U = U 
        self.n = U.shape[0]
        self.offset = offset 
        self.label = label

    def global_unitary(self, N):
        global_U = np.eye(N, dtype=complex)
        start = self.offset
        stop = self.offset + self.n 
        global_U[start:stop, start:stop] = self.U

        return global_U

class Swap(OpticalElement):
    def __init__(self, in_modes, out_modes, offset=None):

        assert len(in_modes) == len(out_modes)
        self.n = len(in_modes)

        self.in_modes = in_modes
        self.out_modes = out_modes

        self.offset = offset

    def global_unitary(self, N):
        U = np.zeros((self.n, self.n), dtype=complex)
        U[self.out_modes, self.in_modes] = 1. 

        U = block_diag(np.eye(self.offset), U, np.eye(N-self.offset-self.n))
        return U

class BS(OpticalUnitary):
    def __init__(self, R=0.5, offset=None, label=''):
        if not label:
            label = r'$B(R)$'
        theta = 2 * np.arccos(np.sqrt(R))
        U = np.array([
            [np.cos(0.5 * theta), 1j * np.sin(0.5 * theta)],
            [1j * np.sin(0.5 * theta), np.cos(0.5 * theta)]])
        super().__init__(U, offset, label)

class PhaseShift(OpticalUnitary):
    def __init__(self, phases, offset=None):
        U = np.diag(np.exp(1j * np.atleast_1d(phases)))
        super().__init__(U, offset)

class DFT(OpticalUnitary):
    def __init__(self, d, offset=None, label=''):
        if not label:
            label = r'$DFT_{{{}}}$'.format(d)
        U = (d ** (-0.5)) * np.fft.fft(np.eye(d))
        super().__init__(U, offset, label)

class I(Swap):
    def __init__(self, n, offset=None):
        in_modes = range(n)
        out_modes = range(n)
        super().__init__(in_modes, out_modes, offset)

class SwapOut(Swap):
    def __init__(self, d, n, offset=None):
        in_modes = range(n * d)
        out_modes = list(chain.from_iterable(
            range(i,n*d,n) for i in range(n)))
        super().__init__(in_modes, out_modes, offset)