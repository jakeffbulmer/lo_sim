import numpy as np 
from itertools import product 
from scipy.linalg import svd

amp_types = (complex, int, float, np.complex64, np.complex128, np.float64)

class QuantumState(dict):
    def __missing__(self, key):
        return 0j 

    def __repr__(self):
        term_strings = []
        for modes, amp in self.items():
            r = float(np.round(abs(amp), 2))
            phi = float(np.round(np.angle(amp)/np.pi, 2))
            amp_string = f'{r:.2f} exp(i {phi: .2f} pi) '
            ket_string = f'|{",".join(str(i) for i in modes)}>'
            term_strings.append(amp_string+ket_string)
        return ' + \n'.join(term_strings)

    @property
    def n_systems(self):
        return len(next(iter(self.keys())))

    def normalise(self):

        if len(self) == 0:
            return 0 

        total_prob = sum(abs(amp)**2 for amp in self.values())
        
        if total_prob == 0:
            return 0 
        
        norm = total_prob ** (-0.5)
        for modes, amp in self.items():
            self[modes] = norm * amp

        return total_prob 

    def overlap(self, state):
        ov = 0j
        for modes, amp in self.items():
            amp = complex(amp)
            ov += state[modes] * amp.conjugate()
        return ov

    def fidelity(self, state):
        ov = self.overlap(state)
        return abs(ov)**2

class PhotonicState(QuantumState):

    @property
    def photon_number(self):
        return super().n_systems

    @property
    def modes(self):
        all_modes = set()
        for modes in self.keys():
            all_modes.update(modes)
        return all_modes

    def is_fixed_photon_number(self):
        p = self.photon_number
        for modes in self.keys():
            if len(modes) != p:
                return False
        return True

    def logical_systems(self):

        # check for higher order occupancy terms in state
        for modes in self.keys():
            if len(set(modes)) != len(modes):
                return None

        # put one photon from each term into a different logical system
        logical_systems = dict()
        for i in range(self.photon_number):
            logical_systems[i] = tuple(sorted(set([modes[i] for modes in self.keys()])))

        # check that logical systems are disjoint
        assigned_modes = set()
        for modes in logical_systems.values():
            modes_set = set(modes)
            if not modes_set.isdisjoint(assigned_modes):
                return None
            assigned_modes.update(modes_set)

        return logical_systems

    def to_qudit_state(self):

        logical_systems = self.logical_systems()

        if logical_systems is None:
            return None, None

        state = QuditState()
        mode_groups = logical_systems.values()

        for modes, amp in self.items():
            logical_modes = []
            for mode in modes:
                mode_group = [m for m in mode_groups if mode in m][0]
                logical_modes.append(mode_group.index(mode))
            state[tuple(logical_modes)] = amp 

        return state, logical_systems

class QuditState(QuantumState):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) > 0 or len(kwargs) > 0:
            self.d = max([max(i) for i in self.keys()]) + 1
            self.n = self.get_n()
            if not all(len(modes) == self.n for modes in self.keys()):
                raise Exception('all terms in state must have n qudits')
        else:
            self.d = 0
            self.n = None

    def __setitem__(self, key, value):

        assert type(key) == tuple
        assert type(value) in amp_types
        assert type(key[0]) == int

        super().__setitem__(key, value)
        if max(key) + 1 > self.d:
            self.d = max(key) + 1
        if len(key) != self.n:
            if self.n is not None:
                raise Exception('all terms in state must have n qudits')
            else:
                self.n = self.n_systems

    def to_photonic_state(self, logical_systems):
        state = PhotonicState()
        for modes, amp in self.items():
            new_modes = tuple(logical_systems[i][mode] for i, mode in enumerate(modes))
            state[new_modes] = amp 
        return state

    def schmidt(self, a_sys=(0,), b_sys=(1,), compute_uv=False):

        if not set(a_sys).isdisjoint(set(b_sys)):
            raise Exception('a_sys and b_sys must not overlap')
        if not set(a_sys) | set(b_sys) == set(range(self.n)):
            raise Exception('all part of the state must be present')

        d_a = self.d ** len(a_sys)
        d_b = self.d ** len(b_sys)

        M = np.zeros((d_a, d_b), dtype=complex)

        for i_a, a_modes in enumerate(product(range(self.d), repeat=len(a_sys))):
            for i_b, b_modes in enumerate(product(range(self.d), repeat=len(b_sys))):
                modes = [0] * self.n 
                for i, sys_a in enumerate(a_sys):
                    modes[sys_a] = a_modes[i]
                for i, sys_b in enumerate(b_sys):
                    modes[sys_b] = b_modes[i]
                M[i_a, i_b] = self[tuple(modes)]

        return svd(M, compute_uv=compute_uv)