import numpy as np 
from photonic_circuit import Circuit 
from states import PhotonicState
from optical_elements import Swap, PhaseShift
from itertools import combinations_with_replacement
from thewalrus import perm
from multi_perm import multi_perm
from get_amplitude import create_get_amp_from_out_modes, create_get_amp_from_in_modes
from math import factorial

fact_list = [factorial(n) for n in range(30)]

class Simulator:

    def __init__(self, circuit, state_threshold=1e-14):
        self.circuit = circuit 
        self.state_threshold = state_threshold
        self.N = circuit.N 
        self.photon_number = circuit.photon_number

    def fock_norm(self, modes):
        return np.prod([fact_list[modes.count(i)] for i in set(modes)], dtype=float)

    def fock_basis(self):
        """
        get all the possible fock states for the given photon number and modes
        """
        return combinations_with_replacement(range(self.N), self.photon_number)

    def heralded_fock_basis(self, detector_pattern):
        """
        get all possible states which could lead to the given detector pattern
        """
        undetected_photons = self.photon_number - sum(detector_pattern)
        undetected_modes = set(range(self.N)) - self.circuit.detected_modes

        #write down the detector outcome in terms of which modes the photons arrived 
        detector_outcome = []
        for mode, occupancy in zip(self.circuit.detected_modes, detector_pattern):
            detector_outcome.extend([mode] * occupancy)

        if undetected_photons > 0:
            #look at all options for where undetected photons could be
            undetected_outcomes = combinations_with_replacement(undetected_modes, undetected_photons)

            #combine detected and undetected outcomes
            return (tuple(sorted(detector_outcome + list(u))) for u in undetected_outcomes)
        else:
            return (tuple(detector_outcome),)

    def reduce_modes(self, modes):
        return tuple(m for m in modes if m not in self.circuit.detected_modes)

class FullUnitaryEvolution(Simulator):

    def __init__(self, circuit, state_threshold=1e-14):
        super().__init__(circuit, state_threshold)
        self.U = self.circuit.U

    def calculate_state_amplitudes(self, outcomes, reduce_state=False):
        input_state = self.circuit.global_input_state
        output_state = PhotonicState()
        for out_modes in outcomes:
            out_norm = self.fock_norm(out_modes)
            get_amp = create_get_amp_from_out_modes(self.U, out_modes)
            for in_modes, in_amp in input_state.items():
                amp = get_amp(in_modes)
                if abs(amp)**2 > self.state_threshold:
                    in_norm = self.fock_norm(in_modes)
                    if reduce_state:
                        modes = self.reduce_modes(out_modes)
                    else:
                        modes = out_modes
                    output_state[modes] += in_amp * amp * ((in_norm * out_norm) ** (-0.5))
                    if abs(output_state[modes]) ** 2 < self.state_threshold:
                        del output_state[modes]
        return output_state

    def full_output_state(self):
        """
        calculate all the terms in the quantum state at the output of the circuit
        """
        outcomes = self.fock_basis()
        return self.calculate_state_amplitudes(outcomes, reduce_state=False)

    def heralded_output_state(self, detector_pattern):
        """
        outcomes = self.heralded_fock_basis(detector_pattern)
        return self.calculate_state_amplitudes(U, outcomes, reduce_state=True)
        """
        outcomes = self.heralded_fock_basis(detector_pattern)
        return self.calculate_state_amplitudes(outcomes, reduce_state=True)

class LayeredEvolution(Simulator):
    """
    simulate circuits by evolving the input state through the circuit
    one layer at a time. This often provides for fast simulations of 
    circuits which low depths compared to the photon number.
    """

    def _evolve_swap_element(self, state, element):
        """
        evolve state through a Swap element. 

        this relabels the terms in the state according to the Swap element
        we are applying
        """
        new_state = PhotonicState()
        for in_modes, amp in state.items():
            out_modes = []
            for in_mode in in_modes:
                offset_in_mode = in_mode - element.offset 
                if offset_in_mode in element.in_modes:
                    index = element.in_modes.index(offset_in_mode)
                    out_mode = element.out_modes[index] + element.offset
                else:
                    out_mode = in_mode 
                out_modes.append(out_mode)
            new_state[tuple(sorted(out_modes))] = amp 
        return new_state

    def _evolve_phaseshift_element(self, state, element):
        """
        Applies a phase shift to the state by multiplying each term's amplitude
        by the appropriate phase
        """
        new_state = PhotonicState()
        for in_modes, in_amp in state.items():
            phases = element.global_U(N).diagonal()
            phase_shift = np.prod([phases[i] for i in in_modes])
            new_state[in_modes] = phase_shift * in_amp 
        return new_state

    def _evolve_element(self, state, element):
        """
        evolves state through an element which applies some linear optical
        unitary transformation
        """
        new_state = PhotonicState()

        # find which modes the element acts on
        element_modes = element.acting_modes
        for in_modes, in_amp in state.items():

            # find which modes in the input state will interfere
            intf_in_modes = tuple(i for i in in_modes if i in element_modes)
            # ... and which ones will not
            non_intf_modes = tuple(i for i in in_modes if i not in element_modes)

            # create the function which will calculate our permanents etc.
            get_amp = create_get_amp_from_in_modes(element.global_unitary(self.N), intf_in_modes)
            in_norm = self.fock_norm(intf_in_modes) # input state normalisation factor

            n_int = len(intf_in_modes) # number of interfering photons
            # find out where interfering photons could end up
            for intf_out_modes in combinations_with_replacement(element_modes, n_int):
                out_amp = get_amp(intf_out_modes)
                out_modes = tuple(sorted(non_intf_modes + intf_out_modes))
                # only save non-zero amplitudes
                if abs(out_amp) ** 2 > self.state_threshold:
                    out_norm = self.fock_norm(intf_out_modes)
                    new_state[out_modes] += in_amp * out_amp * ((in_norm * out_norm) ** (-0.5))
                    # delete terms where interference causes amplitude to become zero
                    if abs(new_state[out_modes]) ** 2 < self.state_threshold:
                        del new_state[out_modes]
        return new_state

    def evolve_element(self, state, element):
        """
        decides which method is appropriate for evolving the state
        """
        if isinstance(element, Swap):
            state = self._evolve_swap_element(state, element)
        elif isinstance(element, PhaseShift):
            state = self._evolve_phaseshift_element(state, element)
        else:
            state = self._evolve_element(state, element)
        return state

    def full_output_state(self):
        """
        evolves the input state through all the layers
        """
        state = self.circuit.global_input_state
        for layer in range(self.circuit.element_layers):
            #TODO: a way to update the state one layer at a time
            #instead of one element at a time might be slightly faster
            for element in self.circuit.elements[layer]:
                state = self.evolve_element(state, element)
        return state

class FeynmanPathEvolution(Simulator):
    # TODO
    pass 

class DistingEvolution(Simulator):
    """
    simulate circuits with distinguishable photons

    uses method of arXiv:1410.7687 
    """
    def __init__(self, circuit, indistinguishability=None, S_matrix=None, state_threshold=1e-14):
        super().__init__(circuit, state_threshold)
        self.U = self.circuit.U 

        if indistinguishability is not None:
            n = self.photon_number
            self.S_matrix = indistinguishability * np.ones((n, n))
            np.fill_diagonal(self.S_matrix, 1.)
        elif S_matrix is not None:
            self.S_matrix = S_matrix
        else:
            raise Exception('must provide a indistinguishability value or an S_matrix')

        self.input_state = self.circuit.global_input_state
        if len(self.input_state) > 1:
            raise Exception('input state needs to be a fock basis state')

    def output_prob(self, out_modes):
        in_modes = list(self.input_state.keys())[0]
        M = self.U[np.ix_(out_modes, in_modes)]
        in_norm = self.fock_norm(in_modes)
        out_norm = self.fock_norm(out_modes)
        return multi_perm(M, self.S_matrix) / (in_norm * out_norm)