import numpy as np 
from itertools import product, combinations_with_replacement
from states import PhotonicState 
from circuit_illustrator import CircuitIllustrator
from optical_elements import Swap, OpticalUnitary, PhaseShift, OpticalElement
from get_amplitude import create_get_amp
from copy import copy
from collections import defaultdict

class Circuit:
    def __init__(self, state_threshold=1e-14, illustrate=True):
      
        self.state_threshold = state_threshold
        self.used_input_modes = set()
        self.photon_number = 0
        self.global_input_state = PhotonicState()

        self.detected_modes = set()

        self.global_U = None

        self.illustrate = illustrate
        if self.illustrate:
            self.circuit_illustrator = CircuitIllustrator()

        self.elements = defaultdict(list)
        self.element_layer = 0

    @property
    def N(self):
        N = 0
        for layer in range(self.element_layer):
            for element in self.elements[layer]:
                N = max(N, element.offset + element.n)
        return N

    def _illustrate_input_state(self, state):
        state_modes = state.modes
        for mode in state_modes: 
            self.circuit_illustrator.add_photon(mode)

    def add_input_state(self, state):

        state = PhotonicState(state)

        if not state.is_fixed_photon_number():
            raise Exception('state is not fixed photon number')

        state.normalise()

        photon_number = state.photon_number
        state_modes = state.modes
        
        if not self.used_input_modes.isdisjoint(state_modes):
            raise Exception('some input modes are already occupied')

        self.used_input_modes.update(state_modes)
        self.photon_number += photon_number

        # tensor product the new input state with the existing input state
        if len(self.global_input_state) == 0:
            self.global_input_state = state 
        else:
            new_input_state = PhotonicState()
            for modes, amp in self.global_input_state.items():
                for added_modes, added_amp in state.items():
                    new_modes = tuple(sorted(modes + added_modes))
                    new_input_state[new_modes] = amp * added_amp

            self.global_input_state = new_input_state

        if self.illustrate:
            self._illustrate_input_state(state)

    def add_input_states(self, states):
        for state in states:
            self.add_input_state(state)

    def add_input_photons(self, modes, photon_numbers=None):

        if photon_numbers is None:
            for mode in modes:
                state = PhotonicState({(mode,) : 1})
                self.add_input_state(state)
        else:
            state = PhotonicState(
                {(mode,)*n : 1 for mode, n in zip(modes, photon_numbers)})
            self.add_input_state(state)

    def _illustrate_optical_element(self, optical_element, modes):
        if isinstance(optical_element, PhaseShift):
            self.circuit_illustrator.add_modulator(modes)
        elif isinstance(optical_element, OpticalUnitary):
            label = optical_element.label
            self.circuit_illustrator.add_box(modes, label)
        elif isinstance(optical_element, Swap):
            mode_starts = optical_element.in_modes
            mode_ends = optical_element.out_modes
            route_offset = optical_element.offset
            self.circuit_illustrator.add_route(mode_starts, 
                mode_ends, route_offset)
        else:
            raise Exception('illustration not implemented for this element')
                
    def add_optical_elements(self, *optical_elements):
        offset = 0
        for optical_element in optical_elements:

            optical_element = copy(optical_element)

            if not isinstance(optical_element, OpticalElement):
                raise Exception('all objects must be an OpticalElement')

            if optical_element.offset is None:
                optical_element.offset = offset 

            top_mode = optical_element.offset
            offset = top_mode + optical_element.n
            modes = range(top_mode, offset)

            self.elements[self.element_layer].append(optical_element)

            if self.illustrate:
                self._illustrate_optical_element(optical_element, modes)
        self.element_layer += 1

    def add_detectors(self, modes):
        self.detected_modes.update(modes)
        if self.illustrate:
            for mode in modes:
                self.circuit_illustrator.add_detector(mode)

    def evaluate_global_U(self):
        N = self.N
        U = np.eye(N, dtype=complex)
        for layer in range(self.element_layer):
            layer_elements = self.elements[layer]
            for element in layer_elements:
                U = element.global_unitary(N) @ U
        self.global_U = U

    def _evolve_step_swap(self, state, element):
        new_state = PhotonicState()
        # if element is a swap, we just need to relabel the state
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


    def _evolve_step_phaseshift(self, state, element):
        new_state = PhotonicState()
        # if element is a phase shift, we mutliply each term by a phase
        for in_modes, in_amp in state.items():
            phases = element.global_U(N).diagonal() 
            phase_shift = np.prod([phases[i] for i in in_modes])
            new_state[in_modes] = phase_shift * in_amp
        return new_state

    def _evolve_step(self, state, element):
        N = self.N 
        new_state = PhotonicState()
        # we figure out where the photons could end up if they go into the element
        element_modes = set(range(element.offset, element.offset+element.n))
        for in_modes, in_amp in state.items():

            # find which input modes can interfere
            intf_in_modes = tuple(i for i in in_modes if i in element_modes)
            # find which input modes can't interfere
            non_intf_modes = tuple(i for i in in_modes if i not in element_modes)

            # create function which will calculate the permanent
            get_amp = create_get_amp(element.global_unitary(N), intf_in_modes, in_amp)

            n_int = len(intf_in_modes) # number of interfering photons
            # find where interfering photons could end up
            for intf_out_modes in combinations_with_replacement(element_modes, n_int):
                out_amp = get_amp(intf_out_modes)
                out_modes = tuple(sorted(non_intf_modes + intf_out_modes))
                if abs(out_amp)**2 > self.state_threshold:
                    new_state[out_modes] += out_amp
                    if abs(new_state[out_modes]) ** 2 < self.state_threshold:
                        del new_state[out_modes]
        return new_state

    def evolve_step(self, state, element):
        if isinstance(element, Swap):
            state = self._evolve_step_swap(state, element)
        elif isinstance(element, PhaseShift):
            state = self._evolve_step_phaseshift(state, element)
        else:
            state = self._evolve_step(state, element)
        return state

    def evolve_in_steps(self):
        state = self.global_input_state
        for layer in range(self.element_layer):
            layer_elements = self.elements[layer]
            for element in layer_elements:
                state = self.evolve_step(state, element)
        return state

    def evolve(self):
        if self.global_U is None:
            self.evaluate_global_U()
        # get amplitudes for all possible output states
        outcomes = combinations_with_replacement(range(self.N), self.photon_number)
        return self.calculate_new_amplitudes_full_U(outcomes)

    def get_detector_pattern_outcomes(self, detector_pattern):
        """
        get all possible photon outcome patterns which are consistent with a detection pattern
        """

        undetected_photons = self.photon_number - sum(detector_pattern)
        undetected_modes = set(range(self.global_U.shape[0])) - self.detected_modes

        #write down the detector outcome in terms of which modes the photons arrived 
        detector_outcome = []
        for mode, occupancy in zip(self.detected_modes, detector_pattern):
            detector_outcome.extend([mode] * occupancy)

        if undetected_photons > 0:
            #look at all options for where undetected photons could be
            undetected_outcomes = combinations_with_replacement(undetected_modes, undetected_photons)

            #combine detected and undetected outcomes
            return (tuple(sorted(detector_outcome + list(u))) for u in undetected_outcomes)
        else:
            return (tuple(detector_outcome),)


    def evolve_to_detector_pattern(self, detector_pattern,
        normalise=True, reduce_state=True):

        if self.global_U is None:
            self.evaluate_global_U()
    
        if len(detector_pattern) != len(self.detected_modes):
            raise Exception(
                f'detector_pattern needs to have {len(self.detected_modes)} modes' +
                f'but {len(detector_pattern)} were given')

        if np.sum(detector_pattern) > self.photon_number:
            raise Exception('more photons in detector pattern than in the state')

        outcomes = self.get_detector_pattern_outcomes(detector_pattern)
        return self.calculate_new_amplitudes_full_U(outcomes, reduce_state=True)

    def calculate_new_amplitudes_full_U(self, outcomes, reduce_state=False):
        
        output_state = PhotonicState()
        outcomes = list(outcomes)

        for in_modes, in_amp in self.global_input_state.items():
            get_amp = create_get_amp(self.global_U, in_modes, in_amp)

            outcomes_amps = [get_amp(out_modes) for out_modes in outcomes]

            for out_modes, out_amp in zip(outcomes, outcomes_amps):
                if abs(out_amp) ** 2 > self.state_threshold:
                    output_state[tuple(out_modes)] += out_amp

        tidy_out_state = PhotonicState()
        for out_modes, out_amp in output_state.items():
            if abs(out_amp) ** 2 > self.state_threshold:
                if reduce_state:
                    modes = tuple([mode for mode in out_modes if mode not in self.detected_modes])
                else:
                    modes = tuple(out_modes)
                tidy_out_state[modes] = out_amp 

        return tidy_out_state

    def gen_detector_patterns(self, detected_photons):
        n_det = len(self.detected_modes)
        for modes in combinations_with_replacement(range(n_det), detected_photons):
            pattern = np.zeros(n_det, dtype=int)
            np.add.at(pattern, np.asarray(modes), 1)
            yield pattern

    def draw(self, extend_output_modes=2):
        if not self.illustrate:
            raise Exception('circuit illustration disabled')
        remaining_modes = self.circuit_illustrator.active_modes
        if len(remaining_modes) > 0:
            self.circuit_illustrator.add_route(remaining_modes, width=extend_output_modes)
        return self.circuit_illustrator.draw()