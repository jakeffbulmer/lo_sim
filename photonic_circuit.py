import numpy as np 
from itertools import product, combinations_with_replacement, combinations
from states import PhotonicState 
from circuit_illustrator import CircuitIllustrator
from optical_elements import Swap, OpticalUnitary, PhaseShift, OpticalElement
from copy import copy
from collections import defaultdict

class Circuit:
    """
    Class for describing photonic circuits. It will store information
    about the optical elements of the circuit and the input state.
    It is also integrated with the CircuitIllustrator class 
    to provide a simple way of drawing the circuit contained.
    """
    def __init__(self, illustrate=True):
        self.used_input_modes = set()
        self.photon_number = 0
        self.global_input_state = PhotonicState()

        self.detected_modes = set()

        self.illustrate = illustrate
        if self.illustrate:
            self.circuit_illustrator = CircuitIllustrator()

        self.elements = defaultdict(list)
        self.element_layers = 0

    @property
    def N(self):
        """
        number of modes in the whole circuit
        """
        N = 0
        for modes in self.global_input_state.keys():
            N = max(N, max(modes)+1)
        for layer in range(self.element_layers):
            for element in self.elements[layer]:
                N = max(N, element.offset + element.n)
        return N

    @property
    def U(self):
        """
        gives the unitary of the whole circuit
        """
        N = self.N
        U = np.eye(N, dtype=complex)
        for layer in range(self.element_layers):
            layer_elements = self.elements[layer]
            for element in layer_elements:
                U = element.global_unitary(N) @ U
        return U

    def _illustrate_input_state(self, state):
        # TODO: something more cool for input states in superposition
        state_modes = state.modes
        for mode in state_modes: 
            self.circuit_illustrator.add_photon(mode)

    def add_input_state(self, state):
        """
        add a state to the input state
        """
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
        """
        add multiple input state
        """
        for state in states:
            self.add_input_state(state)

    def add_input_photons(self, modes, photon_numbers=None):
        """
        add fock states to the input state
        If photon_numbers is None, assumes that all 'modes' contain
        a single photon.
        Otherwise, modes and photon_numbers should be the same length 
        and photon_numbers gives the occupancy of the respective mode
        """
        if photon_numbers is None:
            for mode in modes:
                state = PhotonicState({(mode,) : 1})
                self.add_input_state(state)
        else:
            state = PhotonicState(
                {(mode,)*n : 1 for mode, n in zip(modes, photon_numbers)})
            self.add_input_state(state)

    def _illustrate_optical_element(self, optical_element, modes):
        """
        adds the appropriate components to the circuit_illustrator class
        """
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
                
    def add_optical_layer(self, *optical_elements):
        """
        add a layer of optical elements

        optical_elements should all be instances of an OpticalElement class
        if no 'offset' is specified for the optical element, it will be placed 
        directly below the previously element (or at the top of the circuit).
        """
        offset = 0
        for optical_element in optical_elements:

            optical_element = copy(optical_element)

            if not isinstance(optical_element, OpticalElement):
                raise Exception('all objects must be an OpticalElement')

            if optical_element.offset is None:
                optical_element.offset = offset
            else:
                if optical_element.offset < offset:
                    # advance to a new layer if element overlaps with 
                    # previous elements in the layer
                    self.element_layers += 1

            top_mode = optical_element.offset
            offset = top_mode + optical_element.n
            modes = range(top_mode, offset)

            self.elements[self.element_layers].append(optical_element)

            if self.illustrate:
                self._illustrate_optical_element(optical_element, modes)
        self.element_layers += 1

    def add_detectors(self, modes):
        """
        add detectors to the 'modes' specified
        """
        self.detected_modes.update(modes)
        if self.illustrate:
            for mode in modes:
                self.circuit_illustrator.add_detector(mode)

    def gen_detector_patterns(self, detected_photons):
        """
        generate all the possible detector patterns for a given number
        of 'detected_photons'
        """
        n_det = len(self.detected_modes)
        for modes in combinations_with_replacement(range(n_det), detected_photons):
            pattern = np.zeros(n_det, dtype=int)
            np.add.at(pattern, np.asarray(modes), 1)
            yield pattern

    def gen_constrained_detector_patterns(self, det_group_sizes, photon_numbers=None,
        only_single_click=False):
        """
        pass in two lists describing how many detectors are in each group and how
        many photons each group needs to detect in total
        """

        n_det = len(self.detected_modes)

        if photon_numbers is None:
            photon_numbers = [1] * len(det_group_sizes)

        if sum(photon_numbers) > self.photon_number:
            raise Exception('more photons than in state')

        if sum(det_group_sizes) != n_det:
            raise Exception('group sizes do not cover all detectors')

        det_groups = dict()
        mode = 0
        for size, n in zip(det_group_sizes, photon_numbers):
            group = tuple(range(mode, mode+size))
            det_groups[group] = n
            mode += size

        if not only_single_click:
            group_gen = combinations_with_replacement
        else:
            group_gen = combinations 

        for groups_modes in product(*(
            group_gen(group, photons) 
                for group, photons in det_groups.items())):

            modes = tuple(i for m in groups_modes for i in m)
            pattern = np.zeros(n_det, dtype=int)
            np.add.at(pattern, np.asarray(modes), 1)
            yield pattern

    def draw(self, extend_output_modes=2):
        """
        draw the circuit.
        returns a matplotlib fig, ax 

        extended_output_modes: length of the path to draw for the non
        detected modes at the end of the circuit.
        """
        if not self.illustrate:
            raise Exception('circuit illustration disabled')
        remaining_modes = self.circuit_illustrator.active_modes
        if len(remaining_modes) > 0:
            self.circuit_illustrator.add_route(remaining_modes, width=extend_output_modes)
        return self.circuit_illustrator.draw()