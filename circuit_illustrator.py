import numpy as np 
import matplotlib.pyplot as plt 
from illustrator_elements import (Box, Photon, Dots, Detector,
	 Routing, State, Arrow, Dots, CurlyBracket, Modulator)
from collections import defaultdict

class CircuitIllustrator():
	def __init__(self, 
		fontsize=10,
		width_scaling=0.3,
		input_pad=0.2,
		photon_r=0.15,
		box_width=1.5,
		ignore_vacuum=True):

		self.fontsize = fontsize
		self.width_scaling = width_scaling
		self.input_pad = input_pad
		self.photon_r = photon_r
		self.box_width = box_width

		self.ignore_vacuum = ignore_vacuum

		self.elements = []
		self.active_modes = set()
		self.mode_distance = defaultdict(float)

	def add_box(self, modes, label='',
		colour='white', rounded=True, pad=0.1,
		input_pad=None):

		if input_pad is None:
			input_pad = self.input_pad

		if not any(m in self.active_modes for m in modes):
			raise Exception(
				'at least one of the inputs to the box should be active')

		self.active_modes.update(modes)

		if input_pad > 0:
			self.add_identity(modes, input_pad)

		start = max(self.mode_distance[m] for m in modes)

		box = Box(start, modes, label, self.box_width,
			colour, self.fontsize, rounded, pad)

		self.elements.append(box)
		for mode in modes:
			self.mode_distance[mode] = start + box.width

		# add routes to modes which are not at the box input
		for mode in modes:
			if mode in self.active_modes:
				if self.mode_distance[mode] < start:
					self.add_route([mode], [mode], self.mode_distance[mode] - start)

	def add_photon(self, mode, colour='red'):

		if mode in self.active_modes:
			raise Exception('photon must be created on an inactive mode')

		start = self.mode_distance[mode]

		photon = Photon(start, mode, colour, self.photon_r)

		self.elements.append(photon)
		self.active_modes.add(mode)
		self.mode_distance[mode] = start + photon.width

	def add_photons(self, modes, colour='red'):
		for mode in modes:
			self.add_photon(mode, colour)

	def add_identity(self, modes, width=1):

		for mode in modes:
			if mode not in self.active_modes:
				raise Exception('route must start on active modes')

		start = max(self.mode_distance[m] for m in modes)

		route = Routing(start, modes, modes, width)

		self.elements.append(route)
		self.mode_distance[mode] = start + route.width

	def add_route(self, mode_starts, mode_ends=None, offset=0,
		width=None, input_pad=None):

		if input_pad is None:
			input_pad = self.input_pad

		if mode_ends is None:
			mode_ends = mode_starts

		mode_starts = [m + offset for m in mode_starts]
		mode_ends = [m + offset for m in mode_ends]

		plotted_mode_starts = []
		plotted_mode_ends = []
		for mode_start, mode_end in zip(mode_starts, mode_ends):
			if mode_start in self.active_modes:
				plotted_mode_starts.append(mode_start)
				plotted_mode_ends.append(mode_end)
			else:
				if not self.ignore_vacuum:
					raise Exception('route must start on active modes')
		mode_starts, mode_ends = plotted_mode_starts, plotted_mode_ends

		if len(mode_starts) == 0:
			return 

		start = max(self.mode_distance[m] for m in mode_starts + mode_ends)
		
		for mode in mode_starts:
			if self.mode_distance[mode] < start:
				self.add_identity([mode], start - self.mode_distance[mode])

		if self.input_pad > 0:
			self.add_identity(mode_starts, self.input_pad)
			start += self.input_pad

		longest_route = max(abs(a - b) for a, b in zip(mode_starts, mode_ends))

		if width is None:
			width = longest_route * self.width_scaling

		route = Routing(start, mode_starts, mode_ends, width)

		self.elements.append(route)
		self.active_modes.difference_update(mode_starts)
		self.active_modes.update(mode_ends)
		for mode in mode_ends:
			self.mode_distance[mode] = start + route.width

	def add_detector(self, mode, r=0.2, stretch=1.8):

		if mode not in self.active_modes:
			raise Exception('detector must appear on an active mode')

		start = self.mode_distance[mode]

		detector = Detector(start, mode, r, stretch)

		self.elements.append(detector)
		self.active_modes.remove(mode)
		self.mode_distance[mode] = start + detector.width

	def add_detectors(self, modes, r=0.2, stretch=1.4):
		for mode in modes:
			self.add_detector(mode, r, stretch)

	def add_state(self, mode, label, width=None):

		start = self.mode_distance[mode]

		state = State(start, mode, label, self.fontsize, width)

		self.elements.append(state)
		self.active_modes.symmetric_difference_update({mode})
		self.mode_distance[mode] = start + state.width

	def add_arrow(self, modes, label='', width=0.2):

		start = max(self.mode_distance[m] for m in modes)

		arrow = Arrow(start, modes, label, self.fontsize, width)

		self.elements.append(arrow)
		for mode in modes:
			self.mode_distance[mode] = start + arrow.width

	def add_dots(self, bottom, top, offset=0.5):

		modes = range(int(np.floor(bottom)), int(np.ceil(top)))

		start = max(self.mode_distance[int(m)] for m in modes) + offset

		dots = Dots(start, bottom, top)

		self.elements.append(dots)

	def add_curly_bracket(self, x, y, size=30, left=True):

		curly_bracket = CurlyBracket(x, y, size, left)

		self.elements.append(curly_bracket)

	def add_pair_source(self, modes, 
		colour='gray', box_width=0.5,
		photon1_colour='red', photon2_colour='blue'):

		start = max(self.mode_distance[m] for m in modes)

		source_box = Box(start, modes,
			colour=colour, rounded=False, width=box_width)
		photon_start = start + source_box.width
		source_photon1 = Photon(photon_start, mode[0], colour='photon1_colour')
		source_photon2 = Photon(photon_start, mode[-1], colour='photon2_colour')

		for mode in modes:
			self.mode_distance[mode] = photon_start + source_photon1.width
		self.elements.extend([source_box, source_photon1, source_photon2])
		self.active_modes.update({modes[0], modes[-1]})

	def add_modulator(self, modes, shape='triangle', colour='blue'):

		start = max(self.mode_distance[m] for m in modes)

		for mode in modes:

			if mode not in self.active_modes:
				raise Exception('modulator modes must be active')

			modulatator = Modulator(start, mode, shape, colour)
			self.elements.append(modulatator)
			self.mode_distance = start + modulator.width

	def draw(self, x_scale=0.4, y_scale=0.4, fig=None, ax=None):

		xlims = (-0.5, max(self.mode_distance.values()) + 0.5)

		ylims = (min(self.mode_distance.keys()) - 0.5,
				max(self.mode_distance.keys()) + 0.5)

		fig_width = xlims[1] - xlims[0]
		fig_height = ylims[1] - ylims[0]

		figsize = (fig_width * x_scale, fig_height * y_scale)

		if ax is None and fig is None:
			fig, ax = plt.subplots(figsize=figsize)

		ax.set_xlim(xlims)
		ax.set_ylim(ylims)
		ax.axis('off')
		ax.invert_yaxis()
		ax.set_aspect('equal')

		for element in self.elements:
			element.draw(ax)

		return fig, ax






