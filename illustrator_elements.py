import abc
import matplotlib.patches as patches
import numpy as np 

class IllustratorElement(abc.ABC):
    """base class for optical illustrator
    elements"""

    mode_plot_kwargs = {'color':'k', 
                        'solid_capstyle':'round'}
    width = 1

    @abc.abstractmethod
    def draw(self, ax):
        pass 

class Box(IllustratorElement):

    min_width = 1

    def __init__(self, start, modes, 
        label='', width=1,
        colour='white', fontsize=10,
        rounded=True, pad = 0.1):

        self.start = start
        self.bottom = max(modes)
        self.top = min(modes)
        self.width = width

        self.colour = colour
        self.label = label
        self.rounded = rounded
        self.pad = pad

        self.label_kwargs = {'fontsize' : fontsize,
        'va':'center', 'ha':'center'}   

    def draw(self, ax):

        if not self.rounded:
            boxstyle = patches.BoxStyle('square', pad=self.pad)
        else:
            boxstyle = patches.BoxStyle('round', pad=self.pad)

        if self.label:
            ax.text(x=self.start+0.5*self.width, y=0.5*(self.top+self.bottom),
              s=self.label, **self.label_kwargs)

        box = patches.FancyBboxPatch(xy=(self.start+self.pad, self.top),
            width=self.width-2*self.pad, height=self.bottom-self.top, 
            boxstyle=boxstyle, fc=self.colour, ec='k')

        ax.add_artist(box)

        return ax 

class Photon(IllustratorElement):

    def __init__(self, start, mode, colour='red', r=0.15):
        self.start = start
        self.mode = mode
        self.colour = colour
        self.r = r

    def draw(self, ax):

        circle = patches.Circle(xy=(self.start+0.5*self.width, self.mode),
            radius=self.r, color=self.colour)
        ax.add_artist(circle)

        return ax
        

class Detector(IllustratorElement):

    def __init__(self, start, mode, r=0.2, stretch=1.2, pad=0.4):
        self.start = start
        self.mode = mode
        self.r = r 
        self.stretch = stretch 
        self.pad = pad

        self.width = 2*self.pad + self.stretch * self.r + 0.4

    def draw(self, ax):

        theta = np.linspace(0,0.5*np.pi)
        x0 = self.start + self.pad
        y0 = self.mode 

        y1 = [y0]
        y1 += (y0 + self.r * np.cos(theta)).tolist()
        y2 = (y0 - self.r * np.sin(theta)).tolist()
        y2 += [y0]
        x = [x0] + (x0 + self.stretch * self.r * np.sin(theta)).tolist()

        ax.fill_between(x, y1, y2[::-1], color='k')

        x_wire = np.linspace(0,0.4)
        ax.plot(x0 + self.stretch * self.r + x_wire,
            y0 + 0.5 * self.r * np.sin(25 * x_wire),
            color='k', lw=2)

        ax.plot([self.start, x0], [y0, y0], **self.mode_plot_kwargs)
        return ax

class Routing(IllustratorElement):

    def __init__(self, start, mode_starts, mode_ends, width=1):
        self.start = start 
        self.mode_starts = mode_starts
        self.mode_ends = mode_ends
        self.width = width


    def draw(self, ax):

        for mode_start, mode_end in zip(self.mode_starts, self.mode_ends):
            ax.plot([self.start, self.start + self.width],
                [mode_start, mode_end], **self.mode_plot_kwargs)

        return ax

class State(IllustratorElement):
    
    def __init__(self, start, mode, label, fontsize, width=None):
        self.start = start 
        self.mode = mode 
        self.label = r'$ \left|' + label + r'\right\rangle$'
        self.label_kwargs = {'fontsize':fontsize,
        'va':'center', 'ha':'left'}
        if width is not None:
            self.width = width

    def draw(self, ax):
        ax.text(x=self.start, y=self.mode, 
            s=self.label, **self.label_kwargs)
        return ax

class Dots(IllustratorElement):
    
    def __init__(self, start, bottom, top):
        self.start = start 
        self.top = top
        self.bottom = bottom

    def draw(self, ax):

        ax.plot([self.start, self.start],
            [self.top, self.bottom], ls=':', **self.mode_plot_kwargs)

        return ax


class Arrow(IllustratorElement):
    def __init__(self, start, modes, 
            label='', fontsize=10, width=None, pad=0.1):

        self.start = start 
        self.top = max(modes)
        self.bottom = min(modes)
        self.label = label
        self.pad = pad

        self.arrow_kwargs = dict()
        self.arrow_kwargs['length_includes_head'] = True
        self.arrow_kwargs['color'] = 'black'
        self.arrow_kwargs['head_width'] = 0.2
        # self.arrow_kwargs['arrowprops'] = {'arrowstyle':'<->'}

        self.label_kwargs = dict()
        self.label_kwargs['va'] = 'center'
        self.label_kwargs['ha'] = 'right'
        self.label_kwargs['fontsize'] = fontsize

        if width is not None:
            self.width = width 

    def draw(self, ax):

        height = self.top - self.bottom
        ax.arrow(self.start+self.pad, self.bottom, 0, height, **self.arrow_kwargs)
        ax.arrow(self.start+self.pad, self.top, 0, -height, **self.arrow_kwargs)

        if self.label:
            ax.text(self.start, 0.5*(self.top+self.bottom),
                s=self.label, **self.label_kwargs)

class CurlyBracket(IllustratorElement):

    def __init__(self, x, y, size, left=True):
        self.x = x
        self.y = y
        self.size = size 
        
        if left:
            self.text = r'${$'
        else:
            self.text = r'$}$'

    def draw(self, ax):

        ax.text(self.start, self.y, 
            s=self.text, fontsize=self.size)

        return ax


### TODO :
class Coupler(IllustratorElement):
    def __init__(self, start, modes):
        self.start = start 
        self.modes = modes

class Modulator(IllustratorElement):
    def __init__(self, start, mode,
        shape='triangle', colour='blue'):
        self.start = start 
        self.mode = mode
        self.shape = shape 
        self.colour = colour 

    def draw(self, ax):
        ax.plot([self.start, self.start+self.width],
            [self.mode, self.mode], **self.mode_plot_kwargs)
        if self.shape == 'rectangle':
            rect_height = 0.3
            rect_pad = 0.15
            rect = patches.Rectangle(
                xy=(self.start+rect_pad,self.mode-0.5*rect_height),
                width=self.width-2*rect_pad,
                height=rect_height,color=self.colour)
            ax.add_artist(rect)
        elif self.shape == 'triangle':
            tri_width = 0.2
            centre_offset = 0.15
            tri_points = np.array([
                [-tri_width,centre_offset],
                [tri_width,centre_offset],
                [0,-centre_offset]])
                # [-tri_width, -centre_offset]])
            tri_points += np.array([
                self.start+0.5*self.width,
                self.mode])

            tri = patches.Polygon(tri_points, color=self.colour)
            ax.add_artist(tri)
        return ax

        