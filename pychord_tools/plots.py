import math
from sklearn import preprocessing
from collections import Counter
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from scipy.stats import beta
from .compositional_data import amalgamate
from .labels import DEGREES


class TernaryDensity:
    def __init__(self, steps=10):
        self.steps = steps
        self.triangles = Counter()
        self.total = 0

    def triangle_id(self, x, y, z):
        xi = int(float(self.steps) * x)
        yi = int(float(self.steps) * y)
        zi = int(float(self.steps) * z)
        return xi, yi, zi

    def add_point(self, x, y, z):
        self.triangles[self.triangle_id(x, y, z)] += 1
        self.total += 1

    def triangle_value(self, n):
        return float(self.triangles[n]) / self.total

    def to_screen_xy(self, triple):
        return [0.5 * (2 * triple[1] + triple[2]) / sum(triple),
                0.5 * math.sqrt(3.0) * triple[2] / sum(triple)]

    def patch_collection(self, x0=0.5, y0=math.sqrt(3) / 2, angle=-4.0 * math.pi / 3.0, gap=0.0):
        # working in 3D coordinates
        triangles = []
        for i in range(self.steps):
            xi = float(i) / self.steps
            line1 = []
            for j in range(self.steps - i + 1):
                yi = float(j) / self.steps
                line1.append([xi, yi, 1.0 - xi - yi])
            line2 = []
            xi += 1.0 / self.steps
            for j in range(self.steps - i):
                yi = float(j) / self.steps
                line2.append([xi, yi, 1.0 - xi - yi])
            # line 1 triangles
            for j in range(len(line1) - 1):
                triangles.append([line1[j], line1[j + 1], line2[j]])
            # line 2 triangles
            for j in range(len(line2) - 1):
                triangles.append([line2[j], line2[j + 1], line1[j + 1]])
        # TODO: shift & other transforms: rotate/compress(?)
        # b = [self.x1, self.y1]
        patches = []
        colors = np.empty(len(triangles))
        i = 0
        initial_shift = np.array([gap, gap])
        b = np.array([x0, y0])
        c = math.cos(angle)
        s = math.sin(angle)
        r = np.array([[c, -s], [s, c]])
        for triangle in triangles:
            x = np.array(
                [self.to_screen_xy(triangle[0]),
                 self.to_screen_xy(triangle[1]),
                 self.to_screen_xy(triangle[2])])
            x += initial_shift
            x = x.dot(r)
            x += b
            polygon = Polygon(x, True)
            center = np.average(triangle, axis=0)
            id = self.triangle_id(center[0], center[1], center[2])
            patches.append(polygon)
            colors[i] = self.triangles[id]
            i += 1
        p = PatchCollection(patches, facecolor='orange', cmap=matplotlib.cm.jet, alpha=1.0)
        p.set_array(colors)
        return p


def degrees_ternary_plot(ax, chroma, d1, d2, d3, steps_resolution=50,
                         x0=0.5, y0=math.sqrt(3)/2, angle=2 * math.pi / 3.0, gap=0.0):
    d = preprocessing.normalize(chroma[:, (DEGREES.index(d1), DEGREES.index(d2), DEGREES.index(d3))], norm='l1')
    t = TernaryDensity(steps_resolution)
    for x in d:
        t.add_point(x[0], x[1], x[2])
    ax.add_collection(t.patch_collection(x0, y0, angle, gap))


def ternary_plot(ax, data, steps_resolution=50,
                 x0=0.5, y0=math.sqrt(3)/2, angle=2 * math.pi / 3.0, gap=0.0):
    t = TernaryDensity(steps_resolution)
    for x in data:
        t.add_point(x[0], x[1], x[2])
    ax.add_collection(t.patch_collection(x0, y0, angle, gap))


def plot_labels(ax, degrees, x0=0.5, y0=math.sqrt(3) / 2, angle=2 * math.pi / 3.0, size=15):
    x = np.zeros((len(degrees), 2), dtype='float64')
    x[0,:] = [0, 0]
    v = np.array([1.0, 0.0])
    x[1, :] = v
    c = math.cos(-math.pi/3.0)
    s = math.sin(-math.pi/3.0)
    r = np.array([[c, -s], [s, c]])
    for i in range(2, len(degrees)):
        v = v.dot(r)
        x[i, :] = v
    b = np.array([x0, y0], dtype='float64')
    c = math.cos(angle)
    s = math.sin(angle)
    r = np.array([[c, -s], [s, c]])
    x = x.dot(r)
    x += b
    bbox_props = dict(boxstyle="circle", fc="w", ec="0.5", alpha=0.75)
    for i in range(len(degrees)):
        ax.text(x[i, 0], x[i, 1], degrees[i], ha="center", va="center", size=size,
                bbox=bbox_props)


def plot_hexagram(ax, chroma, degrees, step=30, gap=0.005, label_size=12, caption_degrees=None):
    if caption_degrees is None:
        caption_degrees = degrees
    ax.axes.set_xlim(-0.5, 1.5)
    ax.axes.set_ylim(0, 1.75)
    angle = 2 * math.pi / 3.0
    for i in range(6):
        degrees_ternary_plot(
            ax,
            chroma,
            degrees[0], degrees[i + 1], degrees[(i+1) % 6 + 1],
            step, angle=angle, gap=gap)
        angle -= math.pi / 3.0
    plot_labels(ax, caption_degrees, size=label_size)


def plot_maj_hexagram(ax, chroma, step=30, gap=0.005, label_size=12):
    plot_hexagram(ax, chroma, degrees=['I', 'V', 'III', 'VI', 'VII', 'II', 'IV'],
                  step=step, gap=gap, label_size=label_size)


def plot_min_hexagram(ax, chroma, step=30, gap=0.005, label_size=12):
    plot_hexagram(ax, chroma, degrees=['I', 'V', 'IIIb', 'VIIb', 'IV', 'II', 'VI'],
                  step=step, gap=gap, label_size=label_size)


def plot_dom_hexagram(ax, chroma, step=30, gap=0.005, label_size=12):
    plot_hexagram(ax, chroma, degrees=['I', 'V', 'III', 'VIIb', 'II', 'IV', 'VI'],
                  step=step, gap=gap, label_size=label_size)


def plot_hdim7_hexagram(ax, chroma, step=30, gap=0.005, label_size=12):
    plot_hexagram(ax, chroma, degrees=['I', 'IIIb', 'VIIb', 'Vb', 'IV', 'V', 'II'],
                  step=step, gap=gap, label_size=label_size)


def plot_dim_hexagram(ax, chroma, step=30, gap=0.005, label_size=12):
    plot_hexagram(
        ax,
        chroma,
        degrees=['I', 'Vb', 'IIIb', 'VI', 'IIb', 'VII', 'VIb'],
        step=step,
        gap=gap,
        label_size=label_size,
        caption_degrees=['I', 'Vb', 'IIIb', 'VIIbb', 'IIb', 'VII', 'VIb'])


# sort chord/scale degrees according to method ('mean', 'entropy', 'beta-likelihood')
def sorted_degrees(chromas, method='mean', flip=False, convert_to_indices=False):
    av = np.mean(chromas, axis=0)
    t = np.empty(len(DEGREES),
                 dtype=[('degree', object), ('entropy', float), ('mean', float), ('beta-likelihood', float)])
    for i in range(len(DEGREES)):
        partition = [i]
        a = amalgamate([partition], chromas).transpose()[0]
        params = beta.fit(a, floc=0, fscale=1)
        e = beta.entropy(*params)
        bl = beta.logpdf(a, *params).sum()
        t[i] = (DEGREES[i], e, av[i], bl)
    t.sort(order=method)
    d = t['degree']
    if flip:
        d = np.flip(d, axis=0)
    if convert_to_indices:
        return [DEGREES.index(x) for x in d]
    else:
        return d


def plot_strong_weak_hexagrams(
        ax1,
        ax2,
        chromas,
        sorted_degrees,
        step=30, gap=0.005, label_size=12):
    weakest = np.empty(7, dtype='object')
    weakest[0:7] = sorted_degrees[0:7]
    # weakest[0] = sortedDegrees[6]
    strongest = np.empty(7, dtype='object')
    strongest[0] = sorted_degrees[11]
    strongest[1:7] = sorted_degrees[5:11]
    plot_hexagram(ax1, chromas, weakest, step=step, gap=gap, label_size=label_size)
    plot_hexagram(ax2, chromas, strongest, step=step, gap=gap, label_size=label_size)
