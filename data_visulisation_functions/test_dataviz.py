# from pathlib import Path
# ROOT = Path(__file__).parent.parent.parent.parent

import numpy as np
import numpy.random as nr
from datoviz import canvas, run, colormap
import random
#
# c = canvas(show_fps=False)
#
# panel = c.panel(controller='axes')
# visual = panel.visual('point')

N = 100
x = np.random.random((N,N))
y = x
print(x)
zero = x


pos = nr.randn(x, y, y)
# color_values = nr.rand(N)
# color = colormap(color_values, vmin=0, vmax=1, alpha=.75 * np.ones(N), cmap='viridis')

# x = np.array([1,2,3,5])
# y = np.array([1,2,3,4])

# print(x)

# pos = np.c_[x, y, np.zeros(len(x))]
# color = colormap(y, cmap='glasbey', alpha=.5)

visual.data('pos', pos)
# visual.data('color', color)
visual.data('ms', np.array([2.]))

run(screenshot="screenshot.png")
