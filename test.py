import numpy as np
import panel as pn
import holoviews as hv
import holoviews.plotting.bokeh

points = hv.Points(np.random.randn(1000,2 )).opts(tools=['box_select', 'lasso_select'])
selection = hv.streams.Selection1D(source=points)

def selected_info(index):
    arr = points.array()[index]
    if index:
        label = 'Mean x, y: %.3f, %.3f' % tuple(arr.mean(axis=0))
    else:
        label = 'No selection'
    return points.clone(arr, label=label).opts(color='red')

layout = points + hv.DynamicMap(selected_info, streams=[selection])

pn.panel(layout).servable(title='HoloViews App')

def sine(frequency, phase, amplitude):
    xs = np.linspace(0, np.pi*4)
    return hv.Curve((xs, np.sin(frequency*xs+phase)*amplitude)).opts(width=800)

ranges = dict(frequency=(1, 5), phase=(-np.pi, np.pi), amplitude=(-2, 2), y=(-2, 2))
dmap = hv.DynamicMap(sine, kdims=['frequency', 'phase', 'amplitude']).redim.range(**ranges)

server = pn.panel(dmap).show()

# server = pn.serve(dmap, start=False, show=False)
#
# server.start()
# server.show('/')

# Outside the notebook ioloop needs to be started
# from tornado.ioloop import IOLoop
# loop = IOLoop.current()
# loop.start()
