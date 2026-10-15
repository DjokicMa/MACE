"""Legends outside the data, for every matplotlib figure ``mace plotting`` saves.

``legend_outside(fig, ax, ...)`` puts an axes' legend right of the axes and of
everything drawn beside them (tick labels, the axis label), never over the
data. The figure grows on the right when the legend needs more room than the
right margin; every axes keeps its size and place in inches, so band and DOS
figures still line up. The band, DOS and IR/Raman plotters all use it (the
GUI's plots follow the same rule: plotkit.js fitLegends).
"""


def _renderer(fig):
    return fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else fig._get_renderer()


def legend_outside(fig, ax, *args, pad_in=0.1, **kwargs):
    """The legend right of the axes (and its right-hand labels); returns it."""
    renderer = _renderer(fig)
    box = ax.get_window_extent(renderer)
    right = max(box.x1, ax.get_tightbbox(renderer).x1)
    x = (right - box.x0 + pad_in * fig.dpi) / box.width
    legend = ax.legend(*args, loc="upper left", bbox_to_anchor=(x, 1.0), borderaxespad=0.0, **kwargs)
    over = legend.get_window_extent(renderer).x1 + pad_in * fig.dpi - fig.bbox.x1
    if over > 0:
        w, h = fig.get_size_inches()
        nw = w + over / fig.dpi
        old = [a.get_position() for a in fig.axes]
        fig.set_size_inches(nw, h, forward=False)
        for a, pos in zip(fig.axes, old):
            a.set_position([pos.x0 * w / nw, pos.y0, pos.width * w / nw, pos.height])
    return legend
