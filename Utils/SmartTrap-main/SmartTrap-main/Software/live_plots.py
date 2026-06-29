"""
This files defines the various tools used to plot data from the SmartTrap system in real-time.
-----------------------------------------------------
Classes:

- PlotLengthWindow: A QWidget window which lets users define the length of plots.
- PlotSubsamplehWindow: A QWidget window which lets users define how many samples to average in the 
real-time plot.
- PlotAxisWindow: A QWidget window which lets users manually set the plot-axis limits with text
input.
- PlotWindow: A QMainWindow used for the live-plots (graphs) .
"""

from PyQt6.QtWidgets import (
    QMainWindow, QLineEdit, QLineEdit, QToolBar, QVBoxLayout, QWidget, QLabel
)

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QDoubleValidator, QIntValidator

import pyqtgraph as pg
from random import randint

import numpy as np
from functools import partial

Colors = {
    'red': (255,0,0),
    'blue': (0,0,255),
    'orange': (255,165,0),
    'navy blue': (0,0,128),
    'green': (0,255,0),
    'gray': (150,150,150),
    'white': (255,255,255),
    'black': (0,0,0),
    'cyan': (0,255,255),
    'yellow': (255,255,0),
    'pink': (255,192,203),
    'purple': (128,0,128),
    'brown': (205, 127, 50),
    'Transparent': (0,0,0,0),
    }

# Dictionary whith all the symbols that can be used for plotting.
# Changed with the Setsymbol function of the plot.
Symbols = {
    'o' : "Default symbol, round circle symbol",
    't' : "Triangle pointing downwards symbol",
    't1' : "Triangle pointing upwards symbol",
    't2' : "Triangle pointing right side symbol",
    't3' : "Triangle pointing left side symbol",
    's' : "Square symbol",
    'p' : "Pentagon symbol",
    'h' : "Hexagon symbol",
    'star' : "Star symbol",
    '+' : "Plus symbol",
    'd' : "Prism symbol",
    'x' : "Cross symbol",
    None : "No marker",
}

class PlotLengthWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, plot_data, idx):
        super().__init__()
        self.plot_data = plot_data
        self.idx = idx
        layout = QVBoxLayout()
        self.label = QLabel(f"Plot length of plot {idx}")
        layout.addWidget(self.label)
        self.QLE = QLineEdit()

        self.QLE.setValidator(QIntValidator(1, 100_000, self))
        self.QLE.setText(str(self.plot_data['L'][idx]))
        self.QLE.textChanged.connect(self.set_plot_length)
        #self.addWidget(self.QLE)
        layout.addWidget(self.QLE)
        self.setLayout(layout)
        
    def set_plot_length(self, plot_length):
        try:
            self.plot_data['L'][self.idx] = int(plot_length)
        except:
            print("Could not set plot length")     

class PlotSubsamplehWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, plot_data, idx):
        super().__init__()
        self.plot_data = plot_data
        self.idx = idx
        layout = QVBoxLayout()
        self.label = QLabel(f"Subsampling of plot {idx}")
        layout.addWidget(self.label)
        self.QLE = QLineEdit()

        self.QLE.setValidator(QIntValidator(1, 10_000, self))
        self.QLE.setText(str(self.plot_data['sub_sample'][idx]))
        self.QLE.textChanged.connect(self.set_subsamples)
        layout.addWidget(self.QLE)
        self.setLayout(layout)
        
    def set_subsamples(self, subsamples):
        try:
            self.plot_data['sub_sample'][self.idx] = int(subsamples)
        except Exception as E:
            if len(self.plot_data['sub_sample']) == 0:
                return
            print("Could not set plotting subsampler")
            print(E)

class PlotAxisWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, graphWidget):
        super().__init__()
        self.graphWidget = graphWidget
        layout = QVBoxLayout()

        # Get default values of the axis
        axX = self.graphWidget.getAxis('bottom')
        lower_limit_X = axX.range[0]
        upper_limit_X = axX.range[1]

        axY = self.graphWidget.getAxis('left')
        lower_limit_Y = axY.range[0]
        upper_limit_Y = axY.range[1]

        # Add x-axis lower limit text and textbox
        self.label_X_low = QLabel("Set the x-axis lower limit")
        layout.addWidget(self.label_X_low)
        self.QLE_X_low = QLineEdit()
        self.QLE_X_low.setValidator(QDoubleValidator(-1000_000, 1000_000, 10))        
        self.QLE_X_low.setText(str(lower_limit_X))
        self.QLE_X_low.textChanged.connect(self.set_x_low)
        layout.addWidget(self.QLE_X_low)

        # Add x-axis lower limit text and textbox
        self.label_X_high = QLabel("Set the x-axis upper limit")
        layout.addWidget(self.label_X_high)
        self.QLE_X_high = QLineEdit()
        self.QLE_X_high.setValidator(QDoubleValidator(-1000_000, 1000_000, 10))
        self.QLE_X_high.setText(str(upper_limit_X))
        self.QLE_X_high.textChanged.connect(self.set_x_high)
        layout.addWidget(self.QLE_X_high)

        # Add x-axis lower limit text and textbox
        self.label_Y_low = QLabel("Set the y-axis lower limit")
        layout.addWidget(self.label_Y_low)
        self.QLE_Y_low = QLineEdit()
        self.QLE_Y_low.setValidator(QDoubleValidator(-1000_000, 1000_000, 10))        
        self.QLE_Y_low.setText(str(lower_limit_Y))
        self.QLE_Y_low.textChanged.connect(self.set_y_low)
        layout.addWidget(self.QLE_Y_low)

        # Add x-axis lower limit text and textbox
        self.label_Y_high = QLabel("Set the y-axis upper limit")
        layout.addWidget(self.label_Y_high)
        self.QLE_Y_high = QLineEdit()
        self.QLE_Y_high.setValidator(QDoubleValidator(-1000_000, 1000_000, 10))        
        self.QLE_Y_high.setText(str(upper_limit_Y))
        self.QLE_Y_high.textChanged.connect(self.set_y_high)
        layout.addWidget(self.QLE_Y_high)


        self.setLayout(layout)
        
    def set_x_low(self, limit):
        axX = self.graphWidget.getAxis('bottom')
        upper_limit = axX.range[1]
        lower_limit = float(limit)
        try:
            lower_limit = float(limit)
        except ValueError:
            # Harmless, probably only due to not having any number printed
            return

        if lower_limit >= upper_limit:
            return

        if (lower_limit-axX.range[0])/(np.abs(lower_limit+axX.range[0])+1e-4)<0.01:
            # Don't want it reacting to too small changes in the values
            return

        self.graphWidget.setXRange(lower_limit, upper_limit, padding=0)

    def set_x_high(self, limit):
        axX = self.graphWidget.getAxis('bottom')
        lower_limit = axX.range[0]
        try:
            upper_limit = float(limit)
        except ValueError:
            # Harmless, probably only due to not having any number printed
            return

        if lower_limit >= upper_limit:
            return

        if (upper_limit-axX.range[1])/(np.abs(upper_limit+axX.range[1])+1e-4)<0.01:
            # Don't want it reacting to too small changes in the values
            return

        self.graphWidget.setXRange(lower_limit, upper_limit, padding=0)

    def set_y_low(self, limit):
        axY = self.graphWidget.getAxis('left')
        upper_limit = axY.range[1]
        try:
            lower_limit = float(limit)
        except ValueError:
            # Harmless, probably only due to not having any number printed
            return
        if lower_limit >= upper_limit:
            return

        if (lower_limit-axY.range[0])/(np.abs(lower_limit+axY.range[0])+1e-4)<0.01:
            # Don't want it reacting to too small changes in the values
            return

        self.graphWidget.setYRange(lower_limit, upper_limit, padding=0)

    def set_y_high(self, limit):
        axY = self.graphWidget.getAxis('left')
        lower_limit = axY.range[0]
        try:
            upper_limit = float(limit)
        except ValueError:
            # Harmless, probably only due to not having any number printed
            return

        if lower_limit >= upper_limit:
            return

        if (upper_limit-axY.range[1])/(np.abs(upper_limit+axY.range[1])+1e-4)<0.01:
            # Don't want it reacting to too small changes in the values
            return

        self.graphWidget.setYRange(lower_limit, upper_limit, padding=0)


class PlotWindow(QMainWindow):
    """
    Interactive real-time plotting window based on PyQt6 and pyqtgraph.

    The window supports multiple simultaneous traces, configurable
    subsampling/averaging, per-trace styles, and manual axis control. Data
    is streamed from a user-provided mapping of channel objects, which are
    queried periodically by a QTimer loop.

    Parameters
    ----------
    c_p : dict
        Shared configuration/state dictionary. Must include at least:
            - 'program_running' : bool
                Used to stop plotting when the program exits.
            - 'single_sample_channels' : list # TODO remove this
                Identifiers of channels that are single-sample rate, used to
                adjust averaging between x and y.
    data : dict[str, object]
        Mapping from channel names to data providers. Each provider must
        expose:
            - `.get_data(n, offset=0)` or `.get_data_spaced(n, step)`
            - `.unit` : str
                Unit string for axis labeling.
            - `.index` : int
                Current data index (used for averaging alignment).
    x_keys : list[str]
        Keys in `data` to be used for x-axes of initial plots.
    y_keys : list[str]
        Keys in `data` to be used for y-axes of initial plots.
    aspect_locked : bool, optional
        If True, lock the x and y aspect ratio. Default is False.
    grid_on : bool, optional
        If True, display grid lines. Default is False.
    title : str, optional
        Window title. Default is "Data plotter".
    default_plot_length : int, optional
        Default number of data points shown per plot. Default is 5000.

    UI Elements
    -----------
    • Central pyqtgraph.PlotWidget with legend.  
    • Toolbar actions:
        - Pause/resume live plotting.
        - Add new plots dynamically.
        - Manual axis adjustment.
        - Toggle aspect ratio lock.
        - Toggle central reference circle.
    • Menu bar:
        - Background color presets.
        - Per-plot submenus for color, symbol, averaging, subsample length,
          axis data source, and deletion.

    Notes
    -----
    - Plot data metadata is stored in `self.plot_data`, containing
      `x`, `y`, `pen`, `L` (length), `sub_sample`, and `averaging`.
    - Each plot is represented by both a line (`self.data_lines`) and
      a point marker (`self.data_point_markers`) showing the latest sample.
    - The update loop runs every 50 ms by default (`QTimer` interval).
    - Averaging is handled via `non_overlapping_average`, which reduces
      data resolution by chunking.

    Methods
    -------
    add_plot(xname, yname)
        Add a new trace to the window.
    delete_plot(plot_idx)
        Remove an existing trace and its metadata.
    create_plot_menus()
        Rebuild the menu bar based on current plots.
    update_plot_data()
        Update all traces from the `data` sources.
    lock_aspect_ratio()
        Toggle aspect ratio locking.
    add_circle()
        Add or toggle a central reference circle.
    set_axis_labels()
        Update axis labels based on the first trace.
    set_plot_color(color, idx)
        Change the line color of a trace.
    set_symbol_color(color, idx)
        Change the marker symbol color of a trace.
    toggle_averaging(plot_idx)
        Toggle averaging on or off for a trace.
    create_plot_length_window(idx)
        Open a dialog to adjust the number of points shown for a trace.
    create_plot_subsampler_window(idx)
        Open a dialog to adjust subsampling/averaging length for a trace.
    open_plot_axis_window()
        Open a dialog to manually adjust axis limits.
    toggle_live_plotting()
        Pause or resume live updates.
    set_x_data(idx, x_key)
        Change the x-axis data source of a trace.
    set_y_data(idx, y_key)
        Change the y-axis data source of a trace.

    Example
    -------
    >>> c_p = {"program_running": True, "single_sample_channels": []}
    >>> data = {
    ...     "T_time": DummyChannel(unit="s"),
    ...     "Signal": DummyChannel(unit="a.u."),
    ... }
    >>> win = PlotWindow(c_p, data, ["T_time"], ["Signal"])
    >>> win.show()
    """


    # NOTE there was a bug in pyqtgraph 0.12.3 which made the window
    # crash when setting back auto zoom on the axis
    
    def __init__(self, c_p, data, x_keys, y_keys, aspect_locked=False, grid_on=False,
                 title="Data plotter", default_plot_length=5_000):
        super().__init__()
        self.c_p = c_p
        self.data = data
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.addLegend()
        self.sub_widgets = []
        self.x = list(range(100))  # 100 time points
        self.y = [randint(0, 100) for _ in range(100)]  # 100 data points
        self.y2 = [randint(0, 100) for _ in range(100)]  # 100 data points
        self.default_plot_length = default_plot_length
        self.default_subsample = 100
        self.color_idx = 0 # to keep track of which color to use next
        self.circle = None
        self.aspect_locked = aspect_locked
        self.graphWidget.setAspectLocked(self.aspect_locked,1)
        self.graphWidget.showGrid(x=grid_on, y=grid_on, alpha=0.5)  
        # Set up plot data

        self.plot_running = True
        self.graphWidget.setBackground('k')
        self.setWindowTitle(title)

        pen = pg.mkPen(color=Colors['red'])
        self.pen2 = pg.mkPen(color=Colors['green'])

        # Initalizing the plot data
        self.plot_data = {
            'x':[x for x in x_keys],
            'y':[y for y in y_keys],
            }

        self.plot_data['L'] = np.ones(len(x_keys), int) * self.default_plot_length
        self.plot_data['sub_sample'] = self.default_subsample * np.ones(len(x_keys), int) # Changed subsampling here 
        self.plot_data['pen'] = [pg.mkPen(color=Colors['red']) for _ in x_keys]
        self.plot_data['averaging'] = [False for _ in x_keys]

        self.data_lines = []
        self.data_lines.append(self.graphWidget.plot(self.x, self.y, pen=pen, name='plot 0'))

        self.data_point_markers = []
        self.data_point_markers.append(self.graphWidget.plot(self.x, self.y, pen=pen, name='plot 0'))

        self.timer = QTimer()
        self.timer.setInterval(50) # sets the fps of the timer
        self.timer.timeout.connect(self.update_plot_data)
        
        self.toolbar = QToolBar("Main tools")

        self.stop_plot = QAction("Stop plotter", self)
        self.stop_plot.setToolTip("Momentarily freezes the plotting window")
        self.stop_plot.triggered.connect(self.toggle_live_plotting)
        self.stop_plot.setCheckable(True)

        self.add_plot_action = QAction("Add plot", self)
        self.add_plot_action.setToolTip("Adds another plot to the window")
        add_p = partial(self.add_plot, 'T_time', 'PSD_A_F_X')
        self.add_plot_action.triggered.connect(add_p)
        self.add_plot_action.setCheckable(False)

        self.add_circle_action = QAction("Add circle", self)
        self.add_circle_action.setToolTip("Adds a circle in the center of the graph")
        self.add_circle_action.triggered.connect(self.add_circle)
        self.add_circle_action.setCheckable(True) # Should be true

        self.plot_axis_action = QAction("Adjust plot axis", self)
        self.plot_axis_action.setToolTip("Manually change the axis limits.")
        self.plot_axis_action.triggered.connect(self.open_plot_axis_window)
        self.plot_axis_action.setCheckable(False)

        self.lock_aspect_action = QAction("Lock aspect ratio", self)
        self.lock_aspect_action.setToolTip("Lock the aspect ratio to have same scale on both axis.")
        self.lock_aspect_action.triggered.connect(self.lock_aspect_ratio)
        self.lock_aspect_action.setCheckable(True)
        self.graphWidget.addLegend() # Does not work :/ 

        self.toolbar.addAction(self.stop_plot)
        self.toolbar.addAction(self.add_plot_action)
        self.toolbar.addAction(self.plot_axis_action)
        self.toolbar.addAction(self.add_circle_action)
        self.toolbar.addAction(self.lock_aspect_action)
        self.setCentralWidget(self.graphWidget)
        self.addToolBar(self.toolbar)
        self.create_plot_menus()
        
        self.set_axis_labels()

        self.delete_plot(0)
        
        self.timer.start()
        for x,y in zip(x_keys, y_keys):
            self.add_plot(x,y)
            print(f"Adding plot {x} vs {y}")

    def add_circle(self):

        if self.circle is None:
            radius=100.0
            center=[0,0]
            self._t = np.linspace(0,2*np.pi, 100)
            self.x_c = (radius * np.cos(self._t)) + center[0]
            self.y_c = (radius * np.sin(self._t)) + center[1]
            pen = pg.mkPen(color=Colors['white'])
            self.circle = self.graphWidget.plot(self.x_c, self.y_c, pen=pen)
            self.circle.setVisible(True)
            return 
        if self.circle.isVisible():
            self.circle.setVisible(False)
            return
        self.circle.setVisible(True)

    def set_axis_labels(self):
        """
        Sets the axis labels of the plots the the values specified by the
        units of the first[0th] plot.

        Returns
        -------
        None.

        """
        
        y_label = f"{self.plot_data['y'][0]} {self.data[self.plot_data['y'][0]].unit}"
        self.graphWidget.setLabel('left', y_label, color='red', fontsize=200)
        x_label = f"{self.plot_data['x'][0]} {self.data[self.plot_data['x'][0]].unit}"
        self.graphWidget.setLabel('bottom', x_label, color='red', fontsize=200)
        self.graphWidget.addLegend()

    def lock_aspect_ratio(self):
        self.aspect_locked = not self.aspect_locked
        self.graphWidget.setAspectLocked(self.aspect_locked,1)


    def add_plot(self, xname='T_time', yname='PSD_A_P_X'):
        # Adds another data line to the plot
        # almost works, problem with it adding extra lines for changing the number of plot points
        
        self.plot_data['x'].append(xname)
        self.plot_data['y'].append(yname)
        self.plot_data['averaging'].append(False)

        tmp = np.ones(len(self.plot_data['x']), int) * self.default_plot_length
        tmp[:-1] = self.plot_data['L'][:]
        self.plot_data['L'] = tmp # Add sub_sample here
        
        tmp = self.default_subsample * np.ones(len(self.plot_data['x']), int)
        tmp[:-1] = self.plot_data['sub_sample'][:]
        self.plot_data['sub_sample'] = tmp # Add sub_sample here
        self.color_idx += 1
        self.color_idx = self.color_idx % len(Colors)
        key = list(Colors)[self.color_idx]
        pen = pg.mkPen(color=Colors[key]) # Default color
        self.plot_data['pen'].append(pen)
        plot_name = f"{xname} vs {yname} - plot {len(self.data_lines)}"
        self.data_lines.append(
            self.graphWidget.plot(self.x, self.y, name=plot_name,pen=pen, autoDownsample=True) ) # Added auto downsample to improve performance
        self.data_point_markers.append(
            self.graphWidget.plot(self.x, self.y, pen=pen, symbol='o',symbolPen=pen,
                                  symbolBrush=Colors[key]))
        self.menu.clear()
        self.create_plot_menus()
        self.set_axis_labels()

    def delete_plot(self, plot_idx=0):

        self.plot_data['x'].pop(plot_idx)
        self.plot_data['y'].pop(plot_idx)
        self.plot_data['pen'].pop(plot_idx)
        self.plot_data['averaging'].pop(plot_idx)

        self.data_lines[plot_idx].setVisible(False)
        self.data_point_markers[plot_idx].setVisible(False)
        self.graphWidget.removeItem(self.data_lines[plot_idx])
        self.graphWidget.removeItem(self.data_point_markers[plot_idx])
        self.data_lines.pop(plot_idx)
        self.data_point_markers.pop(plot_idx)

        self.plot_data['L'] = np.delete(self.plot_data['L'], int(plot_idx))
        self.plot_data['sub_sample'] = np.delete(self.plot_data['sub_sample'], int(plot_idx))
        if len(self.plot_data['x']) > 0:
            self.set_axis_labels()
        self.menu.clear()
        self.create_plot_menus()
        self.color_idx -= 1
        for idx, line in enumerate(self.data_lines):
            self.set_plot_name(idx, None)

    def create_plot_menus(self):
        # Add menu
        self.menu = self.menuBar()
                
        BG_color_submenu = self.menu.addMenu("BG color")
        for col in Colors:
            
            color_command = partial(self.set_bg_color, Colors[col])
            set_col = QAction(col, self)
            set_col.setStatusTip(f"Set BG color to {col}")
            set_col.triggered.connect(color_command)
            BG_color_submenu.addAction(set_col)

        for idx, line in enumerate(self.data_lines):
            Plot_1_menu = self.menu.addMenu(f"Plot {idx}")
            Plot_1_menu.addSeparator()
            # Create submenu for setting colors
            Plot_1_menu.setTitle(f"Plot {idx} settings")
            
            color_submenu = Plot_1_menu.addMenu("Plot color")
            for col in Colors:
                
                color_command = partial(self.set_plot_color, Colors[col], idx)
                set_col = QAction(col, self)
                set_col.setStatusTip(f"Set plot color to {col}")
                set_col.triggered.connect(color_command)
                color_submenu.addAction(set_col)

            symbol_color_submenu = Plot_1_menu.addMenu("Symbol color")
            for col in Colors:

                color_command = partial(self.set_symbol_color, Colors[col], idx)
                set_col = QAction(col, self)
                set_col.setStatusTip(f"Set plot symbol color to {col}")
                set_col.triggered.connect(color_command)
                symbol_color_submenu.addAction(set_col)

            x_data_submenu = Plot_1_menu.addMenu("X-data")
            for data_idx, x in enumerate(self.data):
                data_command = partial(self.set_x_data, idx, x)
                set_data = QAction(x, self)
                set_data.setStatusTip("Set x-data")
                set_data.triggered.connect(data_command)
                x_data_submenu.addAction(set_data)

            y_data_submenu = Plot_1_menu.addMenu("Y-data")
            for data_idx, y in enumerate(self.data):
                data_command = partial(self.set_y_data, idx, y)
                set_data = QAction(y, self)
                set_data.setStatusTip("Set y-data")
                set_data.triggered.connect(data_command)
                y_data_submenu.addAction(set_data)

            plot_symbol_submenu = Plot_1_menu.addMenu("Plot symbol")
            for symbol in Symbols:
                symbol_command = partial(self.set_plot_symbol, idx, symbol)
                set_symbol = QAction(Symbols[symbol], self)
                set_symbol.setStatusTip("Select symbol")
                set_symbol.triggered.connect(symbol_command)
                plot_symbol_submenu.addAction(set_symbol)

            # Add submenu for marker symbols
            marker_symbol_submenu = Plot_1_menu.addMenu("Marker symbol")
            for symbol in Symbols:
                symbol_command = partial(self.set_marker_symbol, idx, symbol)
                set_symbol = QAction(Symbols[symbol], self)
                set_symbol.setStatusTip("Select marker symbol")
                set_symbol.triggered.connect(symbol_command)
                marker_symbol_submenu.addAction(set_symbol)

            # Add option to remove this plot specifically
            remove_plot_action = QAction("Remove plot", self)
            remove_command = partial(self.delete_plot, idx)
            remove_plot_action.setToolTip("Removes a plot from the window")
            remove_plot_action.triggered.connect(remove_command)
            remove_plot_action.setCheckable(False)
            Plot_1_menu.addAction(remove_plot_action)

            change_length_action = QAction("Adjust plot length", self)
            length_command = partial(self.create_plot_length_window, idx)
            change_length_action.setStatusTip("Change the number of points being plotted")
            change_length_action.triggered.connect(length_command )
            change_length_action.setCheckable(False)
            Plot_1_menu.addAction(change_length_action)

            change_subsample_action = QAction("Adjust subsample/averaging length", self)
            subsample_command = partial(self.create_plot_subsampler_window, idx)
            change_subsample_action.setStatusTip("Changes the subsampling frequency or the averaging length")
            change_subsample_action.triggered.connect(subsample_command)
            change_subsample_action.setCheckable(False)
            Plot_1_menu.addAction(change_subsample_action)

            toggle_averaging_action = QAction("Toggle averaging", self)
            toggle_averaging_command = partial(self.toggle_averaging, idx)
            toggle_averaging_action.setStatusTip("Toggles averaging on/off")
            toggle_averaging_action.triggered.connect(toggle_averaging_command)
            toggle_averaging_action.setCheckable(True)
            Plot_1_menu.addAction(toggle_averaging_action)
 
    def set_plot_symbol(self, plot_idx, symbol):
        try:
            self.data_lines[plot_idx].setSymbol(symbol)
        except Exception as E:
            print(E)

    def set_marker_symbol(self, plot_idx,symbol):
        try:
            self.data_point_markers[plot_idx].setSymbol(symbol)
        except Exception as E:
            print(E)

    def toggle_averaging(self, plot_idx):
        self.plot_data['averaging'][plot_idx] = not self.plot_data['averaging'][plot_idx]

    def create_plot_length_window(self, idx):
        self.PlotLengthWindow = PlotLengthWindow(self.plot_data, idx)
        self.PlotLengthWindow.show()
        self.sub_widgets.append(self.PlotLengthWindow)

    def create_plot_subsampler_window(self, idx):
        self.PlotSubsamplehWindow = PlotSubsamplehWindow(self.plot_data, idx)
        self.PlotSubsamplehWindow.show()
        self.sub_widgets.append(self.PlotSubsamplehWindow)

    def open_plot_axis_window(self):
        self.PlotAxisWindow = PlotAxisWindow(self.graphWidget)
        self.PlotAxisWindow.show()
        self.sub_widgets.append(self.PlotAxisWindow)

    def update_plot_data(self):
        # If the program has closed then we should also close this window
        if self.plot_running and self.c_p['program_running'] and self.isVisible():
            for idx in range(len(self.data_lines)):
                x_key = self.plot_data['x'][idx]
                y_key = self.plot_data['y'][idx]
                L = int(self.plot_data['L'][idx])
                S = int(self.plot_data['sub_sample'][idx])

                # Here we check if the data is sampled at the same rate on the two channels.
                # If not we compensate for that by averaging the data differently.
                a = self.data[x_key] in self.c_p['single_sample_channels']
                b = self.data[y_key] in self.c_p['single_sample_channels']
                prefac = [1,1]
                if a and not b:
                    prefac[1] *= 14
                if b and not a:
                    prefac[0] *= 14
                try:
                    if self.plot_data['averaging'][idx]:
                        # TODO check if we can make this smarter so it does not randomly move around
                        offset_x = self.data[x_key].index % S*prefac[0]
                        x_data = self.non_overlapping_average(
                            self.data[x_key].get_data(L*prefac[0], offset_x), S*prefac[0])
                        offset_y = self.data[y_key].index % S*prefac[1]
                        y_data = self.non_overlapping_average(
                            self.data[y_key].get_data(L*prefac[1], offset_y), S*prefac[1])                   
                    else:
                        nbr_elements = int(L/S)
                        x_data = self.data[x_key].get_data_spaced(nbr_elements, S*prefac[0])
                        y_data = self.data[y_key].get_data_spaced(nbr_elements, S*prefac[1])

                    m_l = min(len(x_data), len(y_data))
                    self.data_lines[idx].setData(x_data[0:m_l], y_data[0:m_l])
                    self.data_point_markers[idx].setData([x_data[-1]], [y_data[-1]])

                except Exception as e:
                    print(x_key, y_key,idx,len(self.data_lines), len(self.plot_data['x']),
                          len(self.plot_data['y']))
                    print('Plotting error:', e)


    def non_overlapping_average(self, signal, N):
        # Calculate the number of chunks
        num_chunks = len(signal) // N
        
        # Truncate the signal to fit an integer number of chunks
        truncated_signal = signal[:N * num_chunks]
        
        # Reshape the truncated signal into a 2D array with 'N' columns
        reshaped_signal = np.reshape(truncated_signal, (num_chunks, N))
        
        # Calculate the average along the columns
        return np.mean(reshaped_signal, axis=1)

    def toggle_live_plotting(self):
        self.plot_running = not self.plot_running
        if self.plot_running:
            self.stop_plot.setIconText("Stop plotter")
        else:
            self.stop_plot.setIconText("Start plotter")

    def set_bg_color(self, color):
        self.graphWidget.setBackground(color)

    def set_symbol_color(self, color, idx):
        pen = pg.mkPen(color=color)

        marker = self.data_point_markers[idx]
        marker.setSymbolPen(pen)
        marker.setSymbolBrush(color)

        line = self.data_lines[idx]
        line.setSymbolPen(pen)
        line.setSymbolBrush(color) # Can be used to set the symbol color

    def set_plot_color(self, color, idx):
        pen = pg.mkPen(color=color)
        line = self.data_lines[idx]
        line.setPen(pen)

        if idx == len(self.plot_data['pen']):
            self.plot_data['pen'].append(pen)
            return
        self.plot_data['pen'][idx] = pen

    def set_plot_name(self, idx, name):
        # Not tested yet
        if name is not None:
            self.data_lines[idx].name = name
            return
        plot_name = f"Plot {idx}: {self.plot_data['x'][idx]} vs {self.plot_data['y'][idx]}"
        self.data_lines[idx].name = plot_name
        
    def set_x_data(self, idx, x_data):
        self.plot_data['x'][idx] = x_data
        plot_name = f" {x_data} vs {self.plot_data['y'][idx]}"
        self.graphWidget.removeItem(self.data_lines[idx])
        self.data_lines[idx] = self.graphWidget.plot(self.x, self.y,name=plot_name,
                                                     pen=self.plot_data['pen'][idx])
        
        self.set_axis_labels()
        
    def set_y_data(self, idx, y_data):
        self.plot_data['y'][idx] = y_data
        plot_name = f" {self.plot_data['x'][idx]} vs {y_data}"
        self.graphWidget.removeItem(self.data_lines[idx])
        self.data_lines[idx] = self.graphWidget.plot(self.x, self.y, name=plot_name,
                                                     pen=self.plot_data['pen'][idx])
        self.set_axis_labels()

    def __del__(self):
        self.timer.stop()

        for widget in self.sub_widgets:
            widget.close()

    def close(self):
        for widget in self.sub_widgets:
            widget.close()
        self.timer.stop()
        super().close()
        
