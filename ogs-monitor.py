import os
from PIL import Image
#import bokeh
#import shutil
import sys
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#from collections import defaultdict, namedtuple
from pathlib import Path
from queue import Queue, Empty
#from time import sleep
#from dateutil import parser
#import tempfile
from watchdog.observers import Observer, ObserverType
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import layout, column, row

from ogstools.logparser.log_file_handler import (
    LogFileHandler,
    normalize_regex,
    parse_line,
    select_regex,
)
from ogstools.logparser.regexes import (
    Context,
    StepStatus,
    Termination,
    TimeStepEnd,
    TimeStepStart,
    AssemblyTime,
    LinearSolverTime,
    IterationEnd,
    IterationStart,
    TimeStepConvergenceCriterion,
    ComponentConvergenceCriterion,
    SimulationExecutionTime
)

from bokeh.models import ColumnDataSource, Select, Range1d, HoverTool

if os.path.exists(sys.argv[1]):
    file_path = sys.argv[1]
else:
    print("File path does not exist. Please provide a valid path.")
    sys.exit(1)
print(f"Monitoring log file: {file_path}")

block = True
data_source = ColumnDataSource(data = {"time_step": [], "step_size": [], "assembly_time": [], "linear_solver_time": [], "step_start_time": [], "iteration_number": []})
data_source_iter = ColumnDataSource(data = {"iteration_number": [], "vspan": [], "line_width": [], "dx_x": [], "dx_x_0": [], "dx_x_1": [],"dx_x_2": [], "dx_x_3": [], "dx_x_4": [],"dx_x_5": []})

original_file = Path(file_path)

records: Queue = Queue()
observer: ObserverType = Observer()
status: Context = Context()

handler = LogFileHandler(
    original_file,
    queue=records,
    status=status,
    stop_callback=lambda: (print("Stop Observer"), observer.stop()),
)
observer.schedule(handler, path=str(original_file.parent), recursive=False)

print("Starting observer...")

observer.start()


logo = "ogs_textlogo.png"

im = Image.open(logo)

imarray = np.array(im.convert("RGBA"))
height, width = imarray.shape[:2]
rgba = imarray.view(dtype=np.uint32).reshape((height, width))
rgba = np.flipud(rgba)
page_logo = figure(width = width, height = height, x_range=(0,1), y_range=(0,1), title="", toolbar_location=None)
page_logo.axis.visible = False
page_logo.grid.visible = False

page_logo.image_rgba(image=[rgba], x=0,y=0, dw=1, dh=1)


fig_1 = figure(  width=500, height=450,
             title="Number of Iterations per Time Step"
            )

fig_1.line(x="time_step", y="iteration_number", line_color="blue", line_width=3.0, source=data_source)

fig_1.xaxis.axis_label = "time_step (s)"
fig_1.yaxis.axis_label = "number of iterations"
hover = HoverTool(
    tooltips=[
        ("Time Step", "@time_step"),
        ("Iteration Number", "@iteration_number")
    ]
)
fig_1.add_tools(hover)

fig_2= figure(  width=500, height=450,
             title="Relative Convergence", y_axis_type="log"
            )

fig_2.line(x="iteration_number", y="dx_x_0", line_color="blue", line_width=3.0, source=data_source_iter)
fig_2.vspan(x="vspan", line_width="line_width", line_color="tomato", source=data_source_iter)

fig_2.xaxis.axis_label = "number of iterations"
fig_2.yaxis.axis_label = "relative convergence"
hover = HoverTool(
    tooltips=[
        ("Iteration Number", "@iteration_number"),
        ("dx_x_0", "@dx_x_0")
    ]
)
fig_2.add_tools(hover)
dropdown_lin = Select(title="Linear Scale Plots:", value="Iteration Number", options=["Iteration Number", "Assembly Time", "Linear Solver Time"])

dropdown_log = Select(title="Logarithmic Scale Plots", value="dx_x_0", options=["Step Size", "Timestep Start Time", "dx_x", "dx_x_0", "dx_x_1", "dx_x_2", "dx_x_3", "dx_x_4", "dx_x_5"])

def update_figure():
    global records, observer, data_source, data_source_iter, block
    try:
        item = records.get(block=block)
        if isinstance(item, Termination):
            print(f"Consumer: Termination signal ({item}) received. Exiting.")
            block = False
            observer.stop()
            observer.join()
        elif isinstance(item, SimulationExecutionTime):
            print(f"Simulation execution time: {item.execution_time}")
            block = False
            observer.stop()
            observer.join()
        elif isinstance(item, TimeStepStart):
            print(f"Timestep: {item.time_step}, Step size: {item.step_size}")
            new_row = {"step_size": [item.step_size,], "time_step": [item.time_step,], "assembly_time": [0], "linear_solver_time": [0], "step_start_time": [item.step_start_time], "iteration_number": [0]}
            data_source.stream(new_row)
        elif isinstance(item, AssemblyTime):
            index = len(data_source.data["assembly_time"])-1
            new_time = data_source.data["assembly_time"][index] + item.assembly_time
            data_source.patch({"assembly_time": [(index, new_time)]})
        elif isinstance(item, LinearSolverTime):
            index = len(data_source.data["linear_solver_time"])-1
            new_time = data_source.data["linear_solver_time"][index] + item.linear_solver_time
            data_source.patch({"linear_solver_time": [(index, new_time)]})
        elif isinstance(item, IterationEnd):
            index = len(data_source.data["iteration_number"])-1
            iteration = item.iteration_number
            data_source.patch({"iteration_number": [(index, iteration)]})
        elif isinstance(item, IterationStart):
            iteration_offset = len(data_source_iter.data["iteration_number"])
            line_width = 0
            if item.iteration_number == 1:
                line_width = 1
            new_row = {"iteration_number": [iteration_offset+1], "vspan": [iteration_offset+0.75], "line_width": [line_width], "dx_x": [1], "dx_x_0": [1], "dx_x_1": [1],"dx_x_2": [1], "dx_x_3": [1], "dx_x_4": [1],"dx_x_5": [1]}
            data_source_iter.stream(new_row)
        elif isinstance(item, TimeStepConvergenceCriterion):
            index = len(data_source_iter.data["iteration_number"])-1
            data_source_iter.patch({f"dx_x": [(index, item.dx_x)]})
        elif isinstance(item, ComponentConvergenceCriterion):
            index = len(data_source_iter.data["iteration_number"])-1
            data_source_iter.patch({f"dx_x_{item.component}": [(index, item.dx_x)]})
    except Empty:
        pass

def newplot_lin(attr, old, new):
    global fig_1, data_source
    fig_1.renderers = []
    print(f"Changing plot type from {old} to {new}")
    if new == "Iteration Number":
        fig_1.line(x="time_step", y="iteration_number", line_color="blue", line_width=3.0, source=data_source)
        fig_1.xaxis.axis_label = "time_step (s)"
        fig_1.yaxis.axis_label = "number of iterations"
        fig_1.title.text = "Number of Iterations per Time Step"
        hover = HoverTool(
            tooltips=[
                ("Time Step", "@time_step"),
                ("Iteration Number", "@iteration_number")
                ])
        fig_1.add_tools(hover)
    elif new == "Assembly Time":
        fig_1.line(x="time_step", y="assembly_time", line_color="blue", line_width=3.0, source=data_source)
        fig_1.xaxis.axis_label = "time_step (s)"
        fig_1.yaxis.axis_label = "assembly time (s)"
        fig_1.title.text = "Assembly Time per Time Step"
        hover = HoverTool(
            tooltips=[
                ("Time Step", "@time_step"),
                ("Assembly Time", "@assembly_time")
                ])
        fig_1.add_tools(hover)
    elif new == "Linear Solver Time":
        fig_1.line(x="time_step", y="linear_solver_time", line_color="blue", line_width=3.0, source=data_source)
        fig_1.xaxis.axis_label = "time_step (s)"
        fig_1.yaxis.axis_label = "linear solver time (s)"
        fig_1.title.text = "Linear Solver Time per Time Step"
        hover = HoverTool(
            tooltips=[
                ("Time Step", "@time_step"),
                ("Linear Solver Time", "@linear_solver_time")
                ])
        fig_1.add_tools(hover)
    

def newplot_log(attr, old, new):
    global fig_2, data_source_iter
    fig_2.renderers = []
    print(f"Changing convergence criterion from {old} to {new}")
    if new == "Step Size":
        fig_2.line(x="time_step", y="step_size", line_color="blue", line_width=3.0, source=data_source)
        fig_2.xaxis.axis_label = "time_step (s)"
        fig_2.yaxis.axis_label = "step size (s)"
        fig_2.title.text = "Step Size per Time Step"
        hover = HoverTool(
            tooltips=[
                ("Time Step", "@time_step"),
                ("Step Size", "@step_size")
                ])
        fig_2.add_tools(hover)
    elif new == "Timestep Start Time":
        fig_2.line(x="time_step", y="step_start_time", line_color="blue", line_width=3.0, source=data_source)
        fig_2.xaxis.axis_label = "time_step (s)"
        fig_2.yaxis.axis_label = "step start time (s)"
        fig_2.title.text = "Timestep Start Time per Time Step"
        hover = HoverTool(
            tooltips=[
                ("Time Step", "@time_step"),
                ("Timestep Start Time", "@step_start_time")
                ])
        fig_2.add_tools(hover)
    elif new == "dx_x":
        fig_2.line(x="iteration_number", y="dx_x", line_color="blue", line_width=3.0, source=data_source_iter)
        fig_2.vspan(x="vspan", line_width="line_width", line_color="tomato", source=data_source_iter)
        fig_2.xaxis.axis_label = "number of iterations"
        fig_2.yaxis.axis_label = "dx_x"
        fig_2.title.text = "Relative Convergence dx_x"
        hover = HoverTool(
            tooltips=[
                ("Number of Iterations", "@iteration_number"),
                ("Relative Convergence dx_x", "@dx_x")
                ])
        fig_2.add_tools(hover)
    elif new == "dx_x_0":
        fig_2.line(x="iteration_number", y="dx_x_0", line_color="blue", line_width=3.0, source=data_source_iter)
        fig_2.vspan(x="vspan", line_width="line_width", line_color="tomato", source=data_source_iter)
        fig_2.xaxis.axis_label = "number of iterations"
        fig_2.yaxis.axis_label = "dx_x_0"
        fig_2.title.text = "Relative Convergence dx_x_0"
        hover = HoverTool(
            tooltips=[
                ("Number of Iterations", "@iteration_number"),
                ("Relative Convergence dx_x_0", "@dx_x_0")
                ])
        fig_2.add_tools(hover)
    elif new == "dx_x_1":
        fig_2.line(x="iteration_number", y="dx_x_1", line_color="blue", line_width=3.0, source=data_source_iter)
        fig_2.vspan(x="vspan", line_width="line_width", line_color="tomato", source=data_source_iter)
        fig_2.xaxis.axis_label = "number of iterations"
        fig_2.yaxis.axis_label = "dx_x_1"
        fig_2.title.text = "Relative Convergence dx_x_1"
        hover = HoverTool(
            tooltips=[
                ("Number of Iterations", "@iteration_number"),
                ("Relative Convergence dx_x_1", "@dx_x_1")
                ])
        fig_2.add_tools(hover)
    elif new == "dx_x_2":
        fig_2.line(x="iteration_number", y="dx_x_2", line_color="blue", line_width=3.0, source=data_source_iter)
        fig_2.vspan(x="vspan", line_width="line_width", line_color="tomato", source=data_source_iter)
        fig_2.xaxis.axis_label = "number of iterations"
        fig_2.yaxis.axis_label = "dx_x_2"
        fig_2.title.text = "Relative Convergence dx_x_2"
        hover = HoverTool(
            tooltips=[
                ("Number of Iterations", "@iteration_number"),
                ("Relative Convergence dx_x_2", "@dx_x_2")
                ])
        fig_2.add_tools(hover)
    elif new == "dx_x_3":
        fig_2.line(x="iteration_number", y="dx_x_3", line_color="blue", line_width=3.0, source=data_source_iter)
        fig_2.vspan(x="vspan", line_width="line_width", line_color="tomato", source=data_source_iter)
        fig_2.xaxis.axis_label = "number of iterations"
        fig_2.yaxis.axis_label = "dx_x_3"
        fig_2.title.text = "Relative Convergence dx_x_3"
        hover = HoverTool(
            tooltips=[
                ("Number of Iterations", "@iteration_number"),
                ("Relative Convergence dx_x_3", "@dx_x_3")
                ])
        fig_2.add_tools(hover)
    elif new == "dx_x_4":
        fig_2.line(x="iteration_number", y="dx_x_4", line_color="blue", line_width=3.0, source=data_source_iter)
        fig_2.vspan(x="vspan", line_width="line_width", line_color="tomato", source=data_source_iter)
        fig_2.xaxis.axis_label = "number of iterations"
        fig_2.yaxis.axis_label = "dx_x_4"
        fig_2.title.text = "Relative Convergence dx_x_4"
        hover = HoverTool(
            tooltips=[
                ("Number of Iterations", "@iteration_number"),
                ("Relative Convergence dx_x_4", "@dx_x_4")
                ])
        fig_2.add_tools(hover)
    elif new == "dx_x_5":
        fig_2.line(x="iteration_number", y="dx_x_5", line_color="blue", line_width=3.0, source=data_source_iter)
        fig_2.vspan(x="vspan", line_width="line_width", line_color="tomato", source=data_source_iter)
        fig_2.xaxis.axis_label = "number of iterations"
        fig_2.yaxis.axis_label = "dx_x_5"
        fig_2.title.text = "Relative Convergence dx_x_5"
        hover = HoverTool(
            tooltips=[
                ("Number of Iterations", "@iteration_number"),
                ("Relative Convergence dx_x_5", "@dx_x_5")
                ])
        fig_2.add_tools(hover)

curdoc().add_periodic_callback(update_figure, 1)

dropdown_lin.on_change("value", newplot_lin)
dropdown_log.on_change("value", newplot_log)
layout = column(page_logo, row(column(dropdown_lin, fig_1), column(dropdown_log, fig_2)))

curdoc().add_root(layout)
curdoc().title = "OGS Log Monitor"

