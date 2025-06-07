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
    SimulationExecutionTime,
    SimulationStartTime
)

from bokeh.models import ColumnDataSource, Select, Range1d, HoverTool, Button

from bokeh.models.widgets import Slider

if os.path.exists(sys.argv[1]):
    file_path = Path(sys.argv[1])
else:
    print("File path does not exist. Please provide a valid path.")
    sys.exit(1)
print(f"Monitoring log file: {file_path}")
if len(sys.argv) > 2:
    print("Set update interval to:", sys.argv[2])
    update_interval = int(sys.argv[2])
else:
    print("No update interval provided, using default of 10 ms.")
    update_interval = 10

data_source = ColumnDataSource(data = {"time_step": [], "step_size": [], "assembly_time": [], "linear_solver_time": [], "step_start_time": [], "iteration_number": []})
data_source_iter = ColumnDataSource(data = {"iteration_number": [], "vspan": [], "line_width": [], "dx_x": [], "dx_x_0": [], "dx_x_1": [],"dx_x_2": [], "dx_x_3": [], "dx_x_4": [],"dx_x_5": []})

def start_observer():
    records: Queue = Queue()
    observer: ObserverType = Observer()
    status: Context = Context()
    handler = LogFileHandler(
            file_path,
            queue=records,
            status=status,
            stop_callback=lambda: (print("Stop Observer"), observer.stop()),
        )
    observer.schedule(handler, path=str(file_path.parent), recursive=False)

    print("Starting observer...")

    observer.start()
    block = True
    return observer, records, handler, status, block

def stop_observer():
    global observer, block
    if observer.is_alive():
        print("Stopping observer...")
        observer.stop()
        observer.join()
        block = False
        print("Observer stopped.")
    else:
        print("Observer is not running.")

observer, records, handler, status, block = start_observer()

os.utime(file_path, None)  # Update the file's last modified time to ensure it is read from the beginning

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

button_start = Button(label = "Start Observer", button_type = "success")
button_stop = Button(label = "Stop Observer", button_type = "danger")
range_ts = 0
range_iter = 100

slider_ts = Slider(start = 0, end = 100,
                value = range_ts, step = 10,
                title = "Time Step Window Size")
slider_iterations = Slider(start = 0, end = 100,
                value = range_iter, step = 10,
                title = "Iteration Window Size")

dropdown_lin = Select(title="Linear Scale Plots:", value="Iteration Number", options=["Iteration Number", "Assembly Time", "Linear Solver Time"])

dropdown_log = Select(title="Logarithmic Scale Plots", value="dx_x_0", options=["Step Size", "Timestep Start Time", "dx_x", "dx_x_0", "dx_x_1", "dx_x_2", "dx_x_3", "dx_x_4", "dx_x_5"])

def update_figure():
    global data_source, data_source_iter, records, block, fig_1, fig_2
    try:
        item = records.get(block=block)
        if isinstance(item, SimulationStartTime):
            print(f"Simulation start time: {item.start_time}")
            block = True
            data_source.data = {"time_step": [], "step_size": [], "assembly_time": [], "linear_solver_time": [], "step_start_time": [], "iteration_number": []}
            data_source_iter.data = {"iteration_number": [], "vspan": [], "line_width": [], "dx_x": [], "dx_x_0": [], "dx_x_1": [],"dx_x_2": [], "dx_x_3": [], "dx_x_4": [],"dx_x_5": []}

        elif isinstance(item, Termination):
            print(f"Consumer: Termination signal ({item}) received. Exiting.")
            block = False
            stop_observer()
        elif isinstance(item, SimulationExecutionTime):
            print(f"Simulation execution time: {item.execution_time}")
            block = False
            stop_observer()
        elif isinstance(item, TimeStepStart):
            print(f"Timestep: {item.time_step}, Step size: {item.step_size}")           
            if range_ts == 0:
                new_row = {"step_size": [item.step_size,], "time_step": [item.time_step,], "assembly_time": [0], "linear_solver_time": [0], "step_start_time": [item.step_start_time], "iteration_number": [0]}
                data_source.stream(new_row)
            else:
                step_size = data_source.data["step_size"] + [item.step_size]
                time_step = data_source.data["time_step"] + [item.time_step]
                assembly_time = data_source.data["assembly_time"] + [0]
                linear_solver_time = data_source.data["linear_solver_time"] + [0]
                step_start_time = data_source.data["step_start_time"] + [item.step_start_time]
                iteration_number = data_source.data["iteration_number"] + [0]
                step_size = step_size[-range_ts:] if len(step_size) > range_ts else step_size
                time_step = time_step[-range_ts:] if len(time_step) > range_ts else time_step
                assembly_time = assembly_time[-range_ts:] if len(assembly_time) > range_ts else assembly_time
                linear_solver_time = linear_solver_time[-range_ts:] if len(linear_solver_time) > range_ts else linear_solver_time
                step_start_time = step_start_time[-range_ts:] if len(step_start_time) > range_ts else step_start_time
                iteration_number = iteration_number[-range_ts:] if len(iteration_number) > range_ts else iteration_number
                data_source.data = {"step_size": step_size, "time_step": time_step, "assembly_time": assembly_time, "linear_solver_time": linear_solver_time, "step_start_time": step_start_time, "iteration_number": iteration_number}

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
            iteration_offset = data_source_iter.data["iteration_number"][-1] if data_source_iter.data["iteration_number"] else 0
            line_width = 0
            if item.iteration_number == 1:
                line_width = 1
            if range_iter == 0:
                new_row = {"iteration_number": [iteration_offset+1], "vspan": [iteration_offset+0.75], "line_width": [line_width], "dx_x": [1], "dx_x_0": [1], "dx_x_1": [1],"dx_x_2": [1], "dx_x_3": [1], "dx_x_4": [1],"dx_x_5": [1]}
                data_source_iter.stream(new_row)
            else:
                iteration_number = data_source_iter.data["iteration_number"] + [iteration_offset+1]
                vspan = data_source_iter.data["vspan"] + [iteration_offset+0.75]
                line_width = data_source_iter.data["line_width"] + [line_width]
                dx_x = data_source_iter.data["dx_x"] + [1]
                dx_x_0 = data_source_iter.data["dx_x_0"] + [1]
                dx_x_1 = data_source_iter.data["dx_x_1"] + [1]
                dx_x_2 = data_source_iter.data["dx_x_2"] + [1]
                dx_x_3 = data_source_iter.data["dx_x_3"] + [1]
                dx_x_4 = data_source_iter.data["dx_x_4"] + [1]
                dx_x_5 = data_source_iter.data["dx_x_5"] + [1]
                iteration_number = iteration_number[-range_iter:] if len(iteration_number) > range_iter else iteration_number
                vspan = vspan[-range_iter:] if len(vspan) > range_iter else vspan
                line_width = line_width[-range_iter:] if len(line_width) > range_iter else line_width
                dx_x = dx_x[-range_iter:] if len(dx_x) > range_iter else dx_x
                dx_x_0 = dx_x_0[-range_iter:] if len(dx_x_0) > range_iter else dx_x_0
                dx_x_1 = dx_x_1[-range_iter:] if len(dx_x_1) > range_iter else dx_x_1
                dx_x_2 = dx_x_2[-range_iter:] if len(dx_x_2) > range_iter else dx_x_2
                dx_x_3 = dx_x_3[-range_iter:] if len(dx_x_3) > range_iter else dx_x_3
                dx_x_4 = dx_x_4[-range_iter:] if len(dx_x_4) > range_iter else dx_x_4
                dx_x_5 = dx_x_5[-range_iter:] if len(dx_x_5) > range_iter else dx_x_5
                data_source_iter.data = {"iteration_number": iteration_number, "vspan": vspan, "line_width": line_width, "dx_x": dx_x, "dx_x_0": dx_x_0, "dx_x_1": dx_x_1,"dx_x_2": dx_x_2, "dx_x_3": dx_x_3, "dx_x_4": dx_x_4,"dx_x_5": dx_x_5}
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
        #range_end = data_source.data["time_step"][-1]
        #range_start = range_end - 10 if range_end > 10 else 0
        #print(f"Setting x_range to {range_start} - {range_end}")
        #fig_1.x_range = Range1d(start=range_start, end=range_end)
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

curdoc().add_periodic_callback(update_figure, update_interval)


def start_observer_wrapper():
    global observer, records, handler, status, block
    if observer.is_alive():
       observer, records, handler, status, block = start_observer()
    else:
        print("Observer is already running.")
#    if observer.is_alive():
#        print("Stopping observer...")
#        block = False
#        observer.stop()
        #observer.join()
#        button.label = "Observer Stopped."
#        button.button_type = "success"
#    else:
#        print("Starting observer...")
#        block = True
#        os.utime(original_file, None)  # Update the file's last modified time to ensure it is read from the beginning
        #observer.schedule(handler, path=str(original_file.parent), recursive=False)
#        observer.start()
#        button.label = "Observer running."
#        button.button_type = "danger"
button_start.on_click(start_observer)
button_stop.on_click(stop_observer)
def update_slider_ts(attr, old, new):
    global range_ts, slider_ts
    range_ts = new
    slider_ts.value = new
def update_slider_iterations(attr, old, new):
    global range_iter, slider_iterations
    range_iter = new
    slider_iterations.value = new
slider_ts.on_change("value", update_slider_ts)
slider_iterations.on_change("value", update_slider_iterations)
dropdown_lin.on_change("value", newplot_lin)
dropdown_log.on_change("value", newplot_log)
layout = column(page_logo, row(button_start, button_stop), row(slider_ts, slider_iterations), row(column(dropdown_lin, fig_1), column(dropdown_log, fig_2)))

curdoc().add_root(layout)
curdoc().title = "OGS Log Monitor"

