import bokeh
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from pathlib import Path
from queue import Queue
from time import sleep
from dateutil import parser
import tempfile
from watchdog.observers import Observer, ObserverType
from bokeh.plotting import figure
from bokeh.io import show, push_notebook, curdoc
from bokeh.layouts import layout

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
    ComponentConvergenceCriterion
)

from bokeh.models import ColumnDataSource

data_source = ColumnDataSource(data = {"time_step": [], "step_size": [], "assembly_time": [], "linear_solver_time": [], "step_start_time": [], "iteration_number": []})
data_source_iter = ColumnDataSource(data = {"iteration_number": [], "vspan": [], "line_width": [], "dx_x": [], "dx_x_0": [], "dx_x_1": [],"dx_x_2": [], "dx_x_3": [], "dx_x_4": [],"dx_x_5": []})

original_file = Path("out_test_transition-sand.log")

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

fig_1 = figure(  width=500, height=450,
             tooltips=[("step_size", "@step_size"),],
             title="Step Size", y_axis_type="log"
            )

fig_1.line(x="time_step", y="step_size", line_color="tomato", line_width=3.0, source=data_source)

fig_1.xaxis.axis_label = "time_step (s)"
fig_1.yaxis.axis_label = "step size (s)"
print("Figure created.")

def update_figure():
    global records, observer, data_source, data_source_iter
    print("Updating figure...")
    item = records.get()
    if item is None:
        print("No new item in queue, skipping update.")
    else:
        print(item)
    if isinstance(item, Termination):
        print(f"Consumer: Termination signal ({item}) received. Exiting.")
        observer.stop()
        return
    elif isinstance(item, TimeStepStart):
        print(f"Timestep: {item.time_step}, Step size: {item.step_size}")
        new_row = {"step_size": [item.step_size,], "time_step": [item.time_step,], "assembly_time": [0], "linear_solver_time": [0], "step_start_time": [item.step_start_time], "iteration_number": [0]}
        data_source.stream(new_row)
        #push_notebook(handle=handle_line_chart)
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
        #print(data_source_iter.data["iteration_number"][-1], len(data_source_iter.data["iteration_number"]))
        iteration_offset = len(data_source_iter.data["iteration_number"])
        line_width = 0
        if item.iteration_number == 1:
            line_width = 1
        new_row = {"iteration_number": [iteration_offset+1], "vspan": [iteration_offset+0.75], "line_width": [line_width], "dx_x": [1], "dx_x_0": [1], "dx_x_1": [1],"dx_x_2": [1], "dx_x_3": [1], "dx_x_4": [1],"dx_x_5": [1]}
        data_source_iter.stream(new_row)
        #push_notebook(handle=handle_line_chart)
    elif isinstance(item, TimeStepConvergenceCriterion):
        index = len(data_source_iter.data["iteration_number"])-1
        data_source_iter.patch({f"dx_x": [(index, item.dx_x)]})
    elif isinstance(item, ComponentConvergenceCriterion):
        index = len(data_source_iter.data["iteration_number"])-1
        data_source_iter.patch({f"dx_x_{item.component}": [(index, item.dx_x)]})

print("Adding periodic callback to update figure...")
curdoc().add_periodic_callback(update_figure, 1000)

print("Adding observer to document...")
curdoc().add_root(layout([[fig_1]]))

observer.join()