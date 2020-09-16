import json
import numpy as np
import tkinter as tk
from os import path
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler

root = tk.Tk()
root.wm_title("Embedding in Tk")
file = open('perceptron/test.json', 'w')
file.write('[]')
file.close()


class Layout(object):
    def __init__(self, root):
        fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set(xlim=(0, 1), ylim=(0, 1))  # arbitrary data

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    # Left click -> Class with 0s
    # Right click -> Class with 1s
    # @staticmethod
    def on_click(self, event):
        ix, iy = event.xdata, event.ydata
        point = {'coord': None, 'expected': None}
        color = 'orange'
        point['expected'] = 0
        if(event.button == 3):
            point['expected'] = 1
            color = 'purple'
        point['coord'] = [ix, iy]
        self.ax.scatter(point['coord'][0], point['coord'][1], color=color)
        self.canvas.draw()
        # Open the file, then read it to append new points in active session
        with open('perceptron/test.json', 'r+') as file:
            data = json.load(file)
            data.append(point)
            file.seek(0)
            json.dump(data, file)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


if __name__ == "__main__":
    layout = Layout(root)
    button = tk.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tk.BOTTOM)
    tk.mainloop()
