import json
import weakref
import numpy as np
from os import path
from perceptron import Perceptron
import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler

root = tk.Tk()
root.wm_title("HW1 - Perceptron")
file = open('bulk_data.json', 'w')   # Everytime the program is executed,
# the a new file is created, replacing the previous one
# (to erase all previous data)
file.write('[]')  # Writes a json list element
file.close()
global quit_button
global train_button
global weights_button
global weights


class Layout(object):
    def __init__(self, root, title, size=5):
        fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title)

        # Makes the plot fixed (prevents from resizing)
        self.ax.set(xlim=(-size, size), ylim=(-size, size))
        # Adds guide lines
        self.ax.axhline(y=0, color="black")
        self.ax.axvline(x=0, color="black")
        # Draw arrow points
        self.ax.scatter(0, size-.1, marker='^', color='black')
        self.ax.scatter(size-.1, 0, marker='>', color='black')
        self.ax.scatter(0, -size+.1, marker='v', color='black')
        self.ax.scatter(-size+.1, 0, marker='<', color='black')

        # Connect the plot with the GUI interface
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        toolbar = NavigationToolbar2Tk(self.canvas, root)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Left click -> Class with 0s
    # Right click -> Class with 1s
    def on_click(self, event):
        ix, iy = event.xdata, event.ydata
        if(ix != None):
            point = {'coord': None, 'expected': None}
            color = 'orange'
            point['expected'] = 0
            if(event.button == 3):
                point['expected'] = 1
                color = 'purple'
            # The round operation on the coords is to prevent a slow convergence of the algorithm,
            # the plot detects a very precise coord of almost 10 decimal places and it is harder for
            # the algorithm to process those values
            point['coord'] = [-1, round(ix, 2), round(iy, 2)]
            self.ax.scatter(point['coord'][1], point['coord'][2], color=color)
            self.canvas.draw()  # Refreshes the canvas
            # Open the file, then read it to append new points in active session
            with open('bulk_data.json', 'r+') as file:
                data = json.load(file)
                data.append(point)
                file.seek(0)
                json.dump(data, file)


class ErrorLayout(object):
    def __init__(self, root, title, x_limit=5, y_limit=5):
        fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title)
        # Makes the plot fixed (prevents from resizing)
        self.ax.set(xlim=(0, x_limit), ylim=(0, y_limit))
        self.ax = fig.gca()
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Connect the plot with the GUI interface
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(self.canvas, root)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def window_error(epoch_amount, error_freq):
    window = tk.Toplevel(root)
    window.geometry("700x800+900+100")
    error_view = ErrorLayout(
        window, 'Error', x_limit=epoch_amount, y_limit=max(error_freq))
    x = np.arange(epoch_amount)
    error_view.ax.bar(x, height=error_freq, align='edge',
                      width=0.6, color='orange')
    error_view.canvas.draw()


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


def _initialize_weights(w0_field, w1_field, w2_field):
    train_button.config(state='normal')
    weights_button.config(state='disabled')
    try:
        weights = [float(w0_field.get()), float(
            w1_field.get()), float(w2_field.get())]

        x = np.linspace(-5, 5, 100)
        if(weights[2] == 0):
            raise ValueError
        y = (weights[0] - (weights[1]*x))/weights[2]
        layout.ax.plot(x, y, color='red')
        layout.canvas.draw()
    except ValueError as error:
        train_button.config(state='disabled')
        weights_button.config(state='normal')
        messagebox.showerror(
            error, 'Weight values must be different from zero!')


# Starts the perceptron algorithm
def _train(eta_field, epoch_field):
    try:
        eta = float(eta_field.get())
        epoch_limit = int(epoch_field.get())

        with open('bulk_data.json', 'r') as json_file:
            data = json.load(json_file)

        args = {'bulk_data': data, 'weights': weights}
        # If eta and epoch_limit are equal to zero, then, the Perceptron must take default values
        # to do that, the args in initialization must be null
        if(eta != 0):
            args['eta'] = eta
        if(epoch_limit != 0):
            args['epoch_limit'] = epoch_limit

        try:
            trainer = Perceptron(**args)
            trainer.process()
        except AttributeError as error:
            messagebox.showerror(error, 'Provided data not found!')

        l = layout.ax.lines.pop(2)
        wl = weakref.ref(l)
        # l.remove()
        del l
        # TODO: Add animation to line
        x = np.linspace(-5, 5, 100)
        for line in trainer.lines:
            y = (line[0] - (line[1]*x))/line[2]
            lines = layout.ax.plot(x, y, color='red')
            l = lines.pop()
            wl = weakref.ref(l)
            layout.canvas.draw()
            l.remove()
            del l

        window_error(trainer.current_epoch, trainer.error_freq)
    except ValueError as error:
        messagebox.showerror(
            error, 'Input values must be float (for eta) and integer (for epoch limit)!')


# TODO: Add weight vector initializer button and the linear function created by those weights
if __name__ == "__main__":
    layout = Layout(root, 'Perceptron', 5)

    # Creates the eta label and field
    eta_label = tk.Label(root, text='η value:', width=10, anchor=tk.S)
    eta_field = tk.Spinbox(master=root, from_=0, to=5, increment=.1, width=5)
    eta_field.place(x=85, y=660)
    eta_label.place(x=1, y=660)

    # Creates the epoch limit label and field
    epoch_label = tk.Label(root, text='Epoch limit:', width=10)
    epoch_field = tk.Spinbox(master=root, from_=0,
                             to=4000, increment=100, width=5)
    epoch_field.place(x=85, y=680)
    epoch_label.place(x=1, y=680)

    # Creates the weights label and each field for the three weights
    weights_label = tk.Label(root, text='ω0: \t ω1: \t   ω2: ')
    w0_field = tk.Spinbox(master=root, from_=0, to=10, increment=.1, width=3)
    w1_field = tk.Spinbox(master=root, from_=0, to=10, increment=.1, width=3)
    w2_field = tk.Spinbox(master=root, from_=0, to=10, increment=.1, width=3)
    weights_label.place(x=1, y=705)
    w0_field.place(x=28, y=705)
    w1_field.place(x=98, y=705)
    w2_field.place(x=168, y=705)
    weights = []    # declares the weights variable in the global scope of the program

    # Creates three buttons (to initialize the weight vector, to start the perceptron's training and to quit the program)
    quit_button = tk.Button(master=root, text='Quit', command=_quit)
    quit_button.pack(side=tk.RIGHT)

    weights_button = tk.Button(
        master=root, text='Initialize weights', command=lambda: _initialize_weights(w0_field, w1_field, w2_field))
    weights_button.pack(side=tk.RIGHT)

    train_button = tk.Button(
        master=root, text='Start Trainning', command=lambda: _train(eta_field, epoch_field))
    train_button.pack(side=tk.RIGHT)

    # Disables the train_button until the weight_button is pressed
    train_button.config(state='disabled')
    tk.mainloop()
