import sys
import json
import weakref
import numpy as np
import tkinter as tk

from os import path
from os import execl
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler

sys.path.insert(0, "./neuron")
sys.path.insert(0, "./network")

from neuron.adaline import Adaline
from neuron.perceptron import Perceptron

# ! This GUI evolves frome the one implemented on the Perceptron project

root = tk.Tk()
root.wm_title("HW2 - Adaline & Perceptron")
root.geometry("700x765")
root.resizable(False, False)
# Everytime the program is executed, a new file is created, replacing the previous one to erase previous data
file = open('bulk_data.json', 'w')
file.write('[]')  # Writes a json list element
file.close()
global quit_button
global perceptron_button
global adaline_button
global weights_button
global refresh_button
global data_set
global weights


class Layout(object):
    def __init__(self, root, title, size=5):
        fig = Figure(figsize=(7, 7), dpi=100)
        self.perceptron_weights = None
        self.perceptron_trained = False
        self.adaline_weights = None
        self.adaline_trained = False
        self.ax = fig.add_subplot(111)
        self.ax.set_title(title)

        # Makes the plot fixed (prevents from resizing)
        self.ax.set(xlim=(-size, size), ylim=(-size, size))
        # Adds guide lines
        self.ax.axhline(y=0, color="black")
        self.ax.axvline(x=0, color="black")
        # Draw arrow points
        self.ax.scatter(0, size - .1, marker='^', color='black')
        self.ax.scatter(size - .1, 0, marker='>', color='black')
        self.ax.scatter(0, -size + .1, marker='v', color='black')
        self.ax.scatter(-size + .1, 0, marker='<', color='black')

        # Connect the plot with the GUI interface
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.NONE, expand=False)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    # Left click -> Class with 0s
    # Right click -> Class with 1s
    def on_click(self, event):
        ix, iy = event.xdata, event.ydata
        if(ix != None):
            if(self.perceptron_trained and event.button == 3):
                point = {'coord': None, 'class': None, 'color': 'pink'}
                point['coord'] = [-1, round(ix, 2), round(iy, 2)]
                point['class'] = 0
                if(np.dot(self.perceptron_weights, np.array(point['coord'])) >= 0):
                    point['class'] = 1
                    point['color'] = 'green'

                self.ax.scatter(point['coord'][1],
                                point['coord'][2], color=point['color'])
                self.ax.annotate('Class {}'.format(
                    point['class']), (point['coord'][1] + .5, point['coord'][2]))
                self.canvas.draw()  # Refreshes the canvas

            elif(self.adaline_trained and event.button == 1):
                point = {'coord': None, 'class': None, 'color': 'gray'}
                point['coord'] = [-1, round(ix, 2), round(iy, 2)]
                point['class'] = 1
                y = np.dot(self.adaline_weights, np.array(point['coord']))
                activation = 1 / (1 + np.exp(-y))
                if(activation <= 0.5):
                    point['class'] = 0
                    point['color'] = 'yellow'

                self.ax.scatter(point['coord'][1],
                                point['coord'][2], color=point['color'])
                self.ax.annotate('Class {}'.format(
                    point['class']), (point['coord'][1] + .5, point['coord'][2]))
                self.canvas.draw()  # Refreshes the canvas
            else:
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
                self.ax.scatter(point['coord'][1],
                                point['coord'][2], color=color)
                self.canvas.draw()  # Refreshes the canvas
                data_set.append(point)


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
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)


def window_error(epoch_amount, error_freq):
    window = tk.Toplevel(root)
    window.geometry("700x765+750+68")
    window.resizable(False, False)
    error_view = ErrorLayout(
        window, 'Error', x_limit=epoch_amount, y_limit=max(error_freq))
    x = np.arange(epoch_amount)
    error_view.ax.bar(x, height=error_freq, align='edge',
                      width=0.6, color='orange')
    error_view.canvas.draw()


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL tstate


def _initialize_weights(w0_field, w1_field, w2_field):
    perceptron_button.config(state='normal')
    adaline_button.config(state='normal')
    weights_button.config(state='disabled')
    try:
        weights = [float(w0_field.get()), float(
            w1_field.get()), float(w2_field.get())]

        x = np.linspace(-5, 5, 100)
        if(weights[2] == 0):
            raise ValueError
        y = (weights[0] - (weights[1] * x)) / weights[2]
        layout.ax.plot(x, y, color='gray')
        layout.canvas.draw()
    except ValueError as error:
        perceptron_button.config(state='disabled')
        adaline_button.config(state='disabled')
        weights_button.config(state='normal')
        messagebox.showerror(
            error, 'Weight values must be different from zero!')


# Starts the perceptron algorithm
def _train(eta_field, epoch_field, neuron='perceptron', sqre_field=None):
    try:
        eta = float(eta_field.get())
        epoch_limit = int(epoch_field.get())
        sqre = 0 if sqre_field == None else float(sqre_field.get())
        with open('bulk_data.json', 'r+') as file:
            file.seek(0)
            json.dump(data_set, file)

        args = {'bulk_data': data_set, 'weights': weights}
        # If eta, epoch_limit or the desired quadratic error are equal to zero,
        # then, the Neuron must take default values; to do that
        # the args in initialization must be null
        if(eta != 0):
            args['eta'] = eta
        if(epoch_limit != 0):
            args['epoch_limit'] = epoch_limit
        if(neuron == 'adaline' and sqre != 0):
            args['sqre'] = sqre

        try:
            if(neuron == 'perceptron'):
                trainer = Perceptron(**args)
                trainer.process()
                layout.perceptron_weights = trainer.weights
                layout.perceptron_trained = True
            else:
                trainer = Adaline(**args)
                trainer.process()
                layout.adaline_weights = trainer.weights
                layout.adaline_trained = True
        except AttributeError as error:
            messagebox.showerror(error, 'Provided data not found!')

        if(layout.adaline_trained == False or layout.perceptron_trained == False):
            l = layout.ax.lines.pop(2)
            wl = weakref.ref(l)
            del l
        x = np.linspace(-5, 5, 100)
        if(neuron == 'perceptron'):
            line_color = 'blue'
        elif(neuron == 'adaline'):
            line_color = 'red'
        for line in trainer.lines:
            y = (line[0] - (line[1] * x)) / line[2]
            lines = layout.ax.plot(x, y, color=line_color)
            l = lines.pop()
            wl = weakref.ref(l)
            layout.canvas.draw()
            l.remove()
            del l
        y = (trainer.weights[0][0] -
             (trainer.weights[0][1] * x)) / trainer.weights[0][2]
        layout.ax.plot(x, y, color=line_color)
        messagebox.showinfo('{} training has finished'.format(neuron.title()),
                            'The solution was found in the epoch number {}'.format(trainer.current_epoch))
        refresh_button.config(state='normal')
        window_error(trainer.current_epoch, trainer.error_freq)
    except ValueError as error:
        messagebox.showerror(
            error, 'Input values must be float (for eta & quadratic error) and integer (for epoch limit)!')


def _refresh():
    root.destroy()
    python = sys.executable
    execl(python, python, *sys.argv)


if __name__ == "__main__":
    data_set = []
    layout = Layout(root, 'Perceptron', 5)

    # Creates the eta label and field
    eta_label = tk.Label(root, text='η value:', width=10, anchor=tk.S)
    eta_field = tk.Spinbox(master=root, from_=0, to=5, increment=.1, width=5)
    eta_field.place(x=85, y=700)
    eta_label.place(x=1, y=700)

    # Creates the epoch limit label and field
    epoch_label = tk.Label(root, text='Epoch limit:', width=10)
    epoch_field = tk.Spinbox(master=root, from_=0,
                             to=4000, increment=100, width=5)
    epoch_field.place(x=85, y=720)
    epoch_label.place(x=1, y=720)

    # Creates the squared error label and field
    sqre_label = tk.Label(root, text='Desired ε²:', width=10)
    sqre_field = tk.Spinbox(master=root, from_=0,
                            to=10, increment=.001, width=5)
    sqre_field.place(x=85, y=740)
    sqre_label.place(x=1, y=740)

    # Creates the weights label and each field for the three weights
    weights_label = tk.Label(root, text='ω0:\n\nω1:\nω2:')
    w0_field = tk.Spinbox(master=root, from_=0, to=10, increment=.1, width=3)
    w0_label = tk.Label(root, text='ω₀:')

    w1_field = tk.Spinbox(master=root, from_=0, to=10, increment=.1, width=3)
    w1_label = tk.Label(root, text='ω₁:')

    w2_field = tk.Spinbox(master=root, from_=0, to=10, increment=.1, width=3)
    w2_label = tk.Label(root, text='ω₂:')

    w0_field.place(x=178, y=700)
    w0_label.place(x=150, y=700)

    w1_field.place(x=178, y=720)
    w1_label.place(x=150, y=720)

    w2_field.place(x=178, y=740)
    w2_label.place(x=150, y=740)

    weights = []    # declares the weights variable in the global scope of the program

    # * Creates three buttons (to initialize the weight vector, to start the perceptron's training and to quit the program)
    quit_button = tk.Button(master=root, text='Quit', command=_quit)
    quit_button.place(x=645, y=730)

    refresh_button = tk.Button(master=root, text='Restart', command=_refresh)
    refresh_button.place(x=575, y=730)

    weights_button = tk.Button(
        master=root, text='Initialize weights', command=lambda: _initialize_weights(w0_field, w1_field, w2_field))
    weights_button.place(x=445, y=730)

    # * Training buttons
    perceptron_button = tk.Button(
        master=root, text='Start Perceptron Training', command=lambda: _train(eta_field=eta_field, epoch_field=epoch_field, neuron='perceptron'))
    perceptron_button.place(x=326, y=700)

    adaline_button = tk.Button(
        master=root, text='Start Adaline Training', command=lambda: _train(eta_field=eta_field, epoch_field=epoch_field, sqre_field=sqre_field, neuron='adaline'))
    adaline_button.place(x=523, y=700)

    # Disables the perceptron & adaline buttons until the weight_button is pressed
    perceptron_button.config(state='disabled')
    adaline_button.config(state='disabled')
    refresh_button.config(state='disabled')
    tk.mainloop()
