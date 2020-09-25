# Artificial Intelligence

## Summary
This project is intended to host the main artificial intelligence algorithms and more specifically those related to machine learning; it extends from the main artificial neurons to more complex neural networks, all accompanied by graphical interfaces and theoretical content. The theoretical content is based on the content taught in the subject of Artificial Intelligence II at UdG's CUCEI, by [Dr. Nancy Guadalupe Arana Daniel](http://www.cucei.udg.mx/doctorados/electronica/es/dra-nancy-guadalupe-arana-daniel).

## Structure
This repository has the following structure:
```
artificial-intelligence
│   README.md
|   requirements.txt
│   license.txt    
│
└───.vscode
│   │   settings.json
│   
└───perceptron
|   │   main.py
|   │   perceptron.py
|   │   ...
│   
└───adaline
|   │   main.py
|   │   adaline.py
|   │   ...
|
└───...  
```

## Languages, tools & packages

### [Python 3.8.2](https://www.python.org/downloads/release/python-382/)
Python is a great language for data science and AI due to the various libraries and packages that it has. Some of these libraries help to perform algebraic and matrix operations (many of the strategies applied in the present practices have been carried out through matrix operations). To configure the environment to deploy this project, we must work on a virtual environment provide by the virtualenv tool.

#### [virtualenv](https://virtualenv.pypa.io/en/latest/)
`virtualenv` is a tool to create isolated [Python environments](https://docs.python.org/3/tutorial/venv.html). Since Python 3.3, a subset of it has been integrated into the standard library under the venv module. The `venv` module included in the standar library does not offer all features of this library, therefore we must install the full library. To do this, I recommend using Python's package installer `pip`.

```bash
pip install virtualenv | pip3 install virtualenv
```
To create a virtual environment for the project, the following command must be executed inside the project folder:

```bash
python -m venv `env_name` | python3 -m venv `env_name`
```
Finally, to activate the venv from inside the project folder:

```bash
# On Linux/macOS
source `env_name`/bin/activate
```
```shell
# On windows
\`env_name`\Scripts\activate.bat
```

#### pip
Once the virtual environment is created, the required packages can be installed via `pip` using the [requirements.txt](https://github.com/the-eternal-newbie/artificial-intelligence/blob/master/requirements.txt) file with the following command:

```bash
pip install -r requirements.txt
```
### Packages
The required packages are used to make algebraic operations and GUI design. Below are shown the packages and why are they used for.

#### [numpy](https://numpy.org/)
NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

At the core of the NumPy package, is the ndarray object. This encapsulates n-dimensional arrays of homogeneous data types, with many operations being performed in compiled code for performance.

#### [tkinter](https://docs.python.org/3/library/tkinter.html)
Tkinter is a Python binding to the Tk GUI toolkit. It is the standard Python interface to the Tk GUI toolkit, and is Python's de facto standard GUI. Tkinter is included with standard Linux, Microsoft Windows and Mac OS X installs of Python. Tkinter is free software released under a Python license.

#### [matplotlib](https://matplotlib.org/)
<img src=https://warehouse-camo.ingress.cmh1.psfhosted.org/42ca79ff99d75bf2cb4e6097c8006b52d36484df/68747470733a2f2f6d6174706c6f746c69622e6f72672f5f7374617469632f6c6f676f322e737667 width=300p align=left></br></br></br>
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib produces publication-quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shell, web application servers, and various graphical user interface toolkits.
<img src=https://warehouse-camo.ingress.cmh1.psfhosted.org/e7ea6d65132d8dca8553640aac16d4b6389f89d6/68747470733a2f2f6d6174706c6f746c69622e6f72672f5f7374617469632f726561646d655f707265766965772e706e67 width=700p align=center>

## Content
- [Perceptron](https://github.com/the-eternal-newbie/artificial-intelligence/tree/master/perceptron)
- [Adaline]
