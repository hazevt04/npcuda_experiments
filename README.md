# npcuda-example

I modified the Rober McGibbon's version GitHub.com to work with Python 3, Swig3.0 and the Jetson Xavier. The nvcc command in the setup.py targeted an OLD GPU architecture 2.0 (Fermi), compared to the Xavier's 7.2 (Volta))

The URL to the original GitHub repo:
https://github.com/rmcgibbo/npcuda-example

The rest of this README is a copy from there with minor mods.

This is an example of a simple Python C++ extension which uses CUDA and is compiled via nvcc. The idea is to use this coda as an example or template from which to build your own CUDA-accelerated Python extensions.

The extension is a single C++ class which manages the GPU memory and provides methods to call operations on the GPU
data. This C++ class is wrapped via *swig* or *cython* -- effectively exporting this class into python land.

## swig vs cython

### swig
Swig is a widely used code generator for exposing C and C++ libraries in high level dynamically typed languages.
In principle, it involves minimal code rewriting. You just have to write swig interface files that instruct swig
on how to do the translation.

In practice, swig and numpy don't work together that well. The numpy interface relies on a bunch of magical macros
that are extremely difficult to debug.

### !!cython!!
Cython is sweet. It's basically python, with optional static type declarations and the ability to call c functions
directly. Take a look at `wrapper.pyx`. It looks like python, but it gets translated into C, and then compiled into
a shared object file which you import from python (look at the `test.py` file)

Cython is the way to go.

## difference from PyCUDA

The point of this project is not to enable you to access the CUDA API in python, to write cuda code in strings and have
them be dynamically compiled, or anything like that.

Instead, the goal is to demonstrate some of the biolerplate and tricks needed to make a CPython extension module that
uses CUDA compiled with setuptools/distutils just like your standard C exension modules.

## authors
- Robert McGibbon
- Yutong Zhao
- Glenn Hazelwood

## installation

Requirements:
- python3
- python3 setuptools, numpy 1.18
- nvcc. I'm using version 10.0.326
- swig3.0 for the swig wrapping method. I've tested with version 1.3.40
- cython for the cython wrapping method. I've tested with version 0.29.15

To install, `cd` into your directory of choice -- either `swig` or `cython`. Then, just run `$ sudo CUDAHOME=/usr/local/cuda ./setup.py install`. 
To see if everything is working, run `$ ./test.py`

Silence is golden!

