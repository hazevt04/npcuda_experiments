## cython wrapped CUDA/C++

This code makes an explicit cython class that wraps the C++ class, exposing it in python. It involves a little bit more repitition than the swig code in principle, but in practice it's MUCH easier.

This uses cudaHostRegister() to pin the array memory for use with CUDA for much higher bandwidth than unregistered pageable memory.

To install:

`$ sudo CUDAHOME=<path to CUDA> python setup.py install`

to test:

`$ ./test.py`

you need a relatively recent version of cython (>=0.16).



