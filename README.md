[![CI](https://github.com/tdegeus/GooseFEM/workflows/CI/badge.svg)](https://github.com/tdegeus/GooseFEM/actions)
[![Doxygen -> gh-pages](https://github.com/tdegeus/GooseFEM/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/GooseFEM)
[![readthedocs](https://readthedocs.org/projects/goosefem/badge/?version=latest)](https://readthedocs.org/projects/goosefem/badge/?version=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/goosefem.svg)](https://anaconda.org/conda-forge/goosefem)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/python-goosefem.svg)](https://anaconda.org/conda-forge/python-goosefem)

## GooseFEM

Library to perform static or dynamic finite elements computations. The core of the implementation is a C++ library. For user convenience a Python interface is provided too. Please consult the documentation for more information:

https://goosefem.readthedocs.io

and for C++ documentation

https://tdegeus.github.io/GooseFEM

## Credit / copyright

(c) T.W.J. de Geus | [www.geus.me](http://www.geus.me) | [tom@geus.me](mailto:tom@geus.me)

Tom de Geus was financially supported by:

*   [Swiss National Science Foundation (FNSF), Switzerland](http://www.snfs.ch)
*   [École Polytechnique Fédérale de Lausanne (EPFL), Lausanne, Switzerland](http://www.epfl.ch)
*   [Eindhoven University of Technology (TU/e), Eindhoven, The Netherlands](http://www.tue.nl)
*   [The Netherlands Research Council (NWO), The Netherlands](http://www.nwo.nl)
*   [Materials Innovation Institute (M2i), The Netherlands](http://www.m2i.nl)


## Installation Guidelines / Help with GMatTensor
If changes in GMatTensor are not recognized by GooseFEM, then because GooseFEM looks for the GMatTensor files under e.g.C:\Users\20250672\AppData\Local\anaconda3\envs\goose\Library. Often, if GMatTensor is compiled, the files are installed under site-packages.

To overcome this, you can force GMatTensor to install under your desired path C:\Users\20250672\AppData\Local\anaconda3\envs\goose\Library by providing this
path when compiling.

In your GMatTensor package, you do the following:
1. cmake -S . -B build -DCMAKE_INSTALL_PREFIX=C:\Users\20250672\AppData\Local\anaconda3\envs\goose\Library
2. cmake --build build
3. cmake --install build


