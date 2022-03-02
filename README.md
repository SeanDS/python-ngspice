# Ngspice for Python
This is a Python package for running [ngspice](ngspice.sourceforge.net/) simulations and extracting
output. It wraps ngspice as a static library so it does not need to be available on the system, and
no configuration is needed. Output data are provided as numpy arrays. The library is fast, using a
C++ wrapper to perform low-level interaction with ngspice.

## Requirements
In terms of system dependencies, only Python 3.8+ is required to install and run `pyngspice`. Numpy
is also required as a Python dependency; this gets installed automatically if you use a packaging
tool like pip.

For most users, running the following from a console is all that is needed:

```console
$ pip install git+https://github.com/SeanDS/pyngspice.git
```

To build `pyngspice` yourself, you will also need the `ngspice` development header files, and a
few more Python dependencies (also automatically installed during the build). Building can then be
done by running `pip install .` from your local working copy of the project's git repository.

## Usage
`pyngspice` provides `run` and `run_file` functions to run netlist strings and files,
respectively. These return a dictionary containing solutions for the analyses defined in the
netlist.

```python
from ngspice import run

# Define the ngspice script.
netlist = (
    """
    Voltage divider operating point simulation.
    R1 n1 n2 1k
    R2 n2 0 2k
    V1 n1 0 DC 1
    .op
    .end
    """
)

# Run ngspice and get the results.
solutions = run(netlist)
op = solutions["op1"]

# Print the results.
for name, vector in zip(op.row_names, op.vectors):
    rowvals = [f"{value:10.2e}" for value in vector.data]
    print(f"{name:10s} {' '.join(rowvals)}")
```

The output from the above script is:

```
n1           1.00e+00
n2           6.67e-01
v1#branch   -3.33e-04
```

## Information for developers

### Building from source
To build `python-ngspice` from source, you must have Ngspice installed and available as a shared
library (`sharedspice.h` header file and a `libngspice` shared object available on the system
include path).

### Code description
The project provides a small C++ wrapper to interact with ngspice (based on that of
[KiCad](https://www.kicad.org/)), which itself is operated from Python by Cython-generated bindings.
In theory Cython would be capable of directly interacting with ngspice but its first class support
for C++ is limited.

The C++ wrapper dynamically loads the ngspice shared object at runtime. This would not normally be
necessary except ngspice often seems to crash when it encounters an error, and so it is necessary
to reset its state.

Data outputs from ngspice simulations are returned to Python as numpy arrays through the use of a
single memory copy operations in C, avoiding any copies to Python objects. This means simulations
are very fast to perform from Python, even inside loops. In principle even the single copy operation
can be avoided by manually assigning responsibility for management of the underlying memory to
Numpy, but this has not been implemented yet.

### Running tests
The project uses `pytest`. Simply run `pytest` from the project root.

### Contributing
Bug reports and feature requests are always welcome, as are code contributions. Please use the
project's [issue tracker](https://github.com/SeanDS/pyngspice/issues).

## Credits
Sean Leavey  
<https://github.com/SeanDS/>

Some code for interfacing with ngspice based on KiCad, licenced under GPL. Ngspice has various
licences. All adapted and statically linked code is compatible with this project's BSD licence.
