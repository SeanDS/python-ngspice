# python-ngspice
`python-ngspice` is a simple Python wrapper for running [ngspice](ngspice.sourceforge.net/)
simulations and extracting outputs as Numpy arrays. It wraps ngspice as a static library so it does
not need to be available on the system, and no configuration is needed.

## Requirements
In terms of system dependencies, only Python 3.8+ is required to install and run `python-ngspice`.
Numpy is also required as a Python dependency; this gets installed automatically if you use a
packaging tool like pip.

For most users, running the following from a console is all that is needed:

```console
$ pip install git+https://github.com/SeanDS/python-ngspice.git
```

To build `python-ngspice` yourself, you will also need the `ngspice` development header files, and a
few more Python dependencies (also automatically installed during the build). Building can then be
done by running `pip install .` from your local working copy of the project's git repository.

## Usage
`python-ngspice` provides `run` and `run_file` functions to run netlist strings and files,
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

## Contributing
Bug reports and feature requests are always welcome, as are code contributions. Please use the
project's [issue tracker](https://github.com/SeanDS/python-ngspice/issues).
