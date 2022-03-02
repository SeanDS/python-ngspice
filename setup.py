from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        [
            Extension(
                "ngspice.ngspice",
                ["ngspice/ngspice.pyx", "ngspice/NgspiceSession.cpp"],
                libraries=["ngspice"],  # Dynamically loaded at runtime by NgspiceSession.cpp.
                include_dirs=[np.get_include()],
                language="c++"
            )
        ],
        compiler_directives={"profile": True},
        gdb_debug=True
    )
)
