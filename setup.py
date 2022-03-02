from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        [
            Extension(
                "ngspice.ngspice",
                ["ngspice/ngspice.pyx", "ngspice/NgspiceSession.cpp"],
                runtime_library_dirs=["ngspice"],
                include_dirs=[np.get_include()],
                language="c++"
            )
        ],
        compiler_directives={"profile": True},
        gdb_debug=True
    )
)
