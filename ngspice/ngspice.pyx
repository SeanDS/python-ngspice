"""Interface to shared ngspice library."""

from io import StringIO
import logging
from libc.stdlib cimport free
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.complex cimport complex
from cpython cimport PyObject, Py_INCREF
import numpy as np
cimport numpy as np
from .data import Solution, Vector
from .util import file_lines


# Initialise numpy. When using numpy from C or Cython we must always do this, otherwise there will
# be segfaults.
np.import_array()


LOGGER = logging.getLogger(__name__)


cdef extern from "NgspiceSession.h":
    cdef cppclass PlotVector:
        PlotVector(int, string, bool) except +
        int index
        string name
        bool real
        vector[double] data_real
        vector[complex[double]] data_complex

    cdef cppclass PlotInfo:
        PlotInfo(string name, string title, string type) except +
        string name
        string title
        string type

    cdef cppclass NgspiceSession:
        NgspiceSession() except +
        void init()
        void reinit()
        bool run()
        bool run_async()
        bool stop_async()
        bool is_running_async()
        bool command(string)
        bool read_netlist(string)
        void print_data()
        vector[PlotInfo] plots()
        vector[PlotVector] plot_vectors(string plot_name)
        PlotVector plot_vector(const string&, const string&)


def run(netlist):
    """Run netlist from string with ngspice and return solutions.

    Parameters
    ----------
    netlist : str
        The netlist to run.

    Returns
    -------
    :class:`dict`
        Map of solution names to :class:`.Solution` objects.
    """
    return run_file(StringIO(netlist))

def run_file(netlist_file):
    """Run netlist from file with ngspice and return solutions.

    Parameters
    ----------
    netlist_file : str, :class:`pathlib.Path`, or :class:`io.FileIO`
        The path or file object to read netlist from. If an open file object is passed, it will be
        read from and left open. If a path is passed, it will be opened, read from, then closed.

    Returns
    -------
    :class:`dict`
        Map of solution names to :class:`.Solution` objects.
    """
    session = Session()
    session.read_file(netlist_file)
    session.run()

    return session.solutions()


cdef class Session:
    """Ngspice session handler."""
    cdef NgspiceSession* session

    def __cinit__(self):
        self.session = new NgspiceSession()

    def __dealloc__(self):
        del self.session

    def read(self, netlist):
        """Read netlist from string.

        Parameters
        ----------
        netlist : str
            The netlist to read.
        """
        return self.read_file(StringIO(netlist))

    def read_file(self, netlist_file):
        """Read netlist from file.

        Parameters
        ----------
        netlist_file : str, :class:`pathlib.Path`, or :class:`io.FileIO`
            The path or file object to read netlist from. If an open file object is passed, it will
            be read from and left open. If a path is passed, it will be opened, read from, then
            closed.
        """
        lines = []
        for line in file_lines(netlist_file):
            line = line.strip()

            if not line or line.startswith("*"):
                continue

            LOGGER.debug(repr(line))
            lines.append(line)

        if not lines:
            raise ValueError("Empty netlist.")
        elif lines[-1].casefold() != ".end":
            raise ValueError("Missing .end statement in netlist.")

        cdef string cscript = "\n".join(lines).encode()

        self.session.reinit()
        return self.session.read_netlist(cscript)

    def run(self):
        """Run ngspice (blocking).

        Returns
        -------
        :class:`int`
            True if the simulation ran successfully, False otherwise.
        """
        cdef int status = self.session.run()
        self.session.print_data()
        return status

    def solutions(self):
        cdef vector[PlotInfo] cplots = self.session.plots()

        solutions = {}

        for i in range(cplots.size()):
            # What ngspice calls "type" is what we call the name, e.g. "op1". What ngspice calls
            # "name", we call the description, e.g. "DC transfer characteristic".
            name = cplots[i].type.decode()
            description = cplots[i].name.decode()
            vectors = self.vectors(name)
            # The type is parsed from the description.
            solution = Solution(name=name, type=description, vectors=vectors)
            solutions[solution.name] = solution

        return solutions

    def vectors(self, plot_name):
        cdef string cname = plot_name.encode()
        cdef vector[PlotVector] cvectors = self.session.plot_vectors(cname)
        vectors = []

        for i in range(cvectors.size()):
            name = cvectors[i].name.decode()
            data = self._vector_array(plot_name, name, cvectors[i].real)
            vectors.append(Vector(name=name, data=data))

        return vectors

    def _vector_array(self, plot_name, vector_name, is_real):
        cdef string analysis = <string> plot_name.encode()
        cdef string vector = <string> vector_name.encode()

        array_wrapper = _ArrayWrapper()

        try:
            array_wrapper.fill(self.session, analysis, vector, is_real)
        except IndexError as err:
            path = f"{plot_name}.{vector_name}"
            raise KeyError(f"Vector {repr(path)} not found") from err

        cdef np.ndarray ndarray = np.array(array_wrapper, copy=False)

        # Assign our object to the 'base' of the ndarray object.
        ndarray.base = <PyObject*> array_wrapper

        # Increment the reference count, as the above assignement was done in C, and Python does not
        # know that there is this additional reference
        Py_INCREF(array_wrapper)

        return ndarray


cdef class _ArrayWrapper:
    """A means of providing pre-existing memory to a new numpy array without copies.

    The NgspiceSession object holds its data in a std::vector which gets held in memory for the
    lifetime of the object. This class holds a reference to a particular array (a std::vector) in
    the object and provides an interface to create a numpy array from it. Encapsulating the
    reference in this way allows the referenced to the memory used for the numpy array to be kept
    as long as the numpy array is kept in memory.

    The NgspiceSession handles the deallocation of the underlying data itself.

    Some background:
      - https://gist.github.com/GaelVaroquaux/1249305
      - Original blog post, no longer available on the original website:
        http://web.archive.org/web/20160321001549/http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/#.Vu89PlSnxhE
    """

    cdef void* dataptr
    cdef int length
    cdef int typenum

    cdef fill(self, NgspiceSession* session, string analysis_type, string vector_name, is_real) except +:
        """Set the data of the array.

        This cannot be done in the constructor as it must recieve C-level arguments.

        Raises
        ------
        IndexError
            When a key doesn't match.
        """
        if is_real:
            self.dataptr = <void*> session.plot_vector(analysis_type, vector_name).data_real.data()
            self.length = session.plot_vector(analysis_type, vector_name).data_real.size()
            self.typenum = np.NPY_DOUBLE
        else:
            self.dataptr = <void*> session.plot_vector(analysis_type, vector_name).data_complex.data()
            self.length = session.plot_vector(analysis_type, vector_name).data_complex.size()
            self.typenum = np.NPY_CDOUBLE

    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.length

        # Create a 1D array with the required length.
        return np.PyArray_SimpleNewFromData(1, shape, self.typenum, self.dataptr)


class SimulationError(Exception):
    """An ngspice simulation error."""
