"""Interface to shared ngspice library."""

from io import StringIO
import logging
import warnings
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.complex cimport complex
import numpy as np
cimport numpy as np
from .data import Solution, Vector
from .util import file_lines


# Initialise numpy. When using numpy from C or Cython we must always do this, otherwise there will
# be segfaults.
np.import_array()


LOGGER = logging.getLogger(__name__)
CLOGGER = logging.getLogger(f"{__name__}.lib")


cdef extern from "NgspiceSession.h":
    ctypedef void (*MessageHandler)(string)

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
        NgspiceSession(MessageHandler log_handler) except +
        void reinit()
        bool run() except +
        bool run_async()
        bool stop_async()
        bool is_running_async()
        bool command(string) except +
        bool read_netlist(string) except +
        vector[PlotInfo] plots()
        vector[PlotVector] plot_vectors(string plot_name)
        PlotVector plot_vector(const string&, const string&)


cdef void _message_handler(string cmessage):
    message = cmessage.decode()

    if message.startswith("stderr Warning: "):
        # Emit warning.
        message = message[16:]
        warnings.warn(message)
    else:
        if message.startswith("stdout "):
            message = message[7:]

        # Emit a log message.
        CLOGGER.info(message)


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
        self.session = new NgspiceSession(<MessageHandler> _message_handler)

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
        cdef int status

        LOGGER.debug("Circuit lines:")
        lines = []
        for line in file_lines(netlist_file):
            line = line.strip()

            if not line or line.startswith("*"):
                continue

            LOGGER.debug(f"  {repr(line)}")
            lines.append(line)

        if not lines:
            raise ValueError("Empty netlist.")
        elif lines[-1].casefold() != ".end":
            raise ValueError("Missing .end statement in netlist.")

        cdef string cscript = "\n".join(lines).encode()
        self.session.reinit()

        try:
            status = self.session.read_netlist(cscript)
        except Exception as err:
            self.session.reinit()
            raise SimulationError(err) from err

        return status

    def run(self):
        """Run ngspice (blocking).

        Returns
        -------
        :class:`int`
            True if the simulation ran successfully, False otherwise.
        """
        cdef int status

        try:
            status = self.session.run()
        except Exception as err:
            self.session.reinit()
            raise SimulationError(err) from err

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
        cdef string cplot_name = plot_name.encode()
        cdef vector[PlotVector] cvectors = self.session.plot_vectors(cplot_name)
        vectors = {}

        for i in range(cvectors.size()):
            name = cvectors[i].name.decode()
            data = self._vector_array(cplot_name, cvectors[i].name, cvectors[i].real)
            vectors[name] = Vector(name=name, data=data)

        return vectors

    cdef np.ndarray _vector_array(self, string analysis, string vector, bool is_real):
        cdef np.ndarray array
        cdef np.npy_intp shape[1]

        if is_real:
            shape[0] = <np.npy_intp> self.session.plot_vector(analysis, vector).data_real.size()
            array = np.PyArray_SimpleNewFromData(
                1,
                shape,
                np.NPY_DOUBLE,
                self.session.plot_vector(analysis, vector).data_real.data()
            )
        else:
            shape[0] = <np.npy_intp> self.session.plot_vector(analysis, vector).data_complex.size()
            array = np.PyArray_SimpleNewFromData(
                1,
                shape,
                np.NPY_CDOUBLE,
                self.session.plot_vector(analysis, vector).data_complex.data()
            )

        return np.array(array, copy=True)


class SimulationError(Exception):
    """An ngspice simulation error."""
