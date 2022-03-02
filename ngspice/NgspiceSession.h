#ifndef NGSPICESESSION_H
#define NGSPICESESSION_H

#include <dlfcn.h>
#include <algorithm>
#include <complex>
#include <map>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <ngspice/sharedspice.h>

using namespace std;


typedef void (*MessageHandler)(string);


/**
 * Plot vector data and metadata container.
 *
 * This holds the data for a particular node, junction, or sweep of a
 * simulation, alongside the name of the node/junction/sweep and other metadata.
 */
class PlotVector {
public:
    PlotVector(int index, string name, bool is_real) : index(index), name(name), real(is_real) {}

    int index;
    string name;
    bool real;

    // Use separate variables for real and complex valued vectors so we can pass
    // references to Python. Only one is actually used, depending on the value of
    // `real`.
    vector<double> data_real;
    vector<complex<double>> data_complex;
};

/**
 * Plot metadata container.
 *
 * This holds the metadata for what ngspice calls a "plot", which is a
 * particular analysis as defined in the netlist (e.g. ".op").
 */
class PlotInfo {
public:
    PlotInfo(string name, string title, string type) : name(name), title(title), type(type) {}

    string name;
    string title;
    string type;
};

/**
 * Ngspice session handler.
 *
 * This interfaces with the ngspice shared library to set up, run and handle the
 * outputs from simulations.
 */
class NgspiceSession {
public:
    NgspiceSession(MessageHandler message_handler);
    virtual ~NgspiceSession();

    // DLL (re)initialization at runtime.
    bool init();

    // Perform a blocking run.
    bool run();

    // Perform a non-blocking run, waiting in another thread until ngspice finishes, then returning.
    bool run_async();
    // Start/stop a non-blocking run manually, and check if it's running.
    bool start();
    bool stop();
    bool running();

    // Send command to ngspice.
    bool command(const string& command);
    // Read a netlist into ngspice.
    bool read_netlist(const string& netlist);

    // Get ngspice output data.
    vector<PlotInfo> plots();
    PlotInfo& plot(const string& plot_type);
    vector<PlotVector> plot_vectors(const string& plot_type);
    PlotVector& plot_vector(const string& plot_type, const string& vector_name);

    // Print ngspice output data.
    void print_data();

    // Store data produced by ngspice (only intended to be used by ngspice callbacks).
    void _add_ngspice_plot(vecinfoall*);
    void _add_ngspice_data(vecvaluesall*);

    // Send messages from ngspice to the registered handler.
    void emit_message(string message);

private:
    // Ensure ngspice is in a valid state.
    void validate();

    // Error flag indicating that ngspice needs to be reloaded.
    bool m_error = false;

    // Ngspice shared object handle and function signatures.
    void* m_ngspice = nullptr;
    typedef int (*ngSpice_Init)(SendChar*, SendStat*, ControlledExit*, SendData*, SendInitData*, BGThreadRunning*, void*);
    typedef int (*ngSpice_Circ)(char** circarray);
    typedef int (*ngSpice_Command)(char* command);
    typedef bool (*ngSpice_Running)(void);

    // Shared object functions.
    ngSpice_Init m_ngSpice_Init;
    ngSpice_Circ m_ngSpice_Circ;
    ngSpice_Command m_ngSpice_Command;
    ngSpice_Running m_ngSpice_Running;

    // Callback functions for ngspice simulations.
    static int cb_send_char(char* what, int aId, void* aUser);
    static int cb_send_status(char* what, int aId, void* aUser);
    static int cb_background_thread_running(bool aFinished, int aId, void* aUser);
    static int cb_controlled_exit(int aStatus, bool aImmediate, bool aExitOnQuit, int aId, void* aUser);
    static int cb_send_data(vecvaluesall*, int len, int ng_ident, void* userdata);
    static int cb_send_plot_data(vecinfoall*, int ng_ident, void* userdata);

    // Ngspice output handler.
    MessageHandler message_handler;

    // Ngspice simulation plot and vector data storage; `plots` is keyed by plot type (e.g. 'op1'),
    // `plot_vectors` is keyed by plot type then vector name (e.g. 'n1').
    map<string, PlotInfo> plot_info_map;
    string current_plot_name;
    map<string, map<string, PlotVector>> plot_vector_map;
};

#endif /* NGSPICESESSION_H */
