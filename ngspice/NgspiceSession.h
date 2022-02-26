#ifndef NGSPICESESSION_H
#define NGSPICESESSION_H

#include <algorithm>
#include <complex>
#include <map>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <tuple>
#include <utility>
#include <vector>
#include <ngspice/sharedspice.h>


typedef void (*LogHandler)(std::string);


/**
 * Plot vector data and metadata container.
 *
 * This holds the data for a particular node, junction, or sweep of a
 * simulation, alongside the name of the node/junction/sweep and other metadata.
 */
class PlotVector {
public:
    PlotVector(int index, std::string name, bool is_real)
        : index(index), name(name), real(is_real) {}

    int index;
    std::string name;
    bool real;

    // Use separate variables for real and complex valued vectors so we can pass
    // references to Python. Only one is actually used, depending on the value of
    // `real`.
    std::vector<double> data_real;
    std::vector<std::complex<double>> data_complex;
};

/**
 * Plot metadata container.
 *
 * This holds the metadata for what ngspice calls a "plot", which is a
 * particular analysis as defined in the netlist (e.g. ".op").
 */
class PlotInfo {
public:
    PlotInfo(std::string name, std::string title, std::string type)
        : name(name), title(title), type(type) {}

    std::string name;
    std::string title;
    std::string type;
};

/**
 * Ngspice session handler.
 *
 * This interfaces with the ngspice shared library to set up, run and handle the
 * outputs from simulations.
 */
class NgspiceSession {
public:
    NgspiceSession(LogHandler log_handler);
    virtual ~NgspiceSession();

    bool init();
    bool reinit();
    bool run();
    bool run_async();
    bool stop_async();
    bool is_running_async();
    bool command(const std::string& command);
    bool read_netlist(const std::string& netlist);
    std::vector<PlotInfo> plots();
    std::vector<PlotVector> plot_vectors(const std::string& plot_type);
    PlotVector& plot_vector(const std::string& plot_type, const std::string& vector_name);

    void print_data();
    void _add_ngspice_plot(vecinfoall*);
    void _add_ngspice_data(vecvaluesall*);

    void log(std::string message);

private:
    LogHandler log_handler;

    // Ngspice simulation plot and vector data storage; `plots` is keyed by plot
    // type (e.g. 'op1'), `plot_vectors` is keyed by plot type then vector name
    // (e.g. 'n1').
    std::map<std::string, PlotInfo> plot_info_map;
    std::string current_plot_name;
    std::map<std::string, std::map<std::string, PlotVector>> plot_vector_map;

    // Callback functions.
    static int cb_send_char(char* what, int aId, void* aUser);
    static int cb_send_status(char* what, int aId, void* aUser);
    static int cb_background_thread_running(bool aFinished, int aId, void* aUser);
    static int cb_controlled_exit(int aStatus, bool aImmediate, bool aExitOnQuit, int aId, void* aUser);
    static int cb_send_data(vecvaluesall*, int len, int ng_ident, void* userdata);
    static int cb_send_plot_data(vecinfoall*, int ng_ident, void* userdata);
};

#endif /* NGSPICESESSION_H */
