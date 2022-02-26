#include "NgspiceSession.h"

using namespace std;

NgspiceSession::NgspiceSession(LogHandler log_handler) : log_handler(log_handler) {
    bool success;

    success = ngSpice_Init(
        &cb_send_char,
        &cb_send_status,
        &cb_controlled_exit,
        &cb_send_data,
        &cb_send_plot_data,
        &cb_background_thread_running,
        this
    ) == 0;

    success &= command("reset");

    if (!success) {
        throw std::runtime_error("Could not initialise ngspice");
    }
}

NgspiceSession::~NgspiceSession() {

}

bool NgspiceSession::reinit() {
    // FIXME: causes segfault
    // plot_info_map.clear();
    // current_plot_name = nullptr;
    // plot_vector_map.clear();
    return command("destroy all");
}

bool NgspiceSession::read_netlist(const string& netlist) {
    std::vector<char*> lines;
    std::string line;
    stringstream ss(netlist);

    while (std::getline(ss, line, '\n')) {
        lines.push_back(strdup(line.c_str()));
    }

    lines.push_back(nullptr); // ngSpice_Circ wants a null-terminated array.
    ngSpice_Circ(lines.data());

    // Use of strdup above to satisfy ngSpice_Circ's required type requires freeing of memory on
    // the heap.
    for( auto line : lines ) {
        free(line);
    }

    return true;
}

bool NgspiceSession::run() {
    return command("run");
}

bool NgspiceSession::run_async() {
    return command("bg_run");
}

bool NgspiceSession::stop_async() {
    return command("bg_halt");
}

bool NgspiceSession::is_running_async() {
    return ngSpice_running();
}

bool NgspiceSession::command(const string& command) {
    return ngSpice_Command((char*) command.c_str()) == 0;
}

std::vector<PlotInfo> NgspiceSession::plots() {
    std::vector<PlotInfo> plots;

    for (const auto& name_and_plot : plot_info_map) {
        plots.push_back(name_and_plot.second);
    }

    return plots;
}

std::vector<PlotVector> NgspiceSession::plot_vectors(const std::string& plot_type) {
    std::vector<PlotVector> plot_vectors;

    for (const auto& name_and_vector : plot_vector_map.at(plot_type)) {
        plot_vectors.push_back(name_and_vector.second);
    }

    return plot_vectors;
}

PlotVector& NgspiceSession::plot_vector(const std::string& plot_type, const std::string& vector_name) {
    return plot_vector_map.at(plot_type).at(vector_name);
}

void NgspiceSession::print_data() {
    for (const auto& name_and_plot : plot_info_map) {
        PlotInfo plot = name_and_plot.second;

        printf("%s ('%s'):\n", plot.name.c_str(), plot.title.c_str());

        for (const auto& name_and_vectors :
            plot_vector_map.at(name_and_plot.first)) {
            PlotVector vec = name_and_vectors.second;

            printf("\t%s: ", vec.name.c_str());

            if (vec.real) {
                for (const auto& value : vec.data_real) {
                    printf("%f ", value);
                }
            } else {
                for (const auto& value : vec.data_complex) {
                    printf("%f + %fi ", value.real(), value.imag());
                }
            }

            printf("\n");
        }
    }
}

void NgspiceSession::_add_ngspice_plot(vecinfoall* vinfo) {
    plot_info_map.emplace(
        std::piecewise_construct, std::make_tuple(std::string(vinfo->type)),
        std::make_tuple(std::string(vinfo->name), std::string(vinfo->title),
            std::string(vinfo->type)));

    // Set the current plot name.
    current_plot_name = vinfo->type;

    // Create the plot vectors.
    for (int i = 0; i < vinfo->veccount; i++) {
        plot_vector_map[current_plot_name].emplace(
            std::piecewise_construct,
            std::make_tuple((std::string)vinfo->vecs[i]->vecname),
            std::make_tuple(vinfo->vecs[i]->number,
                (std::string)vinfo->vecs[i]->vecname,
                (bool)vinfo->vecs[i]->is_real));
    }
}

void NgspiceSession::_add_ngspice_data(vecvaluesall* vinfo) {
    for (int i = 0; i < vinfo->veccount; i++) {
        PlotVector& vec = plot_vector_map[current_plot_name].at(vinfo->vecsa[i]->name);

        if (vec.real) {
            vec.data_real.emplace_back(vinfo->vecsa[i]->creal);
        } else {
            vec.data_complex.emplace_back(vinfo->vecsa[i]->creal,
                vinfo->vecsa[i]->cimag);
        }
    }
}

void NgspiceSession::log(std::string message) {
    (*log_handler)(message);
}

int NgspiceSession::cb_send_char(char* aWhat, int aId, void* aUser) {
    NgspiceSession* sim = reinterpret_cast<NgspiceSession*>(aUser);
    sim->log((std::string) aWhat);
    return 0;
}

int NgspiceSession::cb_send_status(char* aWhat, int aId, void* aUser) {
    NgspiceSession* sim = reinterpret_cast<NgspiceSession*>(aUser);
    sim->log((std::string) aWhat);
    return 0;
}

int NgspiceSession::cb_background_thread_running(bool aFinished, int aId, void* aUser) {
    return 0;
}

int NgspiceSession::cb_controlled_exit(int aStatus, bool aImmediate, bool aExitOnQuit, int aId, void* aUser) {
    return 0;
}

int NgspiceSession::cb_send_data(vecvaluesall* vdata, int len, int ng_ident, void* aUser) {
    NgspiceSession* sim = reinterpret_cast<NgspiceSession*>(aUser);
    sim->_add_ngspice_data(vdata);
    return 0;
}

int NgspiceSession::cb_send_plot_data(vecinfoall* vecdata, int ng_ident, void* aUser) {
    NgspiceSession* sim = reinterpret_cast<NgspiceSession*>(aUser);
    sim->_add_ngspice_plot(vecdata);
    return 0;
}
