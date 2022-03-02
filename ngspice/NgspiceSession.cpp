#include "NgspiceSession.h"

using namespace std;

NgspiceSession::NgspiceSession(MessageHandler message_handler) : message_handler(message_handler) {
    init();
}

NgspiceSession::~NgspiceSession() {
    dlclose(m_ngspice);
}

bool NgspiceSession::init() {
    bool success;

    // Close any existing object.
    if (m_ngspice != nullptr) {
        dlclose(m_ngspice);
    }

    // Load the DLL.
    m_ngspice = dlopen("/usr/lib/libngspice.so", RTLD_NOW);

    if (m_ngspice == nullptr) {
        throw std::runtime_error("Could not find/initialise libngspice");
    }

    m_error = false;

    // Assign functions.
    m_ngSpice_Init = (ngSpice_Init) dlsym(m_ngspice, "ngSpice_Init");
    m_ngSpice_Circ = (ngSpice_Circ) dlsym(m_ngspice, "ngSpice_Circ");
    m_ngSpice_Command = (ngSpice_Command) dlsym(m_ngspice, "ngSpice_Command");
    m_ngSpice_Running = (ngSpice_Running) dlsym(m_ngspice, "ngSpice_running");  // Not a typo.

    success = m_ngSpice_Init(
        &cb_send_char,
        &cb_send_status,
        &cb_controlled_exit,
        &cb_send_data,
        &cb_send_plot_data,
        &cb_background_thread_running,
        this
    ) == 0;

    // Workarounds to avoid crashes on certain errors.
    success &= command("unset interactive");
    success &= command("set noaskquit");
    success &= command("set nomoremode");

    if (!success) {
        throw std::runtime_error("Could not initialize simulation");
    }

    return success;
}

void NgspiceSession::validate() {
    if (m_error) {
        init();
    }
}

bool NgspiceSession::read_netlist(const string& netlist) {
    bool status;
    std::vector<char*> lines;
    std::string line;
    stringstream ss(netlist);

    while (std::getline(ss, line, '\n')) {
        lines.push_back(strdup(line.c_str()));
    }

    lines.push_back(nullptr);  // ngSpice_Circ wants a null-terminated array.
    // Parse the script. This return nonzero only in very limited circumstances, so we have to rely
    // on message parsing in ngspice.pyx to fully detect errors.
    status = m_ngSpice_Circ(lines.data()) == 0;

    // The ngSpice_Circ command requires a char** which necessitates strdup for each line. These get
    // allocated on the heap so need manually freed.
    for( auto line : lines ) {
        free(line);
    }

    return status;
}

bool NgspiceSession::run() {
    return command("run");
}

bool NgspiceSession::run_async() {
    bool success = start();

    if (success) {
        // Wait for end of simulation.
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } while(running());
    }

    return success;
}

bool NgspiceSession::start() {
    return command("bg_run");
}

bool NgspiceSession::stop() {
    return command("bg_halt");
}

bool NgspiceSession::running() {
    return m_ngSpice_Running();
}

bool NgspiceSession::command(const string& command) {
    validate();
    return m_ngSpice_Command((char*) command.c_str()) == 0;
}

std::vector<PlotInfo> NgspiceSession::plots() {
    std::vector<PlotInfo> plots;

    for (const auto& name_and_plot : plot_info_map) {
        plots.push_back(name_and_plot.second);
    }

    return plots;
}

PlotInfo& NgspiceSession::plot(const std::string& plot_type) {
    return plot_info_map.at(plot_type);
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

void NgspiceSession::emit_message(std::string message) {
    // Prefix search terms.
    std::string stderr_prefix("stderr ");
    std::string mif_error_prefix("stdout MIF-ERROR - ");

    if (!message.compare(0, stderr_prefix.size(), stderr_prefix)) {
        // Error on stderr.
        m_error = true;
        throw std::runtime_error(message.substr(stderr_prefix.size()));
    } else if (!message.compare(0, mif_error_prefix.size(), mif_error_prefix)) {
        // MIF-ERROR on stdout.
        m_error = true;
        throw std::runtime_error(message.substr(mif_error_prefix.size()));
    } else {
        // Forward the message to the registered handler.
        (*message_handler)(message);
    }
}

int NgspiceSession::cb_send_char(char* aWhat, int aId, void* aUser) {
    NgspiceSession* sim = reinterpret_cast<NgspiceSession*>(aUser);
    sim->emit_message((std::string) aWhat);
    return 0;
}

int NgspiceSession::cb_send_status(char* aWhat, int aId, void* aUser) {
    NgspiceSession* sim = reinterpret_cast<NgspiceSession*>(aUser);
    sim->emit_message((std::string) aWhat);
    return 0;
}

int NgspiceSession::cb_background_thread_running(bool aFinished, int aId, void* aUser) {
    return 0;
}

int NgspiceSession::cb_controlled_exit(int aStatus, bool aImmediate, bool aExitOnQuit, int aId, void* aUser) {
    // Set the error flag, which will force a reload of the DLL.
    NgspiceSession* sim = reinterpret_cast<NgspiceSession*>(aUser);
    sim->m_error = true;

    std::ostringstream ss;
    ss << "Ngspice exiting with status " << aStatus;
    sim->emit_message(ss.str());

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
