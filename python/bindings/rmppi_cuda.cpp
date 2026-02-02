#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(rmppi_cuda, m) {
    m.doc() = "R-MPPI CUDA bindings (smoke test)";
    m.def("hello", [](){ return "rmppi_cuda loaded"; });
}
