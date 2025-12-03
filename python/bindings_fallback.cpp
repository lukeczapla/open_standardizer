#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <GraphMol/ROMol.h>
#include <RDGeneral/Exceptions.h>

#include "fallback_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rdkit_fallback_engine, m) {
    m.doc() = "GPU→CPU fallback engine for RDKit Standardizer";

    py::class_<FallbackEngine>(m, "FallbackEngine")
        .def(py::init<>())
        .def("standardize", &FallbackEngine::standardize);
}

