/**
 * @file pyGooseFEM_MeshCohesiveHex8.h
 * @copyright Copyright 2025. All rights reserved.
 *
 * Pybind11 bindings for GooseFEM cohesive mesh classes in the Hex8 namespace.
 */

#ifndef PYGOOSEFEM_MESHCOHESIVE_HEX8_H
#define PYGOOSEFEM_MESHCOHESIVE_HEX8_H

#include <GooseFEM/MeshCohesiveHex8.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

#include "MeshCohesive.hpp"

namespace py = pybind11;

void init_MeshCohesiveHex8(py::module& m)
{
    {
        py::class_<GooseFEM::MeshCohesive::Hex8::RegularCohesive> cls(m, "RegularCohesive");

        cls.def(
            py::init<size_t, size_t, size_t, size_t, double>(),
            "See :cpp:class:`GooseFEM::MeshCohesive::Hex8::RegularCohesive`.",
            py::arg("nelx"),
            py::arg("nely"),
            py::arg("nelz_lower"),
            py::arg("nelz_upper"),
            py::arg("h") = 1.0
        );

        register_Mesh_CohesiveBase3d<GooseFEM::MeshCohesive::Hex8::RegularCohesive, py::class_<GooseFEM::MeshCohesive::Hex8::RegularCohesive>>(cls);

        cls.def("__repr__", [](const GooseFEM::MeshCohesive::Hex8::RegularCohesive&) {
            return "<GooseFEM.MeshCohesive.Hex8.RegularCohesive>";
        });
    }
}

#endif // PYGOOSEFEM_MESHCOHESIVE_HEX8_H