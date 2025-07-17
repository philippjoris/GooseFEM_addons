/**
 * @file
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 *
 */

#ifndef PYGOOSEFEM_MESHCOHESIVE_QUAD4_H
#define PYGOOSEFEM_MESHCOHESIVE_QUAD4_H

#include <GooseFEM/MeshCohesiveQuad4.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

#include "MeshCohesive.hpp"

namespace py = pybind11;

void init_MeshCohesiveQuad4(py::module& m)
{
    {
        py::class_<GooseFEM::MeshCohesive::Quad4::RegularCohesive> cls(m, "RegularCohesive");

        cls.def(
            py::init<size_t, size_t, size_t, double>(),
            "See :cpp:class:`GooseFEM::Mesh::Quad4::RegularCohesive`.",
            py::arg("nelx"),
            py::arg("nely_lower"),
            py::arg("nely_upper"),
            py::arg("h") = 1.0
        );

        register_Mesh_CohesiveBase2d<GooseFEM::MeshCohesive::Quad4::RegularCohesive, py::class_<GooseFEM::MeshCohesive::Quad4::RegularCohesive>>(cls);

        cls.def("__repr__", [](const GooseFEM::MeshCohesive::Quad4::RegularCohesive&) {
            return "<GooseFEM.Mesh.Quad4.RegularCohesive>";
        });
    }
}

#endif