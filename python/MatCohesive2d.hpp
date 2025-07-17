/**
 * @file
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#ifndef PYGOOSEFEM_MATCOHESIVE2D_H
#define PYGOOSEFEM_MATCOHESIVE2D_H

#include <GooseFEM/ConstitutiveModels/MatCohesive2d.h>
#include <pybind11/pybind11.h>
#include <xtensor-python/pytensor.hpp>

#include "Element.hpp"

namespace py = pybind11;

void init_MatCohesive2d(py::module& m)
{
    using C = GooseFEM::ConstitutiveModels::Cartesian2d::Cohesive<2>;
    py::class_<C> cls(m, "Cohesive2d");

        cls.def(
            py::init<
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&>(),
            "See :cpp:class:`GooseFEM::ConstitutiveModels::Cartesian2d::Cohesive`.",
            py::arg("Kn"),
            py::arg("Kt"),
            py::arg("delta0"),
            py::arg("deltafrac"),
            py::arg("beta")
        );

        cls.def("__repr__", [](const C&) {
            return "<GooseFEM::ConstitutiveModels::Cartesian2d::Cohesive>";
        });

        cls.def_property_readonly("Kn", &C::Kn, "Normal stiffness coefficient.");
        cls.def_property_readonly("Kt", &C::Kt, "Tangential stiffness coefficient.");
        cls.def_property_readonly("T", &C::T, "Traction vector in global coordinates.");
        cls.def_property_readonly("T_local", &C::T_local, "Traction vector in local coordinates.");        
        cls.def_property_readonly("C", &C::C, "Tangential stiffness matrix in global coordinates.");
        cls.def_property_readonly("C_local", &C::C_local, "Tangential stiffness matrix in local coordinates.");        
        cls.def_property_readonly("delta0", &C::delta0, "Relative displ. onset of damage.");
        cls.def_property_readonly("Damage", &C::Damage, "Accumulated damage variable.");
        cls.def_property_readonly("failed", &C::failed, "Flag indicating if element has failed.");
        cls.def_property_readonly("delta_eff", &C::delta_eff, "Effective relative displacement.");

        cls.def_property(
            "delta",
            static_cast<xt::pytensor<double, 3>& (C::*)()>(&C::delta),
            static_cast<void (C::*)(const xt::pytensor<double, 3>&)>(&C::set_delta),
            "Effective separation value."
        );

        cls.def_property(
            "ori",
            static_cast<xt::pytensor<double, 4>& (C::*)()>(&C::ori),
            static_cast<void (C::*)(const xt::pytensor<double, 4>&)>(&C::set_ori),
            "Effective separation value."
        );

        cls.def(
            "set_delta",
            py::overload_cast<const xt::pytensor<double, 3>&>(
                &C::set_delta<xt::pytensor<double, 3>>),
            "Overwrite deformation gradient tensor.",
            py::arg("arg")
        );

        cls.def(
            "set_ori",
            py::overload_cast<const xt::pytensor<double, 4>&>(
                &C::set_ori<xt::pytensor<double, 4>>),
            "Overwrite deformation gradient tensor.",
            py::arg("arg")
        );

        cls.def(
            "refresh", 
            py::overload_cast<bool, bool>(&C::refresh), 
            "Recompute traction from relative separation.", 
            py::arg("compute_tangent") = true,
            py::arg("element_erosion") = true
        );

        cls.def("increment", &C::increment, "Update history variables.");
}

#endif
