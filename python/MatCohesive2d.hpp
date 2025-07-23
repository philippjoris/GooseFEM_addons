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
    using C_Bilinear = GooseFEM::ConstitutiveModels::Cartesian2d::CohesiveBilinear<2>;
    py::class_<C_Bilinear> cls(m, "CohesiveBilinear2d");

        cls.def(
            py::init<
                const xt::pytensor<double, 2>&,
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
            py::arg("beta"),
            py::arg("eta")
        );

        cls.def("__repr__", [](const C_Bilinear&) {
            return "<GooseFEM::ConstitutiveModels::Cartesian2d::Cohesive>";
        });

        cls.def_property_readonly("Kn", &C_Bilinear::Kn, "Normal stiffness coefficient.");
        cls.def_property_readonly("Kt", &C_Bilinear::Kt, "Tangential stiffness coefficient.");
        cls.def_property_readonly("T", &C_Bilinear::T, "Traction vector in global coordinates.");
        cls.def_property_readonly("T_local", &C_Bilinear::T_local, "Traction vector in local coordinates.");        
        cls.def_property_readonly("C", &C_Bilinear::C, "Tangential stiffness matrix in global coordinates.");
        cls.def_property_readonly("C_local", &C_Bilinear::C_local, "Tangential stiffness matrix in local coordinates.");        
        cls.def_property_readonly("delta0", &C_Bilinear::delta0, "Relative displ. onset of damage.");
        cls.def_property_readonly("Damage", &C_Bilinear::Damage, "Accumulated damage variable.");
        cls.def_property_readonly("failed", &C_Bilinear::failed, "Flag indicating if element has failed.");
        cls.def_property_readonly("delta_eff", &C_Bilinear::delta_eff, "Effective relative displacement.");
        cls.def_property_readonly("eta", &C_Bilinear::eta, "Viscosity parameter.");

        cls.def_property(
            "delta",
            static_cast<xt::pytensor<double, 3>& (C_Bilinear::*)()>(&C_Bilinear::delta),
            static_cast<void (C_Bilinear::*)(const xt::pytensor<double, 3>&)>(&C_Bilinear::set_delta),
            "Effective separation value."
        );

        cls.def_property(
            "ori",
            static_cast<xt::pytensor<double, 4>& (C_Bilinear::*)()>(&C_Bilinear::ori),
            static_cast<void (C_Bilinear::*)(const xt::pytensor<double, 4>&)>(&C_Bilinear::set_ori),
            "Effective separation value."
        );

        cls.def(
            "set_delta",
            py::overload_cast<const xt::pytensor<double, 3>&>(
                &C_Bilinear::set_delta<xt::pytensor<double, 3>>),
            "Overwrite deformation gradient tensor.",
            py::arg("arg")
        );

        cls.def(
            "set_ori",
            py::overload_cast<const xt::pytensor<double, 4>&>(
                &C_Bilinear::set_ori<xt::pytensor<double, 4>>),
            "Overwrite deformation gradient tensor.",
            py::arg("arg")
        );

        cls.def(
            "refresh", 
            py::overload_cast<double, bool, bool>(&C_Bilinear::refresh), 
            "Recompute traction from relative separation.", 
            py::arg("dt"),
            py::arg("compute_tangent") = true,
            py::arg("element_erosion") = true
        );

        cls.def("increment", &C_Bilinear::increment, "Update history variables.");

    using C_ExpGc = GooseFEM::ConstitutiveModels::Cartesian2d::CohesiveExponential<2>; 
    py::class_<C_ExpGc> cls_expgc(m, "CohesiveExponential2d"); 

        cls_expgc.def(
            py::init<
                const xt::pytensor<double, 2>&,  
                const xt::pytensor<double, 2>&,  
                const xt::pytensor<double, 2>&,  
                const xt::pytensor<double, 2>&,  
                const xt::pytensor<double, 2>&>(), 
            "Cohesive zone model with an exponential softening law, specified by critical energy release rate Gc.",
            py::arg("Kn"),
            py::arg("Kt"),
            py::arg("delta0"),
            py::arg("Gc"),
            py::arg("beta")
        );

        cls_expgc.def("__repr__", [](const C_ExpGc&) {
            return "<GooseFEM::ConstitutiveModels::Cartesian2d::CohesiveExponentialGc>";
        });

        cls_expgc.def_property_readonly("Kn", &C_ExpGc::Kn, "Normal stiffness coefficient.");
        cls_expgc.def_property_readonly("Kt", &C_ExpGc::Kt, "Tangential stiffness coefficient.");
        cls_expgc.def_property_readonly("delta0", &C_ExpGc::delta0, "Effective relative displacement at peak traction (damage initiation).");
        cls_expgc.def_property_readonly("Gc", &C_ExpGc::Gc, "Critical energy release rate.");
        cls_expgc.def_property_readonly("beta", &C_ExpGc::beta, "Weighting for tangential separation.");
        cls_expgc.def_property_readonly("delta_exp_char", &C_ExpGc::delta_exp_char, "Characteristic decay length for exponential softening.");

        cls_expgc.def_property_readonly("T", &C_ExpGc::T, "Traction vector in global coordinates.");
        cls_expgc.def_property_readonly("T_local", &C_ExpGc::T_local, "Traction vector in local coordinates.");         
        cls_expgc.def_property_readonly("C", &C_ExpGc::C, "Tangential stiffness matrix in global coordinates.");
        cls_expgc.def_property_readonly("C_local", &C_ExpGc::C_local, "Tangential stiffness matrix in local coordinates.");         
        cls_expgc.def_property_readonly("Damage", &C_ExpGc::Damage, "Accumulated damage variable.");
        cls_expgc.def_property_readonly("failed", &C_ExpGc::failed, "Flag indicating if element has failed.");
        cls_expgc.def_property_readonly("delta_eff", &C_ExpGc::delta_eff, "Effective relative displacement.");

        cls_expgc.def_property(
            "delta",
            static_cast<xt::pytensor<double, 3>& (C_ExpGc::*)()>(&C_ExpGc::delta),
            static_cast<void (C_ExpGc::*)(const xt::pytensor<double, 3>&)>(&C_ExpGc::set_delta),
            "Current relative displacement vector."
        );

        cls_expgc.def_property(
            "ori",
            static_cast<xt::pytensor<double, 4>& (C_ExpGc::*)()>(&C_ExpGc::ori),
            static_cast<void (C_ExpGc::*)(const xt::pytensor<double, 4>&)>(&C_ExpGc::set_ori),
            "Orientation matrix from local to global coordinates."
        );

        cls_expgc.def(
            "set_delta",
            py::overload_cast<const xt::pytensor<double, 3>&>(
                &C_ExpGc::set_delta<xt::pytensor<double, 3>>),
            "Overwrite relative displacement tensor.",
            py::arg("arg")
        );

        cls_expgc.def(
            "set_ori",
            py::overload_cast<const xt::pytensor<double, 4>&>(
                &C_ExpGc::set_ori<xt::pytensor<double, 4>>),
            "Overwrite orientation matrix.",
            py::arg("arg")
        );

        cls_expgc.def(
            "refresh", 
            py::overload_cast<bool, bool>(&C_ExpGc::refresh), 
            "Recompute traction from relative separation.", 
            py::arg("compute_tangent") = true,
            py::arg("element_erosion") = true
        );

        cls_expgc.def("increment", &C_ExpGc::increment, "Update history variables.");
}

#endif
