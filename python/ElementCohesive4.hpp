/**
 * @file
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#ifndef PYGOOSEFEM_ELEMENTCOHESIVE4_H
#define PYGOOSEFEM_ELEMENTCOHESIVE4_H

#include <GooseFEM/ElementCohesive4.h>
#include <pybind11/pybind11.h>
#include <xtensor-python/pytensor.hpp>

#include "Element.hpp"

namespace py = pybind11;

void init_ElementCohesive4(py::module& m)
{
    py::class_<GooseFEM::Element::Czm4::Quadrature> cls(m, "Quadrature");

    cls.def_property_readonly("shape_rotmatrix",
        &GooseFEM::Element::Czm4::Quadrature::shape_rotmatrix,
        "Shape of rotation matrix");

    cls.def(
        py::init<const xt::pytensor<double, 3>&>(),
        "See :cpp:class:`GooseFEM::Element::Czm4::Quadrature`.",
        py::arg("x")
    );

    cls.def(
        py::init<
            const xt::pytensor<double, 3>&,
            const xt::pytensor<double, 2>&,
            const xt::pytensor<double, 1>&>(),
        "See :cpp:class:`GooseFEM::Element::Czm4::Quadrature`.",
        py::arg("x"),
        py::arg("xi"),
        py::arg("w")
    );

    register_Mesh_QuadratureBase<GooseFEM::Element::Czm4::Quadrature>(cls);


    cls.def("__repr__", [](const GooseFEM::Element::Czm4::Quadrature&) {
        return "<GooseFEM.Element.Czm4.Quadrature>";
    });

    cls.def(
        "relative_disp",
        &GooseFEM::Element::Czm4::Quadrature::relative_disp<
        xt::pytensor<double, 3>, xt::pytensor<double, 3>, xt::pytensor<double, 4>>,
        "Calculate relative displacement.",        
        py::arg("elem_u"),
        py::arg("q_delta_u"),
        py::arg("rotation_matrix")
    );

    cls.def(
        "int_N_dot_traction_dL",
        &GooseFEM::Element::Czm4::Quadrature::int_N_dot_traction_dL<
        xt::pytensor<double, 3>, xt::pytensor<double, 3>>,
        "Integral traction vector and line.",
        py::arg("q_tractions"),
        py::arg("elem_f")
    );

    cls.def(
        "Int_N_dot_traction_dL",
        &GooseFEM::Element::Czm4::Quadrature::Int_N_dot_traction_dL<
        xt::pytensor<double, 3>>,
        "Integral traction vector and line.",
        py::arg("q_tractions")
    );

    cls.def(
        "int_BT_D_B_dL",
        &GooseFEM::Element::Czm4::Quadrature::int_BT_D_B_dL<
        xt::pytensor<double, 4>, xt::pytensor<double, 4>>,
        "Calculate element stiffness matrix.",
        py::arg("q_tangent_stiffness_global"),
        py::arg("elem_K")
    );    

    cls.def(
        "Int_BT_D_B_dL",
        &GooseFEM::Element::Czm4::Quadrature::Int_BT_D_B_dL<
        xt::pytensor<double, 4>>,
        "Calculate element stiffness matrix.",
        py::arg("q_tangent_stiffness_global")
    );

    cls.def("update_x", &GooseFEM::Element::Czm4::Quadrature::update_x<xt::pytensor<double, 3>>,
        "Update global coordinates of element class.",
        py::arg("x"));
}

void init_ElementCohesive4Gauss(py::module& m)
{
    m.def("nip", &GooseFEM::Element::Czm4::Gauss::nip);
    m.def("xi", &GooseFEM::Element::Czm4::Gauss::xi);
    m.def("w", &GooseFEM::Element::Czm4::Gauss::w);
}

#endif
