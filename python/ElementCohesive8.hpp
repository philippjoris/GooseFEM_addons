/**
 * @file
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#ifndef PYGOOSEFEM_ELEMENTCOHESIVE8_H
#define PYGOOSEFEM_ELEMENTCOHESIVE8_H

// Include the new 3D cohesive element header
#include <GooseFEM/ElementCohesive8.h> // Assuming Czm8 is in ElementCohesive8.h
#include <pybind11/pybind11.h>
#include <xtensor-python/pytensor.hpp>

#include "Element.hpp" // Assuming this provides register_Mesh_QuadratureBase

namespace py = pybind11;

void init_ElementCohesive8(py::module& m)
{
    // Bind the Czm8::Quadrature class
    py::class_<GooseFEM::Element::Czm8::Quadrature> cls(m, "Quadrature");

    cls.def_property_readonly("shape_rotmatrix",
        &GooseFEM::Element::Czm8::Quadrature::shape_rotmatrix,
        "Shape of rotation matrix");

    // Constructor with nodal coordinates only
    cls.def(
        py::init<const xt::pytensor<double, 3>&>(),
        "See :cpp:class:`GooseFEM::Element::Czm8::Quadrature`.",
        py::arg("x")
    );

    // Constructor with custom integration points and weights
    cls.def(
        py::init<
            const xt::pytensor<double, 3>&,
            const xt::pytensor<double, 2>&, // xi is 2D for 3D element (nip, ndimxi=2)
            const xt::pytensor<double, 1>&>(),
        "See :cpp:class:`GooseFEM::Element::Czm8::Quadrature`.",
        py::arg("x"),
        py::arg("xi"),
        py::arg("w")
    );

    // Register base class methods (e.g., shape_qscalar, shape_qvector, etc.)
    register_Mesh_QuadratureBase<GooseFEM::Element::Czm8::Quadrature>(cls);

    // Representation for Python
    cls.def("__repr__", [](const GooseFEM::Element::Czm8::Quadrature&) {
        return "<GooseFEM.Element.Czm8.Quadrature>";
    });

    // Binding for relative_disp method
    // elem_u: [#nelem, #nne, #ndim=3] -> pytensor<double, 3>
    // q_delta_u_norm_tan: [#nelem, #nip, #ndim=3] -> pytensor<double, 3>
    // rot_mat: [#nelem, #nip, #ndim=3, #ndim=3] -> pytensor<double, 4>
    cls.def(
        "relative_disp",
        &GooseFEM::Element::Czm8::Quadrature::relative_disp<
        xt::pytensor<double, 3>, xt::pytensor<double, 3>, xt::pytensor<double, 4>>,
        "Calculate relative displacement (normal and two tangential components).",
        py::arg("elem_u"),
        py::arg("q_delta_u"),
        py::arg("rotation_matrix")
    );

    // Binding for int_N_dot_traction_dA method (in-place)
    // q_tractions: [#nelem, #nip, #ndim=3] -> pytensor<double, 3>
    // elem_f: [#nelem, #nne, #ndim=3] -> pytensor<double, 3>
    cls.def(
        "int_N_dot_traction_dA",
        &GooseFEM::Element::Czm8::Quadrature::int_N_dot_traction_dA<
        xt::pytensor<double, 3>, xt::pytensor<double, 3>>,
        "Integral traction vector over element area (in-place).",
        py::arg("q_tractions"),
        py::arg("elem_f")
    );

    // Binding for Int_N_dot_traction_dA method (returns new tensor)
    // q_tractions: [#nelem, #nip, #ndim=3] -> pytensor<double, 3>
    cls.def(
        "Int_N_dot_traction_dA",
        &GooseFEM::Element::Czm8::Quadrature::Int_N_dot_traction_dA<
        xt::pytensor<double, 3>>,
        "Integral traction vector over element area (returns new tensor).",
        py::arg("q_tractions")
    );

    // Binding for int_BT_D_B_dA method (in-place)
    // q_tangent_stiffness_global: [#nelem, #nip, #ndim=3, #ndim=3] -> pytensor<double, 4>
    // elem_K: [#nelem, #nne*#ndim=24, #nne*#ndim=24] -> pytensor<double, 3>
    cls.def(
        "int_BT_D_B_dA",
        &GooseFEM::Element::Czm8::Quadrature::int_BT_D_B_dA<
        xt::pytensor<double, 4>, xt::pytensor<double, 3>>,
        "Calculate element stiffness matrix (in-place).",
        py::arg("q_tangent_stiffness_global"),
        py::arg("elem_K")
    );

    // Binding for Int_BT_D_B_dA method (returns new tensor)
    // q_tangent_stiffness_global: [#nelem, #nip, #ndim=3, #ndim=3] -> pytensor<double, 4>
    cls.def(
        "Int_BT_D_B_dA",
        &GooseFEM::Element::Czm8::Quadrature::Int_BT_D_B_dA<
        xt::pytensor<double, 4>>,
        "Calculate element stiffness matrix (returns new tensor).",
        py::arg("q_tangent_stiffness_global")
    );

    // Binding for update_x method
    // x: [#nelem, #nne, #ndim=3] -> pytensor<double, 3>
    cls.def("update_x", &GooseFEM::Element::Czm8::Quadrature::update_x<xt::pytensor<double, 3>>,
        "Update global coordinates of element class.",
        py::arg("x"));
}

void init_ElementCohesive8Gauss(py::module& m)
{
    // Bind Gauss quadrature functions for Czm8
    m.def("nip", &GooseFEM::Element::Czm8::Gauss::nip);
    m.def("xi", &GooseFEM::Element::Czm8::Gauss::xi);
    m.def("w", &GooseFEM::Element::Czm8::Gauss::w);
}

#endif // PYGOOSEFEM_ELEMENTCOHESIVE8_H