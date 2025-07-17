/**
 * @file
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 *
 */

#include <GooseFEM/Vector.h>
#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

void init_Vector(py::module& m)
{

    py::class_<GooseFEM::Vector>(m, "Vector")

        .def(
            py::init<const xt::pytensor<size_t, 2>&>(),
            "See :cpp:class:`GooseFEM::Vector`.",
            py::arg("dofs")
        )

        .def_property_readonly("nnode", &GooseFEM::Vector::nnode)
        .def_property_readonly("ndim", &GooseFEM::Vector::ndim)
        .def_property_readonly("ndof", &GooseFEM::Vector::ndof)
        .def_property_readonly("dofs", &GooseFEM::Vector::dofs)

        .def(
            "copy",
            &GooseFEM::Vector::copy<xt::pyarray<double>>,
            py::arg("nodevec_src"),
            py::arg("nodevec_dest")
        )

        .def(
            "Copy",
            &GooseFEM::Vector::Copy<xt::pyarray<double>>,
            py::arg("nodevec_src"),
            py::arg("nodevec_dest")
        )

        .def("AsDofs", &GooseFEM::Vector::AsDofs<xt::pyarray<double>>, py::arg("arg"))

        .def(
            "asDofs",
            &GooseFEM::Vector::asDofs<xt::pyarray<double>, xt::pytensor<double, 1>>,
            py::arg("arg"),
            py::arg("ret")
        )

        .def("AsDofs", &GooseFEM::Vector::AsDofs<xt::pyarray<double>, xt::pytensor<size_t, 2>>,
            py::arg("arg"),
            py::arg("conn")
        )

        .def(
            "asDofs",
            &GooseFEM::Vector::asDofs<xt::pyarray<double>, xt::pytensor<size_t, 2>, xt::pytensor<double, 1>>,
            py::arg("arg"),
            py::arg("conn"),
            py::arg("ret")
        )        

        .def("AsNode", &GooseFEM::Vector::AsNode<xt::pyarray<double>, xt::pytensor<size_t, 2>>,
            py::arg("arg"),
            py::arg("conn")
        )

        .def(
            "asNode",
            &GooseFEM::Vector::asNode<xt::pyarray<double>, xt::pytensor<size_t, 2>, xt::pytensor<double, 2>>,
            py::arg("arg"),
            py::arg("conn"),
            py::arg("ret")
        )

        .def("AsNode", &GooseFEM::Vector::AsNode<xt::pyarray<double>>, py::arg("arg"))

        .def(
            "asNode",
            &GooseFEM::Vector::asNode<xt::pyarray<double>, xt::pytensor<double, 2>>,
            py::arg("arg"),
            py::arg("ret")
        )

        .def("AsElement", &GooseFEM::Vector::AsElement<xt::pyarray<double>, xt::pytensor<double, 2>>,
            py::arg("arg"),
            py::arg("conn")
        )

        .def(
            "asElement",
            &GooseFEM::Vector::asElement<xt::pyarray<double>, xt::pytensor<double, 2>, xt::pytensor<double, 3>>,
            py::arg("arg"),
            py::arg("conn"),
            py::arg("ret")
        )

        .def("AssembleDofs", &GooseFEM::Vector::AssembleDofs<xt::pyarray<double>>, py::arg("arg"))

        .def(
            "assembleDofs",
            &GooseFEM::Vector::assembleDofs<xt::pyarray<double>, xt::pytensor<double, 1>>,
            py::arg("arg"),
            py::arg("ret")
        )

        .def("AssembleDofs", &GooseFEM::Vector::AssembleDofs<xt::pyarray<double>, xt::pytensor<double, 2>>,
            py::arg("arg"),
            py::arg("conn")
        )

        .def(
            "assembleDofs",
            &GooseFEM::Vector::assembleDofs<xt::pyarray<double>, xt::pytensor<double, 2>, xt::pytensor<double, 1>>,
            py::arg("arg"),
            py::arg("conn"),
            py::arg("ret")
        )

        .def("AssembleNode", &GooseFEM::Vector::AssembleNode<xt::pyarray<double>, xt::pytensor<double, 2>>,
            py::arg("arg"),
            py::arg("conn")
        )

        .def(
            "assembleNode",
            &GooseFEM::Vector::assembleNode<xt::pyarray<double>, xt::pytensor<double, 2>, xt::pytensor<double, 2>>,
            py::arg("arg"),
            py::arg("conn"),
            py::arg("ret")
        )

        .def_property_readonly("shape_dofval", &GooseFEM::Vector::shape_dofval)
        .def_property_readonly("shape_nodevec", &GooseFEM::Vector::shape_nodevec)

        .def("shape_elemvec", &GooseFEM::Vector::shape_elemvec,
            py::arg("nelem"),
            py::arg("nne")
        )
        .def("shape_elemmat", &GooseFEM::Vector::shape_elemmat,
            py::arg("nelem"),
            py::arg("nne")
        )

        .def(
            "allocate_dofval",
            (xt::pytensor<double, 1> (GooseFEM::Vector::*)() const) 
            &GooseFEM::Vector::allocate_dofval,
            "Allocates an empty 'dofval' (1D array of size #ndof)."
        )

        .def(
            "allocate_dofval",
            (xt::pytensor<double, 1> (GooseFEM::Vector::*)(double) const) 
            &GooseFEM::Vector::allocate_dofval,
            "Allocates and initializes a 'dofval' (1D array of size #ndof).",
            py::arg("val")
        )

        .def(
            "allocate_nodevec",
            (xt::pytensor<double, 2> (GooseFEM::Vector::*)() const)
            &GooseFEM::Vector::allocate_nodevec,
            "Allocates an empty 'nodevec' (2D array of size #nnode, #ndim)."
        )

        .def(
            "allocate_nodevec",
            (xt::pytensor<double, 2> (GooseFEM::Vector::*)(double) const) 
            &GooseFEM::Vector::allocate_nodevec,
            "Allocates and initializes a 'nodevec' (2D array of size #nnode, #ndim).",
            py::arg("val")
        )   

        .def("allocate_elemvec",
            (xt::pytensor<double, 3> (GooseFEM::Vector::*)(size_t, size_t) const) &GooseFEM::Vector::allocate_elemvec,
            "Allocates an empty 'elemvec' (3D array of size [#nelem, #nne, #ndim]).",
            py::arg("nelem"),
            py::arg("nne")
        )

        .def("allocate_elemvec",
            (xt::pytensor<double, 3> (GooseFEM::Vector::*)(double, size_t, size_t) const) &GooseFEM::Vector::allocate_elemvec,
            "Allocates and initializes an 'elemvec' (3D array of size [#nelem, #nne, #ndim]).",
            py::arg("val"),
            py::arg("nelem"),
            py::arg("nne")
        )

        .def("allocate_elemmat",
            (xt::pytensor<double, 3> (GooseFEM::Vector::*)(size_t, size_t) const) &GooseFEM::Vector::allocate_elemmat,
            "Allocates an empty 'elemmat' (3D array of size [#nelem, #nne * #ndim, #nne * #ndim]).",
            py::arg("nelem"),
            py::arg("nne")
        )

        .def("allocate_elemmat",
            (xt::pytensor<double, 3> (GooseFEM::Vector::*)(double, size_t, size_t) const) &GooseFEM::Vector::allocate_elemmat,
            "Allocates and initializes an 'elemmat' (3D array of size [#nelem, #nne * #ndim, #nne * #ndim]).",
            py::arg("val"),
            py::arg("nelem"),
            py::arg("nne")
        )

        .def("__repr__", [](const GooseFEM::Vector&) { return "<GooseFEM.Vector>"; });
}
