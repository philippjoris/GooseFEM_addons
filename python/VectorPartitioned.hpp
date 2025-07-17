/**
 * @file
 * @copyright Copyright 2017. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#include <GooseFEM/GooseFEM.h>
#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

void init_VectorPartitioned(py::module& m)
{

    py::class_<GooseFEM::VectorPartitioned, GooseFEM::Vector>(m, "VectorPartitioned")

        .def(
            py::init<
                const xt::pytensor<size_t, 2>&,
                const xt::pytensor<size_t, 1>&>(),
            "See :cpp:class:`GooseFEM::VectorPartitioned`.",
            py::arg("dofs"),
            py::arg("iip")
        )

        .def_property_readonly("nnu", &GooseFEM::VectorPartitioned::nnu)
        .def_property_readonly("nnp", &GooseFEM::VectorPartitioned::nnp)
        .def_property_readonly("iiu", &GooseFEM::VectorPartitioned::iiu)
        .def_property_readonly("iip", &GooseFEM::VectorPartitioned::iip)
        .def_property_readonly("dofs_is_u", &GooseFEM::VectorPartitioned::dofs_is_u)
        .def_property_readonly("dofs_is_p", &GooseFEM::VectorPartitioned::dofs_is_p)

        .def(
            "DofsFromParitioned",
            py::overload_cast<const xt::pytensor<double, 1>&, const xt::pytensor<double, 1>&>(
                &GooseFEM::VectorPartitioned::DofsFromPartitioned, py::const_
            ),
            py::arg("dofval_u"),
            py::arg("dofval_p")
        )

        .def("AsDofs_u",
            (xt::pytensor<double, 1> (GooseFEM::VectorPartitioned::*)(const xt::pytensor<double, 1>&) const)
            &GooseFEM::VectorPartitioned::AsDofs_u,
            "Converts 'dofval' to 'dofval_u' (unconstrained DOF values).",
            py::arg("dofval")
        )

        .def("AsDofs_u",
            (xt::pytensor<double, 1> (GooseFEM::VectorPartitioned::*)(const xt::pytensor<double, 2>&) const)
            &GooseFEM::VectorPartitioned::AsDofs_u,
            "Converts 'nodevec' to 'dofval_u' (unconstrained DOF values).",
            py::arg("nodevec")
        )

        .def("AsDofs_u",
            (xt::pytensor<double, 1> (GooseFEM::VectorPartitioned::*)(const xt::pytensor<double, 3>&, const xt::pytensor<size_t, 2>&) const)
            &GooseFEM::VectorPartitioned::AsDofs_u,
            "Converts 'elemvec' and 'conn' to 'dofval_u' (unconstrained DOF values).",
            py::arg("elemvec"),
            py::arg("conn")
        )

        .def(
            "asDofs_u",
            py::overload_cast<const xt::pytensor<double, 1>&, xt::pytensor<double, 1>&>(
                &GooseFEM::VectorPartitioned::asDofs_u, py::const_
            ),
            "Extracts unconstrained 'dofval' from a full 'dofval' (in-place).",
            py::arg("dofval"),
            py::arg("dofval_u")
        )

        .def(
            "asDofs_u",
            py::overload_cast<const xt::pytensor<double, 2>&, xt::pytensor<double, 1>&>(
                &GooseFEM::VectorPartitioned::asDofs_u, py::const_
            ),
            "Converts a 'nodevec' to unconstrained DOF values (in-place).",
            py::arg("nodevec"),
            py::arg("dofval_u")
        )

        .def(
            "asDofs_u",
            py::overload_cast<const xt::pytensor<double, 3>&, const xt::pytensor<size_t, 2>&, xt::pytensor<double, 1>&>(
                &GooseFEM::VectorPartitioned::asDofs_u, py::const_
            ),
            "Converts an 'elemvec' to unconstrained DOF values (in-place).",
            py::arg("elemvec"),
            py::arg("conn"),
            py::arg("dofval_u")
        )

        .def(
            "AsDofs_p",
            py::overload_cast<const xt::pytensor<double, 2>&>(
                &GooseFEM::VectorPartitioned::AsDofs_p, py::const_
            ),
            "Converts a 'nodevec' to prescribed DOF values.",
            py::arg("nodevec")
        )
        
        .def(
            "AsDofs_p",
            py::overload_cast<const xt::pytensor<double, 3>&, const xt::pytensor<size_t, 2>&>( 
                &GooseFEM::VectorPartitioned::AsDofs_p, py::const_
            ),
            "Converts an 'elemvec' to prescribed DOF values.",
            py::arg("elemvec"),
            py::arg("conn") 
        )
        
        .def(
            "asDofs_p", 
            py::overload_cast<const xt::pytensor<double, 1>&, xt::pytensor<double, 1>&>( 
                &GooseFEM::VectorPartitioned::asDofs_p, py::const_
            ),
            "Extracts prescribed 'dofval' from a full 'dofval' (in-place).",
            py::arg("dofval"),
            py::arg("dofval_p")
        )
        .def(
            "asDofs_p", 
            py::overload_cast<const xt::pytensor<double, 2>&, xt::pytensor<double, 1>&>( 
                &GooseFEM::VectorPartitioned::asDofs_p, py::const_
            ),
            "Converts a 'nodevec' to prescribed DOF values (in-place).",
            py::arg("nodevec"),
            py::arg("dofval_p")
        )
        .def(
            "asDofs_p", 
            py::overload_cast<const xt::pytensor<double, 3>&, const xt::pytensor<size_t, 2>&, xt::pytensor<double, 1>&>( 
                &GooseFEM::VectorPartitioned::asDofs_p, py::const_
            ),
            "Converts an 'elemvec' to prescribed DOF values (in-place).",
            py::arg("elemvec"),
            py::arg("conn"),
            py::arg("dofval_p")
        )

        .def(
            "NodeFromPartitioned",
            py::overload_cast<const xt::pytensor<double, 1>&, const xt::pytensor<double, 1>&>( 
                &GooseFEM::VectorPartitioned::NodeFromPartitioned, py::const_
            ),
            "Converts partitioned DOF values to nodal vectors.",
            py::arg("dofval_u"),
            py::arg("dofval_p")
        )

        .def(
            "nodeFromPartitioned", 
            &GooseFEM::VectorPartitioned::nodeFromPartitioned,
            "Converts partitioned DOF values to nodal vectors (in-place).",
            py::arg("dofval_u"),
            py::arg("dofval_p"),
            py::arg("nodevec")
        )

        .def(
            "ElementFromPartitioned",
            py::overload_cast<const xt::pytensor<double, 1>&, const xt::pytensor<double, 1>&, const xt::pytensor<size_t, 2>&>( 
                &GooseFEM::VectorPartitioned::ElementFromPartitioned, py::const_
            ),
            "Converts partitioned DOF values to element vectors.",
            py::arg("dofval_u"),
            py::arg("dofval_p"),
            py::arg("conn")
        )

        .def(
            "elementFromPartitioned", 
            &GooseFEM::VectorPartitioned::elementFromPartitioned,
            "Converts partitioned DOF values to element vectors (in-place).",
            py::arg("dofval_u"),
            py::arg("dofval_p"),
            py::arg("conn"),
            py::arg("elemvec")
        )

        .def(
            "Copy_u",
            &GooseFEM::VectorPartitioned::Copy_u,
            "Copies unconstrained part of 'nodevec_src' to 'nodevec_dest'.",
            py::arg("nodevec_src"),
            py::arg("nodevec_dest")
        )
        .def(
            "Copy_p",
            &GooseFEM::VectorPartitioned::Copy_p,
            "Copies prescribed part of 'nodevec_src' to 'nodevec_dest'.",
            py::arg("nodevec_src"),
            py::arg("nodevec_dest")
        )
        .def(
            "copy_u",
            &GooseFEM::VectorPartitioned::copy_u,
            "Copies unconstrained part of 'nodevec_src' to 'nodevec_dest' in-place.",
            py::arg("nodevec_src"),
            py::arg("nodevec_dest")
        )
        .def(
            "copy_p",
            &GooseFEM::VectorPartitioned::copy_p,
            "Copies prescribed part of 'nodevec_src' to 'nodevec_dest' in-place.",
            py::arg("nodevec_src"),
            py::arg("nodevec_dest")
        )

        .def("__repr__", [](const GooseFEM::VectorPartitioned&) {
            return "<GooseFEM.VectorPartitioned>";
        });
}
