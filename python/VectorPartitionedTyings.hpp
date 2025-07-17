/**
 * @file
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#include <GooseFEM/GooseFEM.h>
#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;
using namespace GooseFEM; 

void init_VectorPartitionedTyings(py::module& m)
{
    py::class_<GooseFEM::VectorPartitionedTyings, GooseFEM::Vector>(m, "VectorPartitionedTyings")

        .def(
            py::init<
                const xt::pytensor<size_t, 2>&, 
                const Eigen::SparseMatrix<double>&,    
                const Eigen::SparseMatrix<double>&,    
                const Eigen::SparseMatrix<double>& 
            >(),
            "Constructor for a partitioned vector system with tyings.",
            py::arg("dofs"),
            py::arg("Cdu"),
            py::arg("Cdp"),
            py::arg("Cdi")
        )

        .def_property_readonly("nnu", &GooseFEM::VectorPartitionedTyings::nnu, "Number of independent unknown DOFs.")
        .def_property_readonly("nnp", &GooseFEM::VectorPartitionedTyings::nnp, "Number of independent prescribed DOFs.")
        .def_property_readonly("nni", &GooseFEM::VectorPartitionedTyings::nni, "Number of independent DOFs (unknown + prescribed).")
        .def_property_readonly("nnd", &GooseFEM::VectorPartitionedTyings::nnd, "Number of dependent DOFs.")
        .def_property_readonly("iiu", &GooseFEM::VectorPartitionedTyings::iiu, "Indices of independent unknown DOFs.")
        .def_property_readonly("iip", &GooseFEM::VectorPartitionedTyings::iip, "Indices of independent prescribed DOFs.")
        .def_property_readonly("iii", &GooseFEM::VectorPartitionedTyings::iii, "Indices of all independent DOFs.")
        .def_property_readonly("iid", &GooseFEM::VectorPartitionedTyings::iid, "Indices of all dependent DOFs.")

        .def(
            "copy_p",
            (void (GooseFEM::VectorPartitionedTyings::*)(
                const xt::pytensor<double, 1>&,
                xt::pytensor<double, 1>&
            ) const) &GooseFEM::VectorPartitionedTyings::copy_p,
            "Copies prescribed DOFs from 'dofval_src' to 'dofval_dest' in-place.",
            py::arg("dofval_src"),
            py::arg("dofval_dest")
        )

        .def(
            "asDofs_i",
            (void (GooseFEM::VectorPartitionedTyings::*)(
                const xt::pytensor<double, 2>&,
                xt::pytensor<double, 1>&,
                bool
            ) const) &GooseFEM::VectorPartitionedTyings::asDofs_i<xt::pytensor<double, 2>, xt::pytensor<double, 1>>,
            "Converts 'nodevec' to independent DOF values (in-place).",
            py::arg("nodevec"),
            py::arg("dofval_i"),
            py::arg("apply_tyings") = true
        )

        .def(
            "AsDofs_i",
            (xt::pytensor<double, 1> (GooseFEM::VectorPartitionedTyings::*)(
                const xt::pytensor<double, 2>&) const)
            &GooseFEM::VectorPartitionedTyings::AsDofs_i<xt::pytensor<double, 2>>,
            "Converts 'nodevec' to independent DOF values (in-place).", 
            py::arg("nodevec")
        )

        .def("__repr__", [](const GooseFEM::VectorPartitionedTyings&) {
            return "<GooseFEM.VectorPartitionedTyings>";
        });
}