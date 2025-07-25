/**
 * @file
 * @copyright Copyright 2017. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#include <Eigen/Eigen>
#include <GooseFEM/GooseFEM.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

#include "Matrix.hpp"

namespace py = pybind11;

void init_MatrixPartitionedTyings(py::module& m)
{

    py::class_<GooseFEM::MatrixPartitionedTyings> cls(m, "MatrixPartitionedTyings");
    register_Matrix_MatrixBase<GooseFEM::MatrixPartitionedTyings>(cls);
    register_Matrix_MatrixPartitionedBase<GooseFEM::MatrixPartitionedTyings>(cls);
    register_Matrix_MatrixPartitionedTyingsBase<GooseFEM::MatrixPartitionedTyings>(cls);

    cls.def(
        py::init<
            const xt::pytensor<size_t, 2>&,
            const Eigen::SparseMatrix<double>&,
            const Eigen::SparseMatrix<double>&>(),
        "See :cpp:class:`GooseFEM::MatrixPartitionedTyings`.",
        py::arg("dofs"),
        py::arg("Cdu"),
        py::arg("Cdp")
    );

    cls.def_property_readonly("data_uu", &GooseFEM::MatrixPartitionedTyings::data_uu);
    cls.def_property_readonly("data_up", &GooseFEM::MatrixPartitionedTyings::data_up);
    cls.def_property_readonly("data_pu", &GooseFEM::MatrixPartitionedTyings::data_pu);
    cls.def_property_readonly("data_pp", &GooseFEM::MatrixPartitionedTyings::data_pp);
    cls.def_property_readonly("data_ud", &GooseFEM::MatrixPartitionedTyings::data_ud);
    cls.def_property_readonly("data_pd", &GooseFEM::MatrixPartitionedTyings::data_pd);
    cls.def_property_readonly("data_du", &GooseFEM::MatrixPartitionedTyings::data_du);
    cls.def_property_readonly("data_dp", &GooseFEM::MatrixPartitionedTyings::data_dp);
    cls.def_property_readonly("data_dd", &GooseFEM::MatrixPartitionedTyings::data_dd);

    cls.def("__repr__", [](const GooseFEM::MatrixPartitionedTyings&) {
        return "<GooseFEM.MatrixPartitionedTyings>";
    });

    py::class_<GooseFEM::MatrixPartitionedTyingsSolver<>> slv(m, "MatrixPartitionedTyingsSolver");

    register_MatrixSolver_MatrixSolverBase<
        GooseFEM::MatrixPartitionedTyingsSolver<>,
        GooseFEM::MatrixPartitionedTyings>(slv);

    register_MatrixSolver_MatrixSolverPartitionedBase<
        GooseFEM::MatrixPartitionedTyingsSolver<>,
        GooseFEM::MatrixPartitionedTyings>(slv);

    slv.def(py::init<>(), "See :cpp:class:`GooseFEM::MatrixPartitionedTyingsSolver`.");

    slv.def(
        "Solve_u",
        py::overload_cast<
            GooseFEM::MatrixPartitionedTyings&,
            const xt::pytensor<double, 1>&,
            const xt::pytensor<double, 1>&,
            const xt::pytensor<double, 1>&>(&GooseFEM::MatrixPartitionedTyingsSolver<>::Solve_u),
        py::arg("matrix"),
        py::arg("b_u"),
        py::arg("b_d"),
        py::arg("x_p")
    );

    slv.def(
        "solve_u",
        py::overload_cast<
            GooseFEM::MatrixPartitionedTyings&,
            const xt::pytensor<double, 1>&,
            const xt::pytensor<double, 1>&,
            const xt::pytensor<double, 1>&,
            xt::pytensor<double, 1>&>(&GooseFEM::MatrixPartitionedTyingsSolver<>::solve_u),
        py::arg("matrix"),
        py::arg("b_u"),
        py::arg("b_d"),
        py::arg("x_p"),
        py::arg("x_u")
    );

    slv.def("__repr__", [](const GooseFEM::MatrixPartitionedTyingsSolver<>&) {
        return "<GooseFEM.MatrixPartitionedTyingsSolver>";
    });
}
