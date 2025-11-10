#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SolverPartitionedTyings_PETSc.h"

namespace py = pybind11;
using namespace GooseFEM;

void init_SolverPartitionedTyings_PETSc(py::module& m)
{
    py::class_<SolverPartitionedTyings_PETSc>(m, "SolverPartitionedTyings_PETSc")
        .def(py::init<>())
        .def("factorize", &SolverPartitionedTyings_PETSc::factorize,
             py::arg("A"),
             R"pbdoc(
             Factorize the given partitioned matrix system.
             
             Builds the condensed matrices A'_uu and A'_up and sets up the PETSc KSP solver.
             This must be called before `solve` or `solve_u`.
             )pbdoc")
        
        .def("solve", &SolverPartitionedTyings_PETSc::solve<Eigen::VectorXd>,
             py::arg("A"), py::arg("b"), py::arg("x"),
             R"pbdoc(
             Solve the full partitioned system:
             
                 [A'_uu  A'_up] [x_u] = [b'_u]
                 [C_du   C_dp ] [x_p]   [b_d ]

             Reconstructs the dependent DOFs internally.
             )pbdoc")

        .def("solve_u", &SolverPartitionedTyings_PETSc::solve_u,
             py::arg("A"),
             py::arg("b_u"),
             py::arg("b_d"),
             py::arg("x_p"),
             py::arg("x_u"),
             R"pbdoc(
             Solve the condensed subproblem for primary DOFs (x_u) only:

                 A'_uu * x_u = b'_u - A'_up * x_p

             Inputs:
                 b_u, b_d, x_p : tensors or arrays.
             Output:
                 x_u : updated with the computed result.
             )pbdoc")

        .def("setTolerance", &SolverPartitionedTyings_PETSc::setTolerance,
             py::arg("tol"),
             R"pbdoc(Set the relative solver tolerance for the PETSc KSP solver.)pbdoc")

        .def("setMaxIterations", &SolverPartitionedTyings_PETSc::setMaxIterations,
             py::arg("maxIter"),
             R"pbdoc(Set the maximum number of iterations for the PETSc KSP solver.)pbdoc");
}