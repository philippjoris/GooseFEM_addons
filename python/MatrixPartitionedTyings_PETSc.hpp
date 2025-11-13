#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Sparse>
#include <petscmat.h>
#include "GooseFEM/MatrixPartitionedTyings_PETSc.h"

namespace py = pybind11;

void init_MatrixPartitionedTyings_PETSc(py::module& m)
{
    py::class_<GooseFEM::MatrixPartitionedTyings_PETSc>(m, "MatrixPartitionedTyings_PETSc")
        // constructor
        .def(py::init<
             const xt::pytensor<size_t, 2>&,
             const Eigen::SparseMatrix<double>&,
             const Eigen::SparseMatrix<double>&>(),
             py::arg("dofs"), py::arg("Cdu"), py::arg("Cdp"),
             "Create a PETSc-backed partitioned matrix.")

        // properties
        .def_property_readonly("nnode", &GooseFEM::MatrixPartitionedTyings_PETSc::nnode)
        .def_property_readonly("ndim",  &GooseFEM::MatrixPartitionedTyings_PETSc::ndim)
        .def_property_readonly("nnu",   &GooseFEM::MatrixPartitionedTyings_PETSc::nnu)
        .def_property_readonly("nnp",   &GooseFEM::MatrixPartitionedTyings_PETSc::nnp)
        .def_property_readonly("nnd",   &GooseFEM::MatrixPartitionedTyings_PETSc::nnd)
        .def_property_readonly("ndof",  &GooseFEM::MatrixPartitionedTyings_PETSc::ndof)

        // matrix access
        .def("data_A", [](const GooseFEM::MatrixPartitionedTyings_PETSc &self){
            return py::capsule(self.data_A());
        })
        .def("data_Cdu", [](const GooseFEM::MatrixPartitionedTyings_PETSc &self){
            return py::capsule(self.data_Cdu());
        })
        .def("data_Cdp", [](const GooseFEM::MatrixPartitionedTyings_PETSc &self){
            return py::capsule(self.data_Cdp());
        })
        .def("data_ACuu", [](const GooseFEM::MatrixPartitionedTyings_PETSc &self){
            return py::capsule(self.data_ACuu());
        })

        // public methods
        .def("clear", &GooseFEM::MatrixPartitionedTyings_PETSc::clear,
             "Zero the global PETSc matrix entries.")
        .def("finalize",
            [](GooseFEM::MatrixPartitionedTyings_PETSc &self, bool stabilize=true) {
                return self.finalize(stabilize);
            },
            py::arg("stabilize")=true)
        .def("assemble",
            [](GooseFEM::MatrixPartitionedTyings_PETSc &self,
                const xt::pytensor<double, 3> &elemmat,
                const xt::pytensor<size_t, 2> &conn_elem) {
                self.assemble(elemmat, conn_elem);
            },
            py::arg("elemmat"), py::arg("conn_elem"),
            "Assemble element matrices into the global PETSc matrix.")

        // scatter_solution
        .def("scatter_solution",
            [](GooseFEM::MatrixPartitionedTyings_PETSc &A,
               py::array_t<double, py::array::c_style | py::array::forcecast> Xu_arr,
               py::array_t<double, py::array::c_style | py::array::forcecast> Xd_arr,
               py::array_t<double, py::array::c_style | py::array::forcecast> x_global_arr) {

                py::buffer_info bu = Xu_arr.request();
                py::buffer_info bd = Xd_arr.request();
                py::buffer_info bg = x_global_arr.request();

                if (bu.ndim != 1 || (size_t)bu.shape[0] != A.nnu()) throw std::runtime_error("Xu length mismatch");
                if (bd.ndim != 1 || (size_t)bd.shape[0] != A.nnd()) throw std::runtime_error("Xd length mismatch");
                if (bg.ndim != 1 || (size_t)bg.shape[0] != A.ndof()) throw std::runtime_error("x_global length mismatch");

                Vec Xu, Xd, xg;
                PetscErrorCode ierr;
                ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, (PetscInt)A.nnu(), static_cast<const PetscScalar*>(bu.ptr), &Xu);
                if (ierr) throw std::runtime_error("VecCreateSeqWithArray(Xu) failed");
                ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, (PetscInt)A.nnd(), static_cast<const PetscScalar*>(bd.ptr), &Xd);
                if (ierr) { VecDestroy(&Xu); throw std::runtime_error("VecCreateSeqWithArray(Xd) failed"); }
                ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, PETSC_DECIDE, (PetscInt)A.ndof(), static_cast<const PetscScalar*>(bg.ptr), &xg);
                if (ierr) { VecDestroy(&Xu); VecDestroy(&Xd); throw std::runtime_error("VecCreateMPIWithArray(x_global) failed"); }

                A.scatter_solution(Xu, Xd, xg);

                VecDestroy(&Xu);
                VecDestroy(&Xd);
                VecDestroy(&xg);
            },
            py::arg("X_u"), py::arg("X_d"), py::arg("x_global"),
            "Scatter local unknown/prescribed solutions into the global vector.")

        .def("__repr__", [](const GooseFEM::MatrixPartitionedTyings_PETSc&) {
            return "<MatrixPartitionedTyings_PETSc PETSc-backed matrix>";
        });
}