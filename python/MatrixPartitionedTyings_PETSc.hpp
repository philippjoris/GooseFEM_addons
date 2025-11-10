// bindings_MatrixPartitionedTyings_PETSc.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Sparse>
#include <petscmat.h>
#include <vector>
#include <stdexcept>

#include "MatrixPartitionedTyings_PETSc.hpp" // your header with the class

namespace py = pybind11;

/* Helper: convert dense numpy array (float64) -> Eigen::SparseMatrix<double> */
static Eigen::SparseMatrix<double> dense_numpy_to_eigen_sparse(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    py::buffer_info info = arr.request();
    if (info.ndim != 2) throw std::runtime_error("Expected 2D array for dense matrix.");

    const size_t rows = static_cast<size_t>(info.shape[0]);
    const size_t cols = static_cast<size_t>(info.shape[1]);
    const double* data = static_cast<const double*>(info.ptr);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve( std::min<size_t>(rows * cols, 1024) );

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double v = data[i * cols + j];
            if (v != 0.0) {
                triplets.emplace_back((Eigen::Index)i, (Eigen::Index)j, v);
            }
        }
    }

    Eigen::SparseMatrix<double> S((Eigen::Index)rows, (Eigen::Index)cols);
    if (!triplets.empty()) {
        S.setFromTriplets(triplets.begin(), triplets.end());
    }
    return S;
}

/* Helper: convert numpy uint64 (or int) dofs -> array_type::tensor<size_t,2>
   We keep it simple: convert to std::vector and then to xtensor inside C++ constructor call. */
static array_type::tensor<size_t, 2> numpy_to_dofs(py::array_t<size_t, py::array::c_style | py::array::forcecast> dofs_arr)
{
    py::buffer_info info = dofs_arr.request();
    if (info.ndim != 2) throw std::runtime_error("dofs must be 2D array (nnode, ndim).");
    const size_t nnode = info.shape[0];
    const size_t ndim = info.shape[1];
    const size_t* ptr = static_cast<const size_t*>(info.ptr);

    array_type::tensor<size_t, 2> dofs({nnode, ndim});
    for (size_t i = 0; i < nnode; ++i) {
        for (size_t j = 0; j < ndim; ++j) {
            dofs(i, j) = ptr[i * ndim + j];
        }
    }
    return dofs;
}

/* Helper: convert 3D elemmat numpy array to temporary vector-of-matrices we can pass to assemble.
   We'll directly call Matrix::assemble template expecting elemmat and conn_elem; for convenience
   we provide a wrapper that builds simple C++ containers acceptable by your assemble template.
*/
static std::vector<std::vector<double>> flatten_elem_block(const py::array_t<double>& block)
{
    // Not used directly in binding below; included for future extension.
    return {};
}

void init_MatrixPartitionedTyings_PETSc(py::module& m)
{
    py::class_<MatrixPartitionedTyings_PETSc>(m, "MatrixPartitionedTyings_PETSc")
        // constructor: dofs (np.uint64 2D), Cdu dense (float64 2D), Cdp dense (float64 2D)
        .def(py::init([](py::array_t<size_t, py::array::c_style | py::array::forcecast> dofs_arr,
                         py::array_t<double, py::array::c_style | py::array::forcecast> Cdu_dense,
                         py::array_t<double, py::array::c_style | py::array::forcecast> Cdp_dense) {

            // convert dofs
            array_type::tensor<size_t, 2> dofs = numpy_to_dofs(dofs_arr);

            // convert dense arrays to Eigen sparse
            Eigen::SparseMatrix<double> Cdu_sp = dense_numpy_to_eigen_sparse(Cdu_dense);
            Eigen::SparseMatrix<double> Cdp_sp = dense_numpy_to_eigen_sparse(Cdp_dense);

            // Return new instance (will call your C++ ctor)
            return MatrixPartitionedTyings_PETSc(dofs, Cdu_sp, Cdp_sp);
        }), R"doc(
            Construct MatrixPartitionedTyings_PETSc(dofs, Cdu_dense, Cdp_dense)

            dofs : numpy.ndarray (nnode, ndim) dtype=uint64
            Cdu_dense, Cdp_dense : numpy.ndarray (dense float64) converted to sparse internally
        )doc",
           py::arg("dofs"), py::arg("Cdu_dense"), py::arg("Cdp_dense"))

        // properties: sizes
        .def_property_readonly("nnode", [](const MatrixPartitionedTyings_PETSc &A){ return (size_t)A.m_nnode; })
        .def_property_readonly("ndim", [](const MatrixPartitionedTyings_PETSc &A){ return (size_t)A.m_ndim; })
        .def_property_readonly("ndof", [](const MatrixPartitionedTyings_PETSc &A){ return (size_t)A.m_ndof; })
        .def_property_readonly("nnu", [](const MatrixPartitionedTyings_PETSc &A){ return (size_t)A.m_nnu; })
        .def_property_readonly("nnp", [](const MatrixPartitionedTyings_PETSc &A){ return (size_t)A.m_nnp; })
        .def_property_readonly("nnd", [](const MatrixPartitionedTyings_PETSc &A){ return (size_t)A.m_nnd; })

        // clear() -> reset global matrix entries
        .def("clear", &MatrixPartitionedTyings_PETSc::clear, "Zero the global matrix (MatZeroEntries)")

        // assemble(elemmat: np.ndarray (nelem, size, size), conn_elem: np.ndarray (nelem, nnodes_per_elem) )
        .def("assemble", [](MatrixPartitionedTyings_PETSc &A,
                            py::array_t<double, py::array::c_style | py::array::forcecast> elemmat,
                            py::array_t<int64_t, py::array::c_style | py::array::forcecast> conn_elem) {

            // buffer checks
            py::buffer_info be = elemmat.request();
            if (be.ndim != 3) throw std::runtime_error("elemmat must be 3D (nelem, nrow, ncol)");
            const size_t nelem = (size_t)be.shape[0];
            const size_t nrow  = (size_t)be.shape[1];
            const size_t ncol  = (size_t)be.shape[2];
            if (nrow != ncol) throw std::runtime_error("elemmat blocks must be square");

            py::buffer_info bc = conn_elem.request();
            if (bc.ndim != 2) throw std::runtime_error("conn_elem must be 2D (nelem, nnodes_per_elem)");
            const size_t nelem2 = (size_t)bc.shape[0];
            if (nelem2 != nelem) throw std::runtime_error("elemmat and conn_elem nelem mismatch");

            // Build simple C++ containers to call the templated assemble_impl
            // We'll create a vector of pointers to contiguous element matrices and a 2D conn vector
            // but to keep memory simple we call MatSetValues per element via the public assemble wrapper.

            // Create lightweight wrapper objects:
            // Build conn_elem as std::vector<std::vector<size_t>>
            std::vector<std::vector<size_t>> conn(nelem);
            const int64_t *conn_ptr = static_cast<const int64_t*>(bc.ptr);
            for (size_t e = 0; e < nelem; ++e) {
                conn[e].resize((size_t)bc.shape[1]);
                for (size_t n = 0; n < (size_t)bc.shape[1]; ++n) {
                    conn[e][n] = static_cast<size_t>( conn_ptr[e * bc.shape[1] + n] );
                }
            }

            // Call MatSetValues via the public assemble template wrapper (we need to convert elemmat per-element)
            const double *em_ptr = static_cast<const double*>(be.ptr);
            const size_t block_size = nrow;
            for (size_t e = 0; e < nelem; ++e) {
                // Create a temporary contiguous vector<double> representing the element matrix in row-major
                std::vector<double> block(block_size * block_size);
                for (size_t i = 0; i < block_size; ++i)
                    for (size_t j = 0; j < block_size; ++j)
                        block[i * block_size + j] = em_ptr[e * block_size * block_size + i * block_size + j];

                // Build a tiny lambda-compatible wrapper that your C++ class can consume.
                // We'll call MatSetValues for each element here, matching the behavior in your C++ assemble_impl.
                // Get global idx
                std::vector<PetscInt> idx(block_size);
                for (size_t n = 0; n < block_size; ++n) {
                    // node index mapping expects m_dofs(conn_elem(e,n), j) interleaved with ndim.
                    // For simplicity, assume conn array contains *_dof indices directly (user must pass dof indices).
                    idx[n] = static_cast<PetscInt>( conn[e][n] );
                }

                // Insert values into matrix (row-major -> MatSetValues expects row-major contiguous)
                PetscErrorCode ierr = MatSetValues(A.data_A(), (PetscInt)block_size, idx.data(),
                                                  (PetscInt)block_size, idx.data(),
                                                  block.data(), ADD_VALUES);
                if (ierr) throw std::runtime_error("MatSetValues failed in assemble binding.");
            }
        }, py::arg("elemmat"), py::arg("conn_elem"),
           "Assemble element matrices into global PETSc matrix. \
            elemmat: (nelem, size, size) float64, conn_elem: (nelem, nodes_per_elem) int64. \
            NOTE: conn_elem must contain global DOF indices for each element DOF in natural order.")

        // finalize(stabilize: bool)
        .def("finalize", &MatrixPartitionedTyings_PETSc::finalize, py::arg("stabilize") = true,
             "Finalize assembly and optionally add stabilization to diagonal.")

        // scatter_solution(X_u, X_d, x_global) : all numpy arrays
        .def("scatter_solution",
            [](MatrixPartitionedTyings_PETSc &A,
               py::array_t<double, py::array::c_style | py::array::forcecast> Xu_arr,
               py::array_t<double, py::array::c_style | py::array::forcecast> Xd_arr,
               py::array_t<double, py::array::c_style | py::array::forcecast> x_global_arr) {

                // Xu and Xd are local sequential arrays (length m_nnu and m_nnd)
                py::buffer_info bu = Xu_arr.request();
                py::buffer_info bd = Xd_arr.request();
                py::buffer_info bg = x_global_arr.request();

                if (bu.ndim != 1 || (size_t)bu.shape[0] != A.m_nnu) throw std::runtime_error("Xu length mismatch");
                if (bd.ndim != 1 || (size_t)bd.shape[0] != A.m_nnd) throw std::runtime_error("Xd length mismatch");
                if (bg.ndim != 1 || (size_t)bg.shape[0] != A.m_ndof) throw std::runtime_error("x_global length mismatch");

                // Create Vec views from arrays (sequential local vectors)
                Vec Xu = nullptr, Xd = nullptr, xg = nullptr;
                // Xu, Xd are local (sequential)
                PetscErrorCode ierr;
                ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, (PetscInt)A.m_nnu, static_cast<const PetscScalar*>(bu.ptr), &Xu);
                if (ierr) throw std::runtime_error("VecCreateSeqWithArray(Xu) failed");
                ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, (PetscInt)A.m_nnd, static_cast<const PetscScalar*>(bd.ptr), &Xd);
                if (ierr) { VecDestroy(&Xu); throw std::runtime_error("VecCreateSeqWithArray(Xd) failed"); }

                // Global vector: create MPI Vec that uses existing array if possible
                // For safety create a Seq or MPI Vec and copy content after scatter results
                ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD, 1, PETSC_DECIDE, (PetscInt)A.m_ndof,
                                             static_cast<const PetscScalar*>(bg.ptr), &xg);
                if (ierr) { VecDestroy(&Xu); VecDestroy(&Xd); throw std::runtime_error("VecCreateMPIWithArray(x_global) failed"); }

                // Call scatter_solution
                A.scatter_solution(Xu, Xd, xg);

                // No need to copy back because we used VecCreateMPIWithArray (backed by user array)
                VecDestroy(&Xu);
                VecDestroy(&Xd);
                VecDestroy(&xg);
            },
            py::arg("X_u"), py::arg("X_d"), py::arg("x_global"),
            "Scatter local X_u and X_d into the global vector x_global (all numpy arrays).")

        // repr
        .def("__repr__", [](const MatrixPartitionedTyings_PETSc&) {
            return "<MatrixPartitionedTyings_PETSc PETSc-backed matrix>";
        })
        ;
}