/**
 * Sparse matrix for PETSc solver that is partitioned in:
 * -   unknown DOFs
 * -   prescribed DOFs
 * -   tied DOFs
 *
 * @file MatrixPartitionedTyings.h
 * @copyright Copyright 2025. Philipp van der Loo. All rights reserved.
 * @license This project is released under the GNU Public License (GPLv3).
 */

#ifndef GOOSEFEM_MATRIXPARTITIONEDTYINGS_PETSC_H
#define GOOSEFEM_MATRIXPARTITIONEDTYINGS_PETSC_H

#include <petscmat.h>
#include <petscerror.h>
#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>
#include <limits>

namespace GooseFEM {

inline void initializePetscFromPython() {
    PetscErrorCode ierr;
    PetscBool petsc_initialized;

    // Check if PETSc is already initialized
    ierr = PetscInitialized(&petsc_initialized); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    if (!petsc_initialized) {
        int argc = 0;
        char **argv = nullptr;

        // Initialize PETSc
        ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
        if (ierr) {
            throw std::runtime_error("PETSc initialization failed");
        }
    }

    // After this, PETSc is guaranteed to be initialized
}

// forward declaration for solver if needed
class SolverPartitionedTyings_PETSc;

class MatrixPartitionedTyings_PETSc {
private:
    // NOTE: PETSc types (Mat) are pointers under the hood; initialize to nullptr.
    Mat m_A = nullptr;    ///< global assembled matrix (all dofs)
    Mat m_Cdu = nullptr;  ///< tying matrix (dependent rows, unknown cols)
    Mat m_Cud = nullptr;  ///< transpose of Cdu
    Mat m_Cdp = nullptr;  ///< tying matrix (dependent rows, prescribed cols)
    Mat m_Cpd = nullptr;  ///< transpose of Cdp
    Mat m_ACuu = nullptr; ///< condensed system matrix (optional)
    Mat m_ACup = nullptr; ///< condensed system matrix (optional)

    /* PETSc index sets and scatter contexts (add as members) */
    IS m_IS_u = nullptr;
    IS m_IS_d = nullptr;
    IS m_IS_p = nullptr;

    VecScatter m_scatter_u = nullptr;
    VecScatter m_scatter_d = nullptr;

    // bookkeeping copied from your Eigen version - keep them public/protected as required
protected:
    array_type::tensor<size_t, 2> m_dofs;
    size_t m_nnode = 0, m_ndim = 0;
    size_t m_nnu = 0, m_nnp = 0, m_nnd = 0, m_nni = 0, m_ndof = 0;
    xt::xtensor<size_t,1> m_iiu, m_iip, m_iii, m_iid;
    Eigen::SparseMatrix<double> m_Cud_eig, m_Cpd_eig; // if you still want Eigen copies

    // grant access to solver class
    friend class SolverPartitionedTyings_PETSc;

public:
    MatrixPartitionedTyings_PETSc() = default;

    MatrixPartitionedTyings_PETSc(
        const array_type::tensor<size_t, 2>& dofs,
        const Eigen::SparseMatrix<double>& Cdu,
        const Eigen::SparseMatrix<double>& Cdp
    ) {
        GooseFEM::initializePetscFromPython();
        GOOSEFEM_ASSERT(Cdu.rows() == Cdp.rows());

        m_dofs = dofs;
        m_nnode = m_dofs.shape(0);
        m_ndim = m_dofs.shape(1);
        m_Cud_eig = Cdu.transpose();
        m_Cpd_eig = Cdp.transpose();

        m_nnu = static_cast<size_t>(Cdu.cols());
        m_nnp = static_cast<size_t>(Cdp.cols());
        m_nnd = static_cast<size_t>(Cdp.rows());
        m_nni = m_nnu + m_nnp;
        m_ndof = m_nni + m_nnd;

        m_iiu = xt::arange<size_t>(m_nnu);
        m_iip = xt::arange<size_t>(m_nnu, m_nnu + m_nnp);
        m_iii = xt::arange<size_t>(m_nni);
        m_iid = xt::arange<size_t>(m_nni, m_nni + m_nnd);

        GOOSEFEM_ASSERT(m_ndof <= m_nnode * m_ndim);
        GOOSEFEM_ASSERT(m_ndof == xt::amax(m_dofs)() + 1);

        PetscErrorCode ierr;
        ierr = EigenToPETScMat(Cdu, &m_Cdu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = EigenToPETScMat(Cdp, &m_Cdp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = EigenToPETScMat(m_Cud_eig, &m_Cud); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = EigenToPETScMat(m_Cpd_eig, &m_Cpd); CHKERRABORT(PETSC_COMM_WORLD, ierr);      

        ierr = MatCreate(PETSC_COMM_WORLD, &m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetType(m_A, MATMPIAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        
        int rank, size;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        
        ierr = MatSetSizes(m_A, PETSC_DECIDE, PETSC_DECIDE,
                           (PetscInt)m_ndof, (PetscInt)m_ndof); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        
        // ierr = MatSetType(m_A, MATAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetUp(m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        
        
        this->initialize_index_sets();
    }

    virtual ~MatrixPartitionedTyings_PETSc() {
        destroy_petsc_tyings_objects();
    }    
    // Getters for protected members
    size_t nnode() const { return m_nnode; }
    size_t ndim() const { return m_ndim; }
    size_t nnu()  const { return m_nnu; }
    size_t nnp()  const { return m_nnp; }
    size_t nnd()  const { return m_nnd; }
    size_t ndof() const { return m_ndof; }

    // Accessors
    const Mat& data_ACuu() const { return m_ACuu; }
    const Mat& data_Cdu() const { return m_Cdu; }
    const Mat& data_Cdp() const { return m_Cdp; }
    const Mat& data_A() const { return m_A; }

    // Creates a 'view' (sub) of the data in the global vector 'vec_global'.
    PetscErrorCode GetDofs_u(Vec& vec_global, Vec* sub)
    {
        return VecGetSubVector(vec_global, m_IS_u, sub);
    }

    PetscErrorCode GetDofs_d(Vec& vec_global, Vec* sub)
    {
        return VecGetSubVector(vec_global, m_IS_d, sub);
    }

    PetscErrorCode GetDofs_p(Vec& vec_global, Vec* sub)
    {
        return VecGetSubVector(vec_global, m_IS_p, sub);
    } 
    
    // Restores the sub-vector view 'sub' to the global vector 'vec_global'.
    PetscErrorCode RestoreDofs_u(Vec& vec_global, Vec& sub)
    {
        return VecRestoreSubVector(vec_global, m_IS_u, &sub);
    }

    PetscErrorCode RestoreDofs_d(Vec& vec_global, Vec& sub)
    {
        return VecRestoreSubVector(vec_global, m_IS_d, &sub);
    }

    PetscErrorCode RestoreDofs_p(Vec& vec_global, Vec& sub)
    {
        return VecRestoreSubVector(vec_global, m_IS_p, &sub);
    }

    void scatter_solution(Vec X_u, Vec X_d, Vec& x_global) {
        PetscErrorCode ierr;

        std::vector<PetscInt> idx;
        std::vector<PetscScalar> vals;

        PetscScalar val;
        for (size_t m = 0; m < this->m_nnode; ++m) {
            for (size_t i = 0; i < this->m_ndim; ++i) {
                PetscInt gdof = (PetscInt)this->m_dofs(m,i);

                if (gdof < (PetscInt)this->m_nnu) {
                    ierr = VecGetValues(X_u, 1, &gdof, &val); CHKERRABORT(PETSC_COMM_WORLD, ierr);
                    idx.push_back(gdof);
                    vals.push_back(val);
                } 
                else if (gdof < (PetscInt)this->m_nni) {
                    ierr = VecGetValues(x_global, 1, &gdof, &val); CHKERRABORT(PETSC_COMM_WORLD, ierr);
                    idx.push_back(gdof);
                    vals.push_back(val);
                }
                else { 
                    PetscInt local_d = gdof - this->m_nni;
                    ierr = VecGetValues(X_d, 1, &local_d, &val); CHKERRABORT(PETSC_COMM_WORLD, ierr);
                    idx.push_back(gdof);
                    vals.push_back(val);
                }
            }
        }

        // PetscPrintf(PETSC_COMM_WORLD, "Collected Global DOFs (idx) on Rank %d:\n", 0); // Assuming m_rank is available
 
        // for (size_t k = 0; k < vals.size(); ++k) {
        //     // Print 10 indices per line for readability
        //     if (k % 10 == 0) {
        //         PetscPrintf(PETSC_COMM_WORLD, "\n");
        //     }
        //     PetscPrintf(PETSC_COMM_WORLD, "%4f ", vals[k]);
        // }
        // PetscPrintf(PETSC_COMM_WORLD, "\n");

        std::iota(idx.begin(), idx.end(), 0); // usually 0 or local offset
        ierr = VecSetValues(x_global, idx.size(), idx.data(), vals.data(), INSERT_VALUES); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // ierr = VecSetValues(x_global, vals.size(), NULL, vals.data(), INSERT_VALUES);
        ierr = VecAssemblyBegin(x_global); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyEnd(x_global); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // PetscScalar max_val;
        // VecMax(x_global, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of push-back vector x_py = %g\n", PetscRealPart(max_val));
    }

    // Assemble element matrices into m_A
    template <class T_ElemMat, class T_Conn>
    void assemble(const T_ElemMat& elemmat, const T_Conn& conn_elem)
    {
        // Basic checks
        size_t nelem = elemmat.shape()[0];
        size_t nnodes_per_elem = conn_elem.shape()[1];
        GOOSEFEM_ASSERT(elemmat.shape()[1] == nnodes_per_elem * m_ndim && "elemmat row mismatch");
        GOOSEFEM_ASSERT(elemmat.shape()[2] == nnodes_per_elem * m_ndim && "elemmat col mismatch");

        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        // Get the ownership range of this matrix on this rank
        PetscInt rstart, rend;
        MatGetOwnershipRange(m_A, &rstart, &rend);

        for (ptrdiff_t e = 0; e < (ptrdiff_t)nelem; ++e) {
            size_t size = nnodes_per_elem * m_ndim;

            // Global DOF indices for this element
            std::vector<PetscInt> idx(size);
            for (ptrdiff_t n = 0; n < (ptrdiff_t)nnodes_per_elem; ++n) {
                for (ptrdiff_t j = 0; j < (ptrdiff_t)m_ndim; ++j) {
                    idx[n * m_ndim + j] = static_cast<PetscInt>(m_dofs(conn_elem(e, n), j));
                }
            }

            // Copy element matrix into contiguous vals (row-major)
            std::vector<PetscScalar> vals(size * size);
            for (size_t i = 0; i < size; ++i)
                for (size_t j = 0; j < size; ++j)
                    vals[i * size + j] = static_cast<PetscScalar>(elemmat(e, i, j));

            // Filter rows to only those owned by this rank
            std::vector<PetscInt> idx_local;
            std::vector<PetscScalar> vals_local;

            for (size_t i = 0; i < size; ++i) {
                if (idx[i] >= rstart && idx[i] < rend) {
                    idx_local.push_back(idx[i]);
                }
            }

            // Skip if no owned DOFs
            if (idx_local.empty()) continue;

            // Insert into PETSc matrix
            PetscErrorCode ierr = MatSetValues(m_A,
                                            static_cast<PetscInt>(idx_local.size()),
                                            idx_local.data(),
                                            static_cast<PetscInt>(size),
                                            idx.data(),
                                            vals.data(),
                                            ADD_VALUES);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            #ifdef DEBUG_ASSEMBLE
            if(rank == 0) {
                std::cout << "Element " << e << " inserted idx: ";
                for(auto v : idx_local) std::cout << v << " ";
                std::cout << std::endl;
            }
            #endif
        }
    }

    PetscErrorCode finalize(bool stabilize)
    {
        PetscErrorCode ierr;

        // Assemble the matrix
        ierr = MatAssemblyBegin(m_A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(m_A, MAT_FINAL_ASSEMBLY);   CHKERRQ(ierr);

        if (stabilize) {
            Vec diag;
            ierr = MatCreateVecs(m_A, &diag, NULL); CHKERRQ(ierr);
            ierr = MatGetDiagonal(m_A, diag); CHKERRQ(ierr);

            // Get local ownership info
            PetscInt local_size;
            ierr = VecGetLocalSize(diag, &local_size); CHKERRQ(ierr);
            const PetscScalar* diag_array = nullptr;
            ierr = VecGetArrayRead(diag, &diag_array); CHKERRQ(ierr);

            // Compute minimal nonzero diagonal on this rank
            PetscScalar min_diag = PETSC_MAX_REAL;
            for (PetscInt i = 0; i < local_size; ++i) {
                PetscScalar v = diag_array[i];
                if (std::abs(v) > 1e-12 && std::abs(v) < min_diag) min_diag = std::abs(v);
            }
            ierr = VecRestoreArrayRead(diag, &diag_array); CHKERRQ(ierr);

            // Reduce to find global minimal diagonal
            PetscScalar global_min_diag;
            ierr = MPI_Allreduce(&min_diag, &global_min_diag, 1, MPIU_SCALAR, MPI_MIN, PETSC_COMM_WORLD); CHKERRQ(ierr);

            if (global_min_diag < PETSC_MAX_REAL) {
                PetscScalar spring = 1e-3 * global_min_diag;

                // Create vector to hold diagonal additions
                Vec adddiag;
                ierr = VecDuplicate(diag, &adddiag); CHKERRQ(ierr);
                ierr = VecSet(adddiag, 0.0); CHKERRQ(ierr);

                PetscScalar* adddiag_array = nullptr;
                ierr = VecGetArray(adddiag, &adddiag_array); CHKERRQ(ierr);
                ierr = VecGetArrayRead(diag, &diag_array); CHKERRQ(ierr);

                for (PetscInt i = 0; i < local_size; ++i) {
                    if (std::abs(diag_array[i]) < 1e-12) adddiag_array[i] = spring;
                }

                ierr = VecRestoreArrayRead(diag, &diag_array); CHKERRQ(ierr);
                ierr = VecRestoreArray(adddiag, &adddiag_array); CHKERRQ(ierr);

                ierr = MatDiagonalSet(m_A, adddiag, ADD_VALUES); CHKERRQ(ierr);
                ierr = VecDestroy(&adddiag); CHKERRQ(ierr);
            }

            ierr = VecDestroy(&diag); CHKERRQ(ierr);
        }

        m_changed = true;
        return 0;
    }

    void clear()
    {
        if (m_A) {
            PetscErrorCode ierr = MatZeroEntries(m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
    }

private:
    // destroy mats
    void destroy_petsc_tyings_objects() {
        if (m_A) MatDestroy(&m_A);
        if (m_Cdu) MatDestroy(&m_Cdu);
        if (m_Cdp) MatDestroy(&m_Cdp);
        if (m_ACuu) MatDestroy(&m_ACuu);
        if (m_ACup) MatDestroy(&m_ACup);
        if (m_IS_u) ISDestroy(&m_IS_u);
        if (m_IS_d) ISDestroy(&m_IS_d);
        if (m_IS_p) ISDestroy(&m_IS_p);
        if (m_scatter_u) VecScatterDestroy(&m_scatter_u);
        if (m_scatter_d) VecScatterDestroy(&m_scatter_d);
    }

    // Convert Eigen sparse -> PETSc Mat (creates a SeqAIJ/MPiAIJ depending on communicator)
    PetscErrorCode EigenToPETScMat(const Eigen::SparseMatrix<double>& eigen_mat, Mat* outMat) {
        PetscErrorCode ierr;
        PetscInt rows = (PetscInt)eigen_mat.rows();
        PetscInt cols = (PetscInt)eigen_mat.cols();

        // Count nonzeros per row
        std::vector<PetscInt> nnz_per_row(rows, 0);
        for (int k = 0; k < eigen_mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_mat, k); it; ++it)
                nnz_per_row[it.row()]++;
        PetscInt max_nnz_row = *std::max_element(nnz_per_row.begin(), nnz_per_row.end());

        // --- Create parallel matrix (MPIAIJ) even for 1 rank ---
        ierr = MatCreate(PETSC_COMM_WORLD, outMat); CHKERRQ(ierr);
        ierr = MatSetSizes(*outMat, PETSC_DECIDE, PETSC_DECIDE, rows, cols); CHKERRQ(ierr);
        ierr = MatSetType(*outMat, MATMPIAIJ); CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(*outMat, max_nnz_row, NULL, max_nnz_row, NULL); CHKERRQ(ierr);
        ierr = MatSetUp(*outMat); CHKERRQ(ierr);

        // Insert values
        for (int k = 0; k < eigen_mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_mat, k); it; ++it) {
                PetscInt i = (PetscInt)it.row();
                PetscInt j = (PetscInt)it.col();
                PetscScalar v = (PetscScalar)it.value();
                ierr = MatSetValues(*outMat, 1, &i, 1, &j, &v, INSERT_VALUES); CHKERRQ(ierr);
            }

        ierr = MatAssemblyBegin(*outMat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*outMat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        return 0;
    }

    void initialize_index_sets(MPI_Comm comm = PETSC_COMM_WORLD) {
        PetscErrorCode ierr;
        int rank, size;        
        ierr = MPI_Comm_rank(comm, &rank); CHKERRABORT(comm, ierr);
        ierr = MPI_Comm_size(comm, &size); CHKERRABORT(comm, ierr);

        // Get ownership range of the global matrix/vector
        PetscInt rstart, rend;
        ierr = MPI_Comm_rank(comm, &rank); CHKERRABORT(comm, ierr);
        
        // Create index vectors for local ownership only
        std::vector<PetscInt> iu_local, id_local, ip_local;

        // m_iiu, m_iid, m_iip contain global DOF indices
        for (size_t k = 0; k < m_nnu; ++k) {
            PetscInt idx = static_cast<PetscInt>(m_iiu(k));
            iu_local.push_back(idx);
        }
        for (size_t k = 0; k < m_nnd; ++k) {
            PetscInt idx = static_cast<PetscInt>(m_iid(k));
            id_local.push_back(idx);
        }
        for (size_t k = 0; k < m_nnp; ++k) {
            PetscInt idx = static_cast<PetscInt>(m_iip(k));
            ip_local.push_back(idx);
        }

        // Create global index sets (can be full IS)
        ierr = ISCreateGeneral(comm, iu_local.size(), iu_local.data(), PETSC_COPY_VALUES, &m_IS_u); CHKERRABORT(comm, ierr);
        ierr = ISCreateGeneral(comm, id_local.size(), id_local.data(), PETSC_COPY_VALUES, &m_IS_d); CHKERRABORT(comm, ierr);
        ierr = ISCreateGeneral(comm, ip_local.size(), ip_local.data(), PETSC_COPY_VALUES, &m_IS_p); CHKERRABORT(comm, ierr);

        // Determine local sizes for VecScatter
        PetscInt n_u_local = iu_local.size();
        PetscInt n_d_local = id_local.size();

        // Temporary sequential vectors (local only)
        Vec tmp_u, tmp_d, x_global_dummy;
        ierr = VecCreateSeq(PETSC_COMM_SELF, n_u_local, &tmp_u); CHKERRABORT(comm, ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, n_d_local, &tmp_d); CHKERRABORT(comm, ierr);

        // Global distributed vector
        ierr = VecCreateMPI(comm, PETSC_DECIDE, (PetscInt)m_ndof, &x_global_dummy); CHKERRABORT(comm, ierr);

        // Create scatter from local seq vectors to global vector using IS
        ierr = VecScatterCreate(tmp_u, NULL, x_global_dummy, m_IS_u, &m_scatter_u); CHKERRABORT(comm, ierr);
        ierr = VecScatterCreate(tmp_d, NULL, x_global_dummy, m_IS_d, &m_scatter_d); CHKERRABORT(comm, ierr);

        // Cleanup temporary vectors
        VecDestroy(&tmp_u);
        VecDestroy(&tmp_d);
        VecDestroy(&x_global_dummy);
    }

    // placeholder flag used earlier
    bool m_changed = false;
};

} // namespace GooseFEM

#endif
