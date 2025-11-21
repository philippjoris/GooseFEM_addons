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
    // Mat m_ACuu = nullptr; ///< condensed system matrix (optional)
    // Mat m_ACup = nullptr; ///< condensed system matrix (optional)
    Mat m_Auu = nullptr, m_Aud = nullptr, m_Adu = nullptr, m_Add = nullptr, m_Aup = nullptr, m_Adp = nullptr;
    

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
        
        int rank, size; MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        PetscErrorCode ierr;     

        ierr = MatCreate(PETSC_COMM_WORLD, &m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetType(m_A, MATMPIAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        
        ierr = MatSetSizes(m_A, PETSC_DECIDE, PETSC_DECIDE,
                           (PetscInt)m_ndof, (PetscInt)m_ndof); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        
        PetscInt d_nz = 50;
        PetscInt o_nz = 50;

        ierr = MatMPIAIJSetPreallocation(m_A, d_nz, NULL, o_nz, NULL); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetOption(m_A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = MatSetUp(m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // ierr = MatSetType(m_A, MATAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // ierr = MatSetUp(m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);        
        
        this->initialize_index_sets();
        
        PetscInt local_d_size, local_u_size, local_p_size;
        ierr = ISGetLocalSize(m_IS_d, &local_d_size); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = ISGetLocalSize(m_IS_u, &local_u_size); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = ISGetLocalSize(m_IS_p, &local_p_size); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        
        ierr = EigenToPETScMat(Cdu, &m_Cdu, local_d_size); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = EigenToPETScMat(Cdp, &m_Cdp, local_d_size); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = EigenToPETScMat(m_Cud_eig, &m_Cud, local_u_size); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = EigenToPETScMat(m_Cpd_eig, &m_Cpd, local_p_size); CHKERRABORT(PETSC_COMM_WORLD, ierr); 
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
    // const Mat& data_ACuu() const { return m_ACuu; }
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
        // PetscMPIInt rank;
        // MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        
        // 1. Scatter X_u (Unknown DOFs, local block vector) to x_global (MPI vector)
        ierr = VecScatterBegin(this->m_scatter_u, X_u, x_global, INSERT_VALUES, SCATTER_FORWARD); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecScatterEnd(this->m_scatter_u, X_u, x_global, INSERT_VALUES, SCATTER_FORWARD); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // 2. Scatter X_d (Dependent DOFs, local block vector) to x_global (MPI vector)
        ierr = VecScatterBegin(this->m_scatter_d, X_d, x_global, INSERT_VALUES, SCATTER_FORWARD); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecScatterEnd(this->m_scatter_d, X_d, x_global, INSERT_VALUES, SCATTER_FORWARD); CHKERRABORT(PETSC_COMM_WORLD, ierr);

    }

    // Assemble element matrices into m_A
    template <class T_ElemMat, class T_Conn>
    void assemble(const T_ElemMat& elemmat, const T_Conn& conn_elem)
    {
        // Basic checks
        size_t nelem = elemmat.shape()[0];

        size_t nnodes_per_elem = conn_elem.shape()[1];
        size_t size = nnodes_per_elem * m_ndim; // Element matrix size (size x size)
        GOOSEFEM_ASSERT(elemmat.shape()[1] == size && "elemmat row mismatch");
        GOOSEFEM_ASSERT(elemmat.shape()[2] == size && "elemmat col mismatch");


        // Get the ownership range of this matrix on this rank
        PetscInt rstart, rend;
        MatGetOwnershipRange(m_A, &rstart, &rend);

        for (ptrdiff_t e = 0; e < (ptrdiff_t)nelem; ++e) {
            
            // 1. Global DOF indices for this element (used for J_cols)
            std::vector<PetscInt> idx(size);
            for (ptrdiff_t n = 0; n < (ptrdiff_t)nnodes_per_elem; ++n) {
                for (ptrdiff_t j = 0; j < (ptrdiff_t)m_ndim; ++j) {
                    idx[n * m_ndim + j] = static_cast<PetscInt>(m_dofs(conn_elem(e, n), j));
                }
            }

            // 2. Copy full element matrix into contiguous vals (size x size)
            std::vector<PetscScalar> vals(size * size);
            for (size_t i = 0; i < size; ++i)
                for (size_t j = 0; j < size; ++j)
                    vals[i * size + j] = static_cast<PetscScalar>(elemmat(e, i, j));

            // 3. Filter rows to only those owned by this rank, and extract corresponding matrix rows
            std::vector<PetscInt> idx_local;        // I_rows: Indices of owned rows
            std::vector<PetscScalar> vals_local;     // V_vals: Data corresponding to owned rows (idx_local.size() * size)
            
            for (size_t i = 0; i < size; ++i) {
                // Check if the i-th global DOF for this element is owned by this rank
                if (idx[i] >= rstart && idx[i] < rend) {
                    
                    // Add the global DOF index to the owned row list
                    idx_local.push_back(idx[i]);

                    // Copy the ENTIRE i-th row of the element matrix (size columns)
                    // The row starts at index i * size in the flat 'vals' array
                    size_t start_index = i * size;
                    
                    // Copy 'size' columns from the element matrix into vals_local
                    vals_local.insert(vals_local.end(), 
                                    vals.begin() + start_index, 
                                    vals.begin() + start_index + size);
                }
            }

            // Skip if no owned DOFs in this element
            if (idx_local.empty()) continue;

            // 4. Insert into PETSc matrix (Only for the owned rows)
            PetscErrorCode ierr = MatSetValues(m_A,
                                                static_cast<PetscInt>(idx_local.size()), // N_rows: Count of owned rows
                                                idx_local.data(),                         // I_rows: Global indices of owned rows
                                                static_cast<PetscInt>(size),             // N_cols: Count of ALL element DOFs/columns
                                                idx.data(),                               // J_cols: Global indices of ALL element columns
                                                vals_local.data(),                        // V_vals: Filtered data for owned rows
                                                ADD_VALUES);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);

            #ifdef DEBUG_ASSEMBLE
            if(rank == 0) {
                std::cout << "Element " << e << " inserted " << idx_local.size() << " rows." << std::endl;
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
        if (m_Auu) MatDestroy(&m_Auu);
        if (m_Aup) MatDestroy(&m_Aup);
        if (m_IS_u) ISDestroy(&m_IS_u);
        if (m_IS_d) ISDestroy(&m_IS_d);
        if (m_IS_p) ISDestroy(&m_IS_p);
        if (m_scatter_u) VecScatterDestroy(&m_scatter_u);
        if (m_scatter_d) VecScatterDestroy(&m_scatter_d);
    }

    // Convert Eigen sparse -> PETSc Mat (creates a SeqAIJ/MPiAIJ depending on communicator)
    PetscErrorCode EigenToPETScMat(
    const Eigen::SparseMatrix<double>& eigen_mat, 
    Mat* outMat, 
    // New parameter: Pass PETSC_DECIDE to use the default partitioning, 
    // or a specific size (like the local size of m_IS_d).
    PetscInt local_rows_override = PETSC_DECIDE) 
    {
        PetscErrorCode ierr;
        PetscInt rows = (PetscInt)eigen_mat.rows();
        PetscInt cols = (PetscInt)eigen_mat.cols();

        // Count nonzeros per row (rest of the preallocation logic is the same)
        std::vector<PetscInt> nnz_per_row(rows, 0);
        for (int k = 0; k < eigen_mat.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_mat, k); it; ++it)
                nnz_per_row[it.row()]++;
        PetscInt max_nnz_row = *std::max_element(nnz_per_row.begin(), nnz_per_row.end());

        // --- Create parallel matrix (MPIAIJ) ---
        ierr = MatCreate(PETSC_COMM_WORLD, outMat); CHKERRQ(ierr);
        
        // Core Change: Use the override for local rows
        PetscInt local_rows = (local_rows_override == PETSC_DECIDE) ? PETSC_DECIDE : local_rows_override;
        PetscInt local_cols = PETSC_DECIDE; // Keep default partitioning for columns

        ierr = MatSetSizes(*outMat, 
                        local_rows,      // Use local_rows_override for local rows
                        local_cols,      // Use PETSC_DECIDE for local columns
                        rows, 
                        cols); CHKERRQ(ierr);
        
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
        PetscInt rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Global start indices for each block, based on your implicit ordering:
        // Global ordering: [U block] [P block] [D block]
        const PetscInt u_start_global = 0;
        const PetscInt p_start_global = (PetscInt)m_nnu;
        const PetscInt d_start_global = (PetscInt)m_nni; // m_nni is assumed to be u_end + p_end

        // --- Helper function to determine ownership range for a block ---
        auto get_block_ownership = [&](PetscInt global_size, PetscInt& rstart, PetscInt& rend) {
            if (global_size == 0) {
                rstart = 0;
                rend = 0;
                return (PetscErrorCode)0;
            }
            Vec temp_vec = nullptr;
            ierr = VecCreate(comm, &temp_vec); CHKERRQ(ierr);
            ierr = VecSetSizes(temp_vec, PETSC_DECIDE, global_size); CHKERRQ(ierr);
            ierr = VecSetUp(temp_vec); CHKERRQ(ierr); // Ensure it's partitioned
            ierr = VecGetOwnershipRange(temp_vec, &rstart, &rend); CHKERRQ(ierr);
            ierr = VecDestroy(&temp_vec); CHKERRQ(ierr);
            return ierr;
        };

        // --- 1. Partition U DOFs (Global Size m_nnu) ---
        PetscInt u_rstart, u_rend;
        ierr = get_block_ownership((PetscInt)m_nnu, u_rstart, u_rend); CHKERRABORT(comm, ierr);
        ierr = ISCreateStride(comm,                
                            u_rend - u_rstart,              // Local size
                            u_start_global + u_rstart,      // Starting global index (e.g., 0 + 0)
                            1, &m_IS_u); CHKERRABORT(comm, ierr);

        // --- 2. Partition P DOFs (Global Size m_nnp = m_nni - m_nnu) ---
        PetscInt p_rstart, p_rend;
        PetscInt m_nnp = (PetscInt)m_nni - (PetscInt)m_nnu;
        ierr = get_block_ownership(m_nnp, p_rstart, p_rend); CHKERRABORT(comm, ierr);
        ierr = ISCreateStride(comm,                
                            p_rend - p_rstart,              // Local size
                            p_start_global + p_rstart,      // Starting global index (m_nnu + rstart)
                            1, &m_IS_p); CHKERRABORT(comm, ierr);

        // --- 3. Partition D DOFs (Global Size m_nnd) ---
        PetscInt d_rstart, d_rend;
        ierr = get_block_ownership((PetscInt)m_nnd, d_rstart, d_rend); CHKERRABORT(comm, ierr);
        ierr = ISCreateStride(comm,                
                            d_rend - d_rstart,              // Local size
                            d_start_global + d_rstart,      // Starting global index (m_nni + rstart)
                            1, &m_IS_d); CHKERRABORT(comm, ierr);
        
        // --- Sanity checks and Scatter Creation (Retain original logic, using new IS) ---
        PetscInt local_u, local_d, local_p;
        ISGetLocalSize(m_IS_u, &local_u);
        ISGetLocalSize(m_IS_d, &local_d); // THIS IS NOW NON-ZERO ON RANK 0!
        ISGetLocalSize(m_IS_p, &local_p);


        // Create tmp seq vectors sized to the local block counts for scatters
        Vec tmp_u = nullptr, tmp_d = nullptr;
        ierr = VecCreateSeq(PETSC_COMM_SELF, local_u, &tmp_u); CHKERRABORT(comm, ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, local_d, &tmp_d); CHKERRABORT(comm, ierr);

        // create global dummy vector so the scatters know global layout
        Vec x_global_dummy = nullptr;
        ierr = VecCreateMPI(comm, PETSC_DECIDE, (PetscInt)m_ndof, &x_global_dummy); CHKERRABORT(comm, ierr);

        // Create the scatters (now using the correctly partitioned m_IS_d)
        ierr = VecScatterCreate(tmp_u, NULL, x_global_dummy, m_IS_u, &m_scatter_u); CHKERRABORT(comm, ierr);
        ierr = VecScatterCreate(tmp_d, NULL, x_global_dummy, m_IS_d, &m_scatter_d); CHKERRABORT(comm, ierr);

        VecDestroy(&tmp_u);
        VecDestroy(&tmp_d);
        VecDestroy(&x_global_dummy);
    }

    // placeholder flag used earlier
    bool m_changed = false;
};

} // namespace GooseFEM

#endif
