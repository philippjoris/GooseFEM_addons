// Required headers (add to your file)
#include <petscmat.h>
#include <petscerror.h>
#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>
#include <limits>

// forward declaration for solver if needed
template <class> class SolverPartitionedTyings_PETSc;

class MatrixPartitionedTyings_PETSc {
private:
    // NOTE: PETSc types (Mat) are pointers under the hood; initialize to nullptr.
    Mat m_A = nullptr;    ///< global assembled matrix (all dofs)
    Mat m_Cdu = nullptr;  ///< tying matrix (dependent rows, unknown cols)
    Mat m_Cdp = nullptr;  ///< tying matrix (dependent rows, prescribed cols)
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
    template <class> friend class SolverPartitionedTyings_PETSc;

public:
    MatrixPartitionedTyings_PETSc() = default;

    MatrixPartitionedTyings_PETSc(
        const array_type::tensor<size_t, 2>& dofs,
        const Eigen::SparseMatrix<double>& Cdu,
        const Eigen::SparseMatrix<double>& Cdp
    ) {
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

        ierr = MatCreate(PETSC_COMM_WORLD, &m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetSizes(m_A, PETSC_DECIDE, PETSC_DECIDE,
                           (PetscInt)m_ndof, (PetscInt)m_ndof); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetType(m_A, MATAIJ); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatSetUp(m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        this->initialize_index_sets();
    }

    virtual ~MatrixPartitionedTyings_PETSc() {
        destroy_petsc_tyings_objects();
    }

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
        VecSet(x_global, 0.0);

        VecScatterBegin(m_scatter_u, X_u, x_global, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(m_scatter_u, X_u, x_global, INSERT_VALUES, SCATTER_FORWARD);

        VecScatterBegin(m_scatter_d, X_d, x_global, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(m_scatter_d, X_d, x_global, INSERT_VALUES, SCATTER_FORWARD);

        VecAssemblyBegin(x_global);
        VecAssemblyEnd(x_global);
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
        PetscInt nnz = (PetscInt)eigen_mat.nonZeros();

        // 1. Create and Setup Mat
        ierr = MatCreate(PETSC_COMM_WORLD, outMat); CHKERRQ(ierr);
        ierr = MatSetSizes(*outMat, PETSC_DECIDE, PETSC_DECIDE, rows, cols); CHKERRQ(ierr);
        ierr = MatSetType(*outMat, MATAIJ); CHKERRQ(ierr);
        
        // 2. Pre-allocation (Essential for performance)
        PetscInt local_rows = PETSC_DECIDE;
        ierr = MatSetSizes(*outMat, local_rows, PETSC_DECIDE, rows, cols); CHKERRQ(ierr);
        
        // Simplified preallocation: estimate based on average non-zeros per row.
        PetscInt max_nonzeros_per_row_estimate = nnz / rows + 1; 
        ierr = MatSeqAIJSetPreallocation(*outMat, max_nonzeros_per_row_estimate, NULL); CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(*outMat, max_nonzeros_per_row_estimate, NULL, max_nonzeros_per_row_estimate, NULL); CHKERRQ(ierr);
        
        ierr = MatSetUp(*outMat); CHKERRQ(ierr);
        
        // 3. Extract and Insert Data (Bulk Operation)
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        if (rank == 0) { 
            std::vector<PetscInt> i_indices;
            std::vector<PetscInt> j_indices;
            std::vector<PetscScalar> values;

            i_indices.reserve(nnz);
            j_indices.reserve(nnz);
            values.reserve(nnz);

            // Iterate over Eigen matrix to fill COO arrays
            for (int k = 0; k < eigen_mat.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(eigen_mat, k); it; ++it) {
                    i_indices.push_back((PetscInt)it.row());
                    j_indices.push_back((PetscInt)it.col());
                    values.push_back((PetscScalar)it.value());
                }
            }

            PetscInt one = 1;
            for (PetscInt k = 0; k < nnz; ++k) {
                PetscInt i = i_indices[k];
                PetscInt j = j_indices[k];
                PetscScalar v = values[k];
                ierr = MatSetValues(*outMat, one, &i, one, &j, &v, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
        
        // 4. Assembly (Collective Operation)
        ierr = MatAssemblyBegin(*outMat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*outMat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        return 0;
    }

    void clear()
    {
        if (m_A) {
            PetscErrorCode ierr = MatZeroEntries(m_A); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
    }

    void initialize_index_sets(MPI_Comm comm = PETSC_COMM_WORLD) {
        PetscErrorCode ierr;

        std::vector<PetscInt> iu(m_nnu), id(m_nnd), ip(m_nnp);
        for (size_t k = 0; k < m_nnu; ++k) iu[k] = (PetscInt)m_iiu(k);
        for (size_t k = 0; k < m_nnd; ++k) id[k] = (PetscInt)m_iid(k);
        for (size_t k = 0; k < m_nnp; ++k) ip[k] = (PetscInt)m_iip(k);

        ierr = ISCreateGeneral(comm, iu.size(), iu.data(), PETSC_COPY_VALUES, &m_IS_u); CHKERRABORT(comm, ierr);
        ierr = ISCreateGeneral(comm, id.size(), id.data(), PETSC_COPY_VALUES, &m_IS_d); CHKERRABORT(comm, ierr);
        ierr = ISCreateGeneral(comm, ip.size(), ip.data(), PETSC_COPY_VALUES, &m_IS_p); CHKERRABORT(comm, ierr);

        Vec tmp_u, tmp_d, x_global_dummy;
        VecCreateSeq(PETSC_COMM_SELF, (PetscInt)m_nnu, &tmp_u);
        VecCreateSeq(PETSC_COMM_SELF, (PetscInt)m_nnd, &tmp_d);
        VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, (PetscInt)m_ndof, &x_global_dummy);

        ierr = VecScatterCreate(tmp_u, NULL, x_global_dummy, m_IS_u, &m_scatter_u); CHKERRABORT(comm, ierr);
        ierr = VecScatterCreate(tmp_d, NULL, x_global_dummy, m_IS_d, &m_scatter_d); CHKERRABORT(comm, ierr);

        VecDestroy(&tmp_u);
        VecDestroy(&tmp_d);
        VecDestroy(&x_global_dummy);
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

        // For safety and variable size, use vectors
        for (ptrdiff_t e = 0; e < (ptrdiff_t)nelem; ++e) {
            size_t size = nnodes_per_elem * m_ndim;
            std::vector<PetscInt> idx(size);
            std::vector<PetscScalar> vals(size * size);

            // fill global DOF indices
            for (ptrdiff_t n = 0; n < (ptrdiff_t)nnodes_per_elem; ++n) {
                for (ptrdiff_t j = 0; j < (ptrdiff_t)m_ndim; ++j) {
                    idx[n * m_ndim + j] = static_cast<PetscInt>(m_dofs(conn_elem(e, n), j));
                }
            }

            // copy element matrix into contiguous vals (row-major)
            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    vals[i * size + j] = static_cast<PetscScalar>(elemmat(e, i, j));
                }
            }

            PetscErrorCode ierr = MatSetValues(m_A,
                                              static_cast<PetscInt>(size), idx.data(),
                                              static_cast<PetscInt>(size), idx.data(),
                                              vals.data(),
                                              ADD_VALUES);
            CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }

        // Note: do not call MatAssembly here; call finalize_impl once after all assemble_impl calls.
    }

    void finalize(bool stabilize)
    {
        PetscErrorCode ierr;
        ierr = MatAssemblyBegin(m_A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(m_A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        if (stabilize) {
            // Example stabilization: add tiny diagonal to zero diagonals.
            // Get diagonal vector
            Vec diag;
            ierr = MatGetDiagonal(m_A, &diag); CHKERRQ(ierr);
            PetscScalar min_diag = PETSC_MAX_REAL;
            PetscScalar v;
            PetscInt size;
            ierr = VecGetSize(diag, &size); CHKERRQ(ierr);

            // iterate to find min nonzero diagonal
            const PetscScalar *array = nullptr;
            ierr = VecGetArrayRead(diag, &array); CHKERRQ(ierr);
            for (PetscInt i = 0; i < size; ++i) {
                v = array[i];
                if (std::abs(v) > 1e-12 && std::abs(v) < min_diag) min_diag = std::abs(v);
            }
            ierr = VecRestoreArrayRead(diag, &array); CHKERRQ(ierr);

            if (min_diag < PETSC_MAX_REAL) {
                PetscScalar spring = 1e-3 * min_diag;
                // Create a vector with spring on places where diagonal is ~0
                Vec adddiag;
                ierr = VecDuplicate(diag, &adddiag); CHKERRQ(ierr);
                ierr = VecSet(adddiag, 0.0); CHKERRQ(ierr);

                // mark small diag positions
                ierr = VecGetArray(adddiag, &array); CHKERRQ(ierr); // reuse pointer variable name, restored below
                // Can't write through array from VecGetArrayRead; so get writable array separately
                PetscScalar *warray = nullptr;
                ierr = VecGetArray(adddiag, &warray); CHKERRQ(ierr);
                ierr = VecGetArrayRead(diag, &array); CHKERRQ(ierr);
                for (PetscInt i = 0; i < size; ++i) {
                    if (std::abs(array[i]) < 1e-12) warray[i] = spring;
                }
                ierr = VecRestoreArrayRead(diag, &array); CHKERRQ(ierr);
                ierr = VecRestoreArray(adddiag, &warray); CHKERRQ(ierr);

                ierr = MatDiagonalSet(m_A, adddiag, ADD_VALUES); CHKERRQ(ierr);
                ierr = VecDestroy(&adddiag); CHKERRQ(ierr);
            }

            ierr = VecDestroy(&diag); CHKERRQ(ierr);
        }

        m_changed = true;
    }

    // placeholder flag used earlier
    bool m_changed = false;
};