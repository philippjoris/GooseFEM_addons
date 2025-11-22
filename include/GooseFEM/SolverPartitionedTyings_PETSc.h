#ifndef GOOSEFEM_SOLVERPARTITIONEDTYINGS_PETSC_H
#define GOOSEFEM_SOLVERPARTITIONEDTYINGS_PETSC_H

#include <petscksp.h>
#include <stdexcept>
#include <vector>
#include <Eigen/Sparse> // Used for inputting Cdu/Cdp from Eigen
#include "config.h" // For GOOSEFEM_ASSERT, array_type
#include "MatrixPartitionedTyings_PETSc.h"

namespace GooseFEM {

class SolverPartitionedTyings_PETSc {
private:
    KSP m_ksp = nullptr;
    bool m_factor = true;

    // Destroy KSP safely
    void destroyKSP() {
        if (m_ksp) {
            KSPDestroy(&m_ksp);
            m_ksp = nullptr;
        }
    }

    PetscErrorCode CreatePetscVecFromArray(const xt::pytensor<double, 2>& arr, Vec* outVec)
    {
        PetscErrorCode ierr;
        PetscInt n = arr.shape(0) * arr.shape(1); // total number of DOFs
        PetscScalar* data = const_cast<PetscScalar*>(arr.data()); // PETSc API is non-const

        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, data, outVec); CHKERRQ(ierr);
        return ierr;
    }

public:
    // SolverPartitionedTyings_PETSc() = default;
    SolverPartitionedTyings_PETSc(){
        PetscErrorCode ierr;

        // Create and setup KSP (once)
        ierr = KSPCreate(PETSC_COMM_WORLD, &m_ksp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = KSPSetType(m_ksp, KSPCG); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        PC pc;
        ierr = KSPGetPC(m_ksp, &pc); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PCSetType(pc, PCGAMG); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = KSPSetTolerances(m_ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = KSPSetFromOptions(m_ksp); CHKERRABORT(PETSC_COMM_WORLD, ierr);        
    }

    ~SolverPartitionedTyings_PETSc() { destroyKSP(); }

    // ---- Factorization step ----
    void factorize(MatrixPartitionedTyings_PETSc& A)
    {
        if (!A.m_changed && !m_factor && m_ksp)
            return;

        PetscErrorCode ierr;
        MatReuse reuse_flag = MAT_INITIAL_MATRIX;

        if (A.m_Aud != nullptr) {
            reuse_flag = MAT_REUSE_MATRIX;
        }

        // Clean up old condensed matrices & KSP
        // if (A.m_ACuu) { MatDestroy(&A.m_ACuu); A.m_ACuu = nullptr; }
        // if (A.m_ACup) { MatDestroy(&A.m_ACup); A.m_ACup = nullptr; }
        // destroyKSP();

        // --- Compute transpose of C_du once ---
        PetscInt rank; 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        if (!A.m_Auu) {
            ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_u, MAT_INITIAL_MATRIX, &A.m_Auu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        else {
             ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_u, MAT_REUSE_MATRIX, &A.m_Auu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        if (!A.m_Aud) {
            ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_d, MAT_INITIAL_MATRIX, &A.m_Aud); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        else {
             ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_d, MAT_REUSE_MATRIX, &A.m_Aud); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        if (!A.m_Adu) {
            ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_u, MAT_INITIAL_MATRIX, &A.m_Adu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        else {
             ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_u, MAT_REUSE_MATRIX, &A.m_Adu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        if (!A.m_Add) {
            ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_d, MAT_INITIAL_MATRIX, &A.m_Add); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        else {
             ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_d, MAT_REUSE_MATRIX, &A.m_Add); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        if (!A.m_Aup) {
            ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_p, MAT_INITIAL_MATRIX, &A.m_Aup); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        else {
             ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_p, MAT_REUSE_MATRIX, &A.m_Aup); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        if (!A.m_Adp) {
            ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_p, MAT_INITIAL_MATRIX, &A.m_Adp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }
        else {
             ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_p,MAT_REUSE_MATRIX, &A.m_Adp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        }                
                                
        
        
        // --- Compute condensed matrix A'_uu = A_uu + A_ud*C_du + C_du^T*A_du + C_du^T*A_dd*C_du ---
        Mat temp1 = nullptr, temp2 = nullptr, temp3 = nullptr, inter = nullptr;
        ierr = MatMatMult(A.m_Aud, A.m_Cdu, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp1); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(A.m_Cud, A.m_Adu, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp2); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(A.m_Cud, A.m_Add, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &inter); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(inter, A.m_Cdu, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp3); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        
        Mat ACuu = nullptr, ACup = nullptr;   
        ierr = MatDuplicate(A.m_Auu, MAT_COPY_VALUES, &ACuu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(ACuu, 1.0, temp1, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(ACuu, 1.0, temp2, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(ACuu, 1.0, temp3, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- Compute condensed coupling matrix A'_up = A_up + A_ud*C_dp + C_du^T*A_dp + C_du^T*A_dd*C_dp ---     

        Mat tempA1 = nullptr, tempA2 = nullptr, tempA3 = nullptr, inter2 = nullptr;
        ierr = MatMatMult(A.m_Aud, A.m_Cdp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tempA1); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(A.m_Cud, A.m_Adp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tempA2); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(A.m_Cud, A.m_Add, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &inter2); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(inter2, A.m_Cdp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tempA3); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = MatDuplicate(A.m_Aup, MAT_COPY_VALUES, &ACup); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(ACup, 1.0, tempA1, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(ACup, 1.0, tempA2, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(ACup, 1.0, tempA3, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- Set up KSP solver on A'_uu ---
        // ierr = KSPCreate(PETSC_COMM_WORLD, &m_ksp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = KSPSetOperators(m_ksp, ACuu, ACuu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // ierr = KSPSetType(m_ksp, KSPCG); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // PC pc;
        // ierr = KSPGetPC(m_ksp, &pc); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // ierr = PCSetType(pc, PCGAMG); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // ierr = KSPSetTolerances(m_ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // ierr = KSPSetFromOptions(m_ksp); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- Cleanup temporaries ---
        MatDestroy(&ACup); MatDestroy(&ACuu);
        MatDestroy(&temp1); MatDestroy(&temp2); MatDestroy(&temp3); MatDestroy(&inter);
        MatDestroy(&tempA1); MatDestroy(&tempA2); MatDestroy(&tempA3); MatDestroy(&inter2);

        m_factor   = false;
        A.m_changed = false;
    }

    template <class T>
    void solve(MatrixPartitionedTyings_PETSc& A, const T& b_py, T& x_py)
    {
        PetscErrorCode ierr;
        
        Vec b;
        PetscInt n_global = (PetscInt)b_py.size();
        // ierr = VecCreateSeq(PETSC_COMM_SELF, n_b, &b); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        //ierr = VecPlaceArray(b, b_py.data()); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n_global, &b); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- 2. Create global nodal vector x of size m_ndof ---
        Vec x;
        ierr = VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n_global, &x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // PetscInt n = (PetscInt)x_py.size();
        // ierr = VecCreateSeq(PETSC_COMM_SELF, n, &x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        // ierr = VecPlaceArray(x, x_py.data()); CHKERRABORT(PETSC_COMM_WORLD, ierr);


        // --- 3. Fill x with values from x_py at correct prescribed DOF indices ---
        PetscInt rstart, rend;
        ierr = VecGetOwnershipRange(x, &rstart, &rend); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- 2. Initialize local assembly arrays ---
        std::vector<PetscInt> idx_x, idx_b;
        std::vector<PetscScalar> vals_x, vals_b;

        // --- 3. Iterate over the global DOFs, checking ownership ---
        for (size_t m = 0; m < A.m_nnode; ++m) {
            for (size_t i = 0; i < A.m_ndim; ++i) {
                PetscInt gdof = (PetscInt)A.m_dofs(m,i);
                
                // **CRITICAL CHECK:** Only process global DOFs that this rank owns.
                if (gdof >= rstart && gdof < rend) {
                    
                    // Assemble x (only X_p DOFs)
                    // X_p are indices [A.m_nnu, A.m_nni)
                    if (gdof >= (PetscInt)A.m_nnu && gdof < (PetscInt)A.m_nni) {
                        idx_x.push_back(gdof);
                        vals_x.push_back(x_py(m,i));
                    }
                    
                    // Assemble b (All DOFs are included here)
                    idx_b.push_back(gdof);
                    vals_b.push_back(b_py(m,i));
                }
            }
        }


        ierr = VecSetValues(x, idx_x.size(), idx_x.data(), vals_x.data(), INSERT_VALUES); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyBegin(x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyEnd(x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecSetValues(b, idx_b.size(), idx_b.data(), vals_b.data(), INSERT_VALUES); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyBegin(b); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyEnd(b); CHKERRABORT(PETSC_COMM_WORLD, ierr);        

        // Factorize (build condensed matrices if needed)
        this->factorize(A);    

        // Extract sub-vectors (views)
        Vec B_u, B_d, X_p;
        ierr = A.GetDofs_u(b, &B_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = A.GetDofs_d(b, &B_d); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = A.GetDofs_p(x, &X_p); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Condensed RHS: B_u' = B_u + C_ud * B_d
        Vec B_prime_u, rhs, temp;
        ierr = VecDuplicate(B_u, &B_prime_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecDuplicate(B_u, &rhs); CHKERRABORT(PETSC_COMM_WORLD, ierr); // Reuse the memory for RHS
        ierr = VecDuplicate(B_u, &temp); CHKERRABORT(PETSC_COMM_WORLD, ierr);        

        ierr = VecCopy(B_u, B_prime_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMultAdd(A.m_Cud, B_d, B_prime_u, B_prime_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);      

        // Compute RHS: rhs = B_prime_u - A_ACup * X_p
        ierr = MatMult(A.m_Aup, X_p, temp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecWAXPY(rhs, -1.0, temp, B_prime_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Solve condensed system: A_ACuu * X_u = rhs
        Vec X_u;
        ierr = VecDuplicate(rhs, &X_u); CHKERRABORT(PETSC_COMM_WORLD, ierr); 

        ierr = KSPSolve(m_ksp, rhs, X_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);   

        // Check convergence
        KSPConvergedReason reason;
        KSPGetConvergedReason(m_ksp, &reason);
        if (reason < 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                        "ERROR: PETSc solver did not converge in solve_nodevec_impl (reason %d)\n",
                        reason);
            throw std::runtime_error("PETSc KSP failed to converge in solve_nodevec_impl");
        }

        // Reconstruct dependent DOFs: X_d = C_du * X_u + C_dp * X_p
        Vec X_d;
        ierr = VecDuplicate(B_d, &X_d); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMult(A.m_Cdu, X_u, X_d); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMultAdd(A.m_Cdp, X_p, X_d, X_d); CHKERRABORT(PETSC_COMM_WORLD, ierr);  

        // Scatter full nodal vector
        A.scatter_solution(X_u, X_d, x);

        // --- MANDATORY: Retrieve Final Solution to x_py ---
        Vec x_seq_all;
        VecScatter scatter_all;
        VecScatterCreateToAll(x, &scatter_all, &x_seq_all); // EVERY rank receives full x
        VecScatterBegin(scatter_all, x, x_seq_all, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(scatter_all, x, x_seq_all, INSERT_VALUES, SCATTER_FORWARD);

        PetscMPIInt rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        PetscInt m_ndof;
        VecGetSize(x, &m_ndof);
        
        const PetscScalar *x_const;
        VecGetArrayRead(x_seq_all, &x_const);
        for (PetscInt i = 0; i < m_ndof; ++i) {
            x_py[i] = x_const[i]; // Now all ranks have x_py
        }
        VecRestoreArrayRead(x_seq_all, &x_const);
        VecScatterDestroy(&scatter_all);
        VecDestroy(&x_seq_all); 

        /* Vec x_seq = nullptr;
        VecScatter scatter;
        VecScatterCreateToZero(x, &scatter, &x_seq);

        VecScatterBegin(scatter, x, x_seq, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(scatter, x, x_seq, INSERT_VALUES, SCATTER_FORWARD);

        PetscMPIInt rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        PetscInt m_ndof;
        VecGetSize(x, &m_ndof);

        if (rank == 0) {
            const PetscScalar *x_const;
            VecGetArrayRead(x_seq, &x_const);
            for (PetscInt i = 0; i < m_ndof; ++i) {
                x_py[i] = x_const[i];
            }
            VecRestoreArrayRead(x_seq, &x_const);
        }

        VecScatterDestroy(&scatter);
        VecDestroy(&x_seq); */
        // ------------------------------------------------- */
        // 2. Destroy owned PETSc temporaries (ONCE)
        VecDestroy(&B_prime_u);
        VecDestroy(&rhs);
        VecDestroy(&X_u);
        VecDestroy(&X_d);
        VecDestroy(&temp);

        // 3. Restore subvector views
        ierr = A.RestoreDofs_u(b, B_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = A.RestoreDofs_d(b, B_d); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = A.RestoreDofs_p(x, X_p); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // 4. Destroy x and b
        ierr = VecDestroy(&x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecDestroy(&b); CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }

    // ---- Solving condensed system ----
    void solve_u(MatrixPartitionedTyings_PETSc& A,
             const array_type::tensor<double, 1>& b_u,
             const array_type::tensor<double, 1>& b_d,
             const array_type::tensor<double, 1>& x_p,
             array_type::tensor<double, 1>& x_u)
    {
        // ... ASSERT checks (kept as is) ...
        factorize(A);
        PetscErrorCode ierr;
        
        // --- Vector creation and initialization (using PETSC_COMM_SELF is correct) ---
        Vec B_u, B_d, X_p, X_u, Bp_u, rhs;
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, A.m_nnu, b_u.data(), &B_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, A.m_nnd, b_d.data(), &B_d); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, A.m_nnp, x_p.data(), &X_p); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF, A.m_nnu, &X_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecDuplicate(B_u, &Bp_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecDuplicate(B_u, &rhs); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        
        // b'_u = b_u + C_du^T * b_d
        Vec tmp = nullptr; ierr = VecDuplicate(Bp_u, &tmp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecCopy(B_u, Bp_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        
        ierr = MatMult(A.m_Cud, B_d, tmp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAXPY(Bp_u, 1.0, tmp); CHKERRABORT(PETSC_COMM_WORLD, ierr); // Bp_u is now B'_u

        // Compute RHS: rhs = B'_u - A'_up * X_p
        // Use the pre-computed condensed matrix A.m_ACup
        ierr = MatMult(A.m_Aup, X_p, rhs); CHKERRABORT(PETSC_COMM_WORLD, ierr); // rhs = A'_up * X_p
        ierr = VecAYPX(rhs, -1.0, Bp_u); CHKERRABORT(PETSC_COMM_WORLD, ierr); // rhs = Bp_u - rhs = B'_u - A'_up * X_p

        // Solve A'_{uu} X_u = RHS
        ierr = KSPSolve(m_ksp, rhs, X_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // ... Copy result to x_u (kept as is) ...

        // Cleanup
        VecDestroy(&B_u); VecDestroy(&B_d); VecDestroy(&X_p); VecDestroy(&X_u);
        VecDestroy(&Bp_u); VecDestroy(&tmp); VecDestroy(&rhs);
    }

    void setTolerance(double tol) {
        if (m_ksp) KSPSetTolerances(m_ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    }

    void setMaxIterations(int maxIter) {
        if (m_ksp) KSPSetTolerances(m_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, maxIter);
    }
};
} // namespace GooseFEM

#endif // GOOSEFEM_SOLVERPARTITIONEDTYINGS_PETSC_H