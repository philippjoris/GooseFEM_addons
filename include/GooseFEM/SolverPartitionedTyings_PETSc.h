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
    SolverPartitionedTyings_PETSc() = default;
    ~SolverPartitionedTyings_PETSc() { destroyKSP(); }

    // ---- Factorization step ----
    void factorize(MatrixPartitionedTyings_PETSc& A)
    {
        PetscPrintf(PETSC_COMM_WORLD, "\n--- Inside factorize function ---\n");
        if (!A.m_changed && !m_factor && m_ksp)
            return;

        PetscErrorCode ierr;
        MatReuse reuse_flag = MAT_INITIAL_MATRIX;

        if (A.m_Aud != nullptr) {
            reuse_flag = MAT_REUSE_MATRIX;
        }

        // Clean up old condensed matrices & KSP
        if (A.m_ACuu) { MatDestroy(&A.m_ACuu); A.m_ACuu = nullptr; }
        if (A.m_ACup) { MatDestroy(&A.m_ACup); A.m_ACup = nullptr; }
        destroyKSP();


        PetscPrintf(PETSC_COMM_WORLD, "\n--- Submatrices were extracted ---\n");
        // --- Compute transpose of C_du once ---
        Mat C_du_T = nullptr;
        ierr = MatTranspose(A.m_Cdu, MAT_INITIAL_MATRIX, &C_du_T); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        PetscInt rank; 
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        PetscInt rows_Aud, cols_Aud, rows_Adu, cols_Adu, rows_Cdu, cols_Cdu, rows_Cdu_T, cols_Cdu_T, size_d_is;

        ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_u, MAT_INITIAL_MATRIX, &A.m_Auu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_d, MAT_INITIAL_MATRIX, &A.m_Aud); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_u, MAT_INITIAL_MATRIX, &A.m_Adu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_d, MAT_INITIAL_MATRIX, &A.m_Add); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Get global dimensions of the matrices being multiplied (A_ud * A.m_Cdu)
        ierr = MatGetSize(A.m_Aud, &rows_Aud, &cols_Aud); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatGetSize(A.m_Cdu, &rows_Cdu, &cols_Cdu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatGetSize(A.m_Adu, &rows_Adu, &cols_Adu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatGetSize(C_du_T, &rows_Cdu_T, &cols_Cdu_T); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Get global size of the index set that defines the column count (A.m_IS_d should define A_ud columns)
        ierr = ISGetSize(A.m_IS_d, &size_d_is); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "\n--- Parallel Debugging Dimensions ---\n");
            
            // Check A_ud (Matrix A in the error: 5994x1986)
            PetscPrintf(PETSC_COMM_WORLD, "A_ud Dimensions (Rows x Cols): %D x %D \n", rows_Aud, cols_Aud);
            
            // Check A.m_Cdu (Matrix B in the error: 993x2997)
            PetscPrintf(PETSC_COMM_WORLD, "Cdu Dimensions (Rows x Cols): %D x %D \n", rows_Cdu, cols_Cdu);

            // Check C_du_T
            PetscPrintf(PETSC_COMM_WORLD, "A_du Dimensions (Rows x Cols): %D x %D \n", rows_Adu, cols_Adu);

            // Check A_du
            PetscPrintf(PETSC_COMM_WORLD, "C_du_T Dimensions (Rows x Cols): %D x %D \n", rows_Cdu_T, cols_Cdu_T);            
            
            // Check the d-partition size (Cdu rows should match A_ud columns)
            PetscPrintf(PETSC_COMM_WORLD, "A.m_IS_d Global Size (Should be 993): %D\n", size_d_is);
            PetscPrintf(PETSC_COMM_WORLD, "-----------------------------------\n\n");
        }

        PetscInt lr, lc;
        MatGetLocalSize(A.m_Aud, &lr, &lc);
        PetscPrintf(PETSC_COMM_WORLD, "Rank %d: A_ud local size = %D x %D\n", rank, lr, lc);

        MatGetLocalSize(A.m_Cdu, &lr, &lc);
        PetscPrintf(PETSC_COMM_WORLD, "Rank %d: A.m_Cdu local size = %D x %D\n", rank, lr, lc);

        
        // --- Compute condensed matrix A'_uu = A_uu + A_ud*C_du + C_du^T*A_du + C_du^T*A_dd*C_du ---
        Mat temp1 = nullptr, temp2 = nullptr, temp3 = nullptr, inter = nullptr;
        ierr = MatMatMult(A.m_Aud, A.m_Cdu, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp1); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(C_du_T, A.m_Adu, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp2); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(C_du_T, A.m_Add, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &inter); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(inter, A.m_Cdu, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp3); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        MatGetLocalSize(temp1, &lr, &lc);
        PetscPrintf(PETSC_COMM_WORLD, "Rank %d: temp1 local size = %D x %D\n", rank, lr, lc);

        MatGetLocalSize(A.m_Auu, &lr, &lc);
        PetscPrintf(PETSC_COMM_WORLD, "Rank %d: A.m_Auu local size = %D x %D\n", rank, lr, lc);


        ierr = MatDuplicate(A.m_Auu, MAT_COPY_VALUES, &A.m_ACuu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        MatGetLocalSize(A.m_ACuu, &lr, &lc);
        PetscPrintf(PETSC_COMM_WORLD, "Rank %d: A.m_ACuu local size = %D x %D\n", rank, lr, lc);
        ierr = MatAXPY(A.m_ACuu, 1.0, temp1, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(A.m_ACuu, 1.0, temp2, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(A.m_ACuu, 1.0, temp3, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- Compute condensed coupling matrix A'_up = A_up + A_ud*C_dp + C_du^T*A_dp + C_du^T*A_dd*C_dp ---
        Mat A_up = nullptr, A_dp = nullptr;
        ierr = MatCreateSubMatrix(A.m_A, A.m_IS_u, A.m_IS_p, MAT_INITIAL_MATRIX, &A_up); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatCreateSubMatrix(A.m_A, A.m_IS_d, A.m_IS_p, MAT_INITIAL_MATRIX, &A_dp); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        Mat tempA1 = nullptr, tempA2 = nullptr, tempA3 = nullptr, inter2 = nullptr;
        ierr = MatMatMult(A.m_Aud, A.m_Cdp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tempA1); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(C_du_T, A_dp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tempA2); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(C_du_T, A.m_Add, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &inter2); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMatMult(inter2, A.m_Cdp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tempA3); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        ierr = MatDuplicate(A_up, MAT_COPY_VALUES, &A.m_ACup); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(A.m_ACup, 1.0, tempA1, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(A.m_ACup, 1.0, tempA2, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatAXPY(A.m_ACup, 1.0, tempA3, DIFFERENT_NONZERO_PATTERN); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- Set up KSP solver on A'_uu ---
        ierr = KSPCreate(PETSC_COMM_WORLD, &m_ksp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = KSPSetOperators(m_ksp, A.m_ACuu, A.m_ACuu); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = KSPSetType(m_ksp, KSPCG); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        PC pc;
        ierr = KSPGetPC(m_ksp, &pc); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = PCSetType(pc, PCGAMG); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = KSPSetTolerances(m_ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = KSPSetFromOptions(m_ksp); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- Cleanup temporaries ---
        MatDestroy(&A_up); MatDestroy(&A_dp);
        MatDestroy(&C_du_T);
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
        PetscInt n_b = (PetscInt)b_py.size();
        ierr = VecCreateSeq(PETSC_COMM_SELF, n_b, &b); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecPlaceArray(b, b_py.data()); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // --- 2. Create global nodal vector x of size m_ndof ---
        Vec x;
        PetscInt n = (PetscInt)x_py.size();
        ierr = VecCreateSeq(PETSC_COMM_SELF, n, &x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecPlaceArray(x, x_py.data()); CHKERRABORT(PETSC_COMM_WORLD, ierr);


        // --- 3. Fill x with values from x_py at correct prescribed DOF indices ---
        // Loop like Eigen: global DOF indices = m_dofs(m,i)
        std::vector<PetscInt> idx;          // global indices in PETSc
        std::vector<PetscScalar> vals;      // corresponding values

        for (size_t m = 0; m < A.m_nnode; ++m) {
            for (size_t i = 0; i < A.m_ndim; ++i) {
                PetscInt gdof = (PetscInt)A.m_dofs(m,i);
                // Only insert prescribed DOFs (same condition as Eigen)
                if (gdof >= (PetscInt)A.m_nnu && gdof < (PetscInt)A.m_nni) {
                    idx.push_back(gdof);
                    vals.push_back(x_py(m,i));
                }
            }
        }

        std::vector<PetscInt> idx_b;
        std::vector<PetscScalar> vals_b;

        PetscScalar val_b;
        for (size_t m = 0; m < A.m_nnode; ++m) {
            for (size_t i = 0; i < A.m_ndim; ++i) {
                PetscInt gdof = (PetscInt)A.m_dofs(m,i);

                // 1. Unknown DOFs (X_u): 0 <= gdof < m_nnu
                if (gdof < (PetscInt)A.m_nnu) {
                    idx_b.push_back(gdof);
                    vals_b.push_back(b_py(m,i));
                } 
                else if (gdof < (PetscInt)A.m_nni) {
                    idx_b.push_back(gdof);
                    vals_b.push_back(b_py(m,i));
                }
                else { 
                    PetscInt local_d = gdof - A.m_nni;
                    idx_b.push_back(gdof);
                    vals_b.push_back(b_py(m,i));
                }
            }
        }


        ierr = VecSetValues(x, idx.size(), idx.data(), vals.data(), INSERT_VALUES); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyBegin(x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyEnd(x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecSetValues(b, idx_b.size(), idx_b.data(), vals_b.data(), INSERT_VALUES); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyBegin(b); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAssemblyEnd(b); CHKERRABORT(PETSC_COMM_WORLD, ierr);        

        // Factorize (build condensed matrices if needed)
        this->factorize(A);
        

        // ---- PRINT STATEMENT ----
        // PetscScalar max_val;
        // VecMax(x, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of full vector du = %g\n", PetscRealPart(max_val));
        // ---- PRINT STATEMENT ----        

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

        // ---- PRINT STATEMENT ----        
        // VecMax(X_p, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of asDofs_p(du) = %g\n", PetscRealPart(max_val));

        // VecMax(B_u, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of asDofs_u(fres) = %g\n", PetscRealPart(max_val));        

        // VecMax(B_d, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of asDofs_d(fres) = %g\n", PetscRealPart(max_val));

        // ---- PRINT STATEMENT ----        

        ierr = VecCopy(B_u, B_prime_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMultAdd(A.m_Cud, B_d, B_prime_u, B_prime_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // ---- PRINT STATEMENT ----     
        // VecMax(B_prime_u, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of B_u += A.m_Cud * B_d: %g\n", PetscRealPart(max_val));    
        // ---- PRINT STATEMENT ----            

        // Compute RHS: rhs = B_prime_u - A_ACup * X_p
        ierr = MatMult(A.m_ACup, X_p, temp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecWAXPY(rhs, -1.0, temp, B_prime_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // Solve condensed system: A_ACuu * X_u = rhs
        Vec X_u;
        ierr = VecDuplicate(rhs, &X_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // ---- PRINT STATEMENT ----     
        // VecMax(rhs, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of B_prime_u - A_ACup * X_p = %g\n", PetscRealPart(max_val));    
        // ---- PRINT STATEMENT ----     

        ierr = KSPSolve(m_ksp, rhs, X_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // ---- PRINT STATEMENT ----     
        // VecMax(X_u, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of du_new = %g\n", PetscRealPart(max_val));  
        // ---- PRINT STATEMENT ----     

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

        // ---- PRINT STATEMENT ----     
        // VecMax(X_d, NULL, &max_val);
        // PetscPrintf(PETSC_COMM_WORLD, "Max of x_d = %g\n", PetscRealPart(max_val));  
        // ---- PRINT STATEMENT ----     

        // Scatter full nodal vector
        A.scatter_solution(X_u, X_d, x);

        // Cleanup PETSc temporaries (Owned vectors)
        // 1. Reset array view for x
        ierr = VecResetArray(x); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecResetArray(b); CHKERRABORT(PETSC_COMM_WORLD, ierr);

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
        
        Mat C_du_T = nullptr; // Need to compute C_du^T only for RHS
        ierr = MatTranspose(A.m_Cdu, MAT_INITIAL_MATRIX, &C_du_T); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = MatMult(C_du_T, B_d, tmp); CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecAXPY(Bp_u, 1.0, tmp); CHKERRABORT(PETSC_COMM_WORLD, ierr); // Bp_u is now B'_u

        // Compute RHS: rhs = B'_u - A'_up * X_p
        // Use the pre-computed condensed matrix A.m_ACup
        ierr = MatMult(A.m_ACup, X_p, rhs); CHKERRABORT(PETSC_COMM_WORLD, ierr); // rhs = A'_up * X_p
        ierr = VecAYPX(rhs, -1.0, Bp_u); CHKERRABORT(PETSC_COMM_WORLD, ierr); // rhs = Bp_u - rhs = B'_u - A'_up * X_p

        // Solve A'_{uu} X_u = RHS
        ierr = KSPSolve(m_ksp, rhs, X_u); CHKERRABORT(PETSC_COMM_WORLD, ierr);

        // ... Copy result to x_u (kept as is) ...

        // Cleanup
        MatDestroy(&C_du_T);
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