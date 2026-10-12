#ifndef BLOCK_DIAGONAL_AMG_HPP
#define BLOCK_DIAGONAL_AMG_HPP

#include <deal.II/base/config.h>

#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>

#ifndef DEAL_II_WITH_PETSC
#  error "deal.II must be configured with PETSc."
#endif
#ifndef DEAL_II_PETSC_WITH_HYPRE
#  error "BoomerAMG requires PETSc configured with hypre."
#endif
#ifdef DEAL_II_PETSC_WITH_COMPLEX
#  error "PETSc must be configured with real scalars."
#endif

// Block diagonal preconditioner of the paper (Sec. 6.5):
//
//   P^{-1} = diag(A, M_p / nu)^{-1}
//
// - A^{-1} is approximated by a single V-cycle of BoomerAMG (hypre, via PETSc);
//   theta is the strong threshold of the paper.
// - (M_p / nu)^{-1} is approximated by one Jacobi step.
//
// Both blocks are fixed, linear, symmetric positive definite operators, as
// required by MINRES.
class BlockDiagonalAMG
{
    public:
        void initialize(const dealii::PETScWrappers::MPI::SparseMatrix &velocity_stiffness,
                        const dealii::PETScWrappers::MPI::SparseMatrix &pressure_mass,
                        const double theta)
        {
            // All BoomerAMG parameters are left at their defaults (max_iter = 1,
            // i.e. one V-cycle per application), except:
            dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
            data.symmetric_operator = true;  // symmetric smoothing: needed by MINRES
            data.strong_threshold   = theta; // default 0.25

            amg_velocity.initialize(velocity_stiffness, data);
            jacobi_pressure.initialize(pressure_mass);
        }

        void vmult(dealii::PETScWrappers::MPI::BlockVector &dst, const dealii::PETScWrappers::MPI::BlockVector &src) const
        {
            amg_velocity.vmult(dst.block(0), src.block(0));
            jacobi_pressure.vmult(dst.block(1), src.block(1));
        }

    private:
        dealii::PETScWrappers::PreconditionBoomerAMG amg_velocity;
        dealii::PETScWrappers::PreconditionJacobi    jacobi_pressure;
};

#endif
