#ifndef BLOCK_DIAGONAL_AMG_HPP
#define BLOCK_DIAGONAL_AMG_HPP

#include <deal.II/base/config.h>

#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

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
//   P^{-1} = diag(A, M)^{-1} = diag(A^{-1}, M^{-1})
//
// - A^{-1} is approximated by a single V-cycle of BoomerAMG (hypre, via PETSc);
//   theta is the strong threshold of the paper. This is the only approximation, as in the paper.
// - M is the pressure mass matrix (M_h of the paper, or M_p / nu, see Stokes). M^{-1} is applied
//   exactly, i.e. by CG preconditioned by Jacobi with a relative tolerance far below the one
//   of MINRES, so that it is a fixed linear operator.
//
// Both blocks are fixed, linear, symmetric positive definite operators, as
// required by MINRES.
class BlockDiagonalAMG
{
    public:
        void initialize(const dealii::PETScWrappers::MPI::SparseMatrix &velocity_stiffness,
                        const dealii::PETScWrappers::MPI::SparseMatrix &pressure_mass_,
                        const double theta)
        {
            // All BoomerAMG parameters are left at their defaults (max_iter = 1,
            // i.e. one V-cycle per application), except:
            dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
            data.symmetric_operator = true;  // symmetric smoothing: needed by MINRES
            data.strong_threshold   = theta; // default 0.25

            amg_velocity.initialize(velocity_stiffness, data);
            // deal.II only configures hypre: build the AMG hierarchy now, so that it is measured in the
            // setup time and not in the first application of the preconditioner inside MINRES.
            AssertThrow(PCSetUp(amg_velocity.get_pc()) == 0, dealii::ExcMessage("PCSetUp failed."));

            pressure_mass = &pressure_mass_;
            jacobi_pressure.initialize(pressure_mass_);
        }

        void vmult(dealii::PETScWrappers::MPI::BlockVector &dst, const dealii::PETScWrappers::MPI::BlockVector &src) const
        {
            amg_velocity.vmult(dst.block(0), src.block(0));

            dealii::SolverControl solver_control(1000, 1.0e-12 * src.block(1).l2_norm(), false, false);
            dealii::SolverCG<dealii::PETScWrappers::MPI::Vector> solver(solver_control);
            dst.block(1) = 0.0; // same initial guess at every application: the operator stays linear
            solver.solve(*pressure_mass, dst.block(1), src.block(1), jacobi_pressure);
        }

    private:
        dealii::PETScWrappers::PreconditionBoomerAMG amg_velocity;

        const dealii::PETScWrappers::MPI::SparseMatrix *pressure_mass = nullptr;
        dealii::PETScWrappers::PreconditionJacobi       jacobi_pressure;
};

#endif
