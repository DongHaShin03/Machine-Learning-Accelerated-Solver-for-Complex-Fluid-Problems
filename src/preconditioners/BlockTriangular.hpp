#ifndef BLOCK_TRIANGULAR_HPP
#define BLOCK_TRIANGULAR_HPP

#include "NavierStokesPreconditioner.hpp"

class BlockTriangular : public NavierStokesPreconditioner
{
    public:

        AssemblyFlags get_needed_matrices() const override {return {false, true};}
        void initialize(const RequiredMatrices &data) override
        {
            velocity_stiffness = data.velocity_stiffness;
            pressure_mass      = data.pressure_mass;
            B                  = data.B;

            preconditioner_velocity.initialize(*velocity_stiffness);
            preconditioner_pressure.initialize(*pressure_mass);
        }

        void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
        {
            SolverControl solver_control_velocity(1000, 1e-2 * src.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_velocity(solver_control_velocity);
            solver_gmres_velocity.solve(*velocity_stiffness, dst.block(0), src.block(0), preconditioner_velocity);

            tmp.reinit(src.block(1));
            B->vmult(tmp, dst.block(0));
            tmp.sadd(-1.0, src.block(1));

            SolverControl solver_control_pressure(1000, 1e-2 * src.block(1).l2_norm());
            SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);
            solver_cg_pressure.solve(*pressure_mass, dst.block(1), tmp, preconditioner_pressure);
        }

    private:
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;
        const TrilinosWrappers::SparseMatrix *pressure_mass;
        const TrilinosWrappers::SparseMatrix *B;

        TrilinosWrappers::PreconditionILU preconditioner_velocity;
        TrilinosWrappers::PreconditionILU preconditioner_pressure;

        mutable TrilinosWrappers::MPI::Vector tmp;
};

#endif