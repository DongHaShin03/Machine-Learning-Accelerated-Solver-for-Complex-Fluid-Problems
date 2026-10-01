#ifndef BLOCK_DIAGONAL_HPP
#define BLOCK_DIAGONAL_HPP

#include "NavierStokesPreconditioner.hpp"

class BlockDiagonal : public NavierStokesPreconditioner
{
    public:

        AssemblyFlags get_needed_matrices() const override {return {false, true};}
        void initialize(const RequiredMatrices &data) override
        {
            velocity_stiffness = data.velocity_stiffness;
            pressure_mass      = data.pressure_mass;

            preconditioner_velocity.initialize(*velocity_stiffness);
            preconditioner_pressure.initialize(*pressure_mass);
        }

        void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
        {
            SolverControl solver_control_velocity(1000, 1e-2 * src.block(0).l2_norm());
            SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(solver_control_velocity);
            solver_cg_velocity.solve(*velocity_stiffness, dst.block(0), src.block(0), preconditioner_velocity);

            SolverControl solver_control_pressure(1000, 1e-2 * src.block(1).l2_norm());
            SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(solver_control_pressure);
            solver_cg_pressure.solve(*pressure_mass, dst.block(1), src.block(1), preconditioner_pressure);
        }

    private:
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;
        const TrilinosWrappers::SparseMatrix *pressure_mass;

        TrilinosWrappers::PreconditionILU preconditioner_velocity;
        TrilinosWrappers::PreconditionILU preconditioner_pressure;
};

#endif