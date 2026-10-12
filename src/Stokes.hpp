#ifndef STOKES_HPP
#define STOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "preconditioners/BlockDiagonalAMG.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace dealii;

// Stationary Stokes problem of the paper (eq. 7, Sec. 6.5), Taylor-Hood Q2-Q1,
// solved with MINRES preconditioned by diag(1 V-cycle BoomerAMG on A, Jacobi on M_p/nu).
class Stokes
{
    public:
        static constexpr unsigned int dim = 2;

        using Value = std::function<double(const Point<dim> &p)>;
        using VectorField = std::function<Tensor<1, dim>(const Point<dim> &p)>;

        // Result of the solves with a given strong threshold theta.
        struct SolverResult
        {
            double       theta          = 0.0;
            bool         converged      = false;
            unsigned int iterations     = 0;   // N_it = min{k : ||r^(k)|| / ||r^(0)|| < 1e-8}
            double       time_amg_setup = 0.0; // mean over the repetitions [s]
            double       time_solve     = 0.0; // mean over the repetitions [s]
            double       time_solve_std = 0.0; // standard deviation [s]
        };

        Stokes
        (
            const unsigned int n_refinements_,
            const unsigned int degree_velocity_,
            const unsigned int degree_pressure_,
            const Value &nu_,
            const VectorField &f_
        )
        : n_refinements(n_refinements_)
        , degree_velocity(degree_velocity_)
        , degree_pressure(degree_pressure_)
        , nu(nu_)
        , f(f_)
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , mesh(MPI_COMM_WORLD)
        {}

        void setup();
        void assemble();
        // The matrix does not depend on theta: assemble once, solve for each theta.
        SolverResult solve(const double theta, const unsigned int n_repetitions = 1);
        // Approximate convergence factor of the last solve (paper, Sec. 5):
        //   rho^(k) = (||r^(k)|| / ||r^(0)||)^(1/k),   rho = rho^(N_it),
        // with r^(k) = b - K x^(k) and the euclidean norm. Without argument it returns rho.
        double compute_convergence_factor(const unsigned int k = numbers::invalid_unsigned_int) const;
        void output(const std::string &output_file_name = "output-stokes") const;

        double get_h() const {return h_max;}
        types::global_dof_index n_dofs() const {return dof_handler.n_dofs();}

        // Dirichlet data on the velocity, by boundary id. Must be set before setup().
        std::map<types::boundary_id, const Function<dim> *> dirichlet;

    protected:
        const unsigned int n_refinements;
        const unsigned int degree_velocity;
        const unsigned int degree_pressure;
        const Value nu; // viscosity
        const VectorField f;

        const unsigned int mpi_size;
        const unsigned int mpi_rank;
        ConditionalOStream pcout;

        parallel::fullydistributed::Triangulation<dim> mesh;
        double h_max = 0.0; // maximum cell diameter

        std::unique_ptr<FiniteElement<dim>> fe;
        std::unique_ptr<Quadrature<dim>> quadrature;

        // DoF handler.
        DoFHandler<dim> dof_handler;
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        std::vector<IndexSet> block_owned_dofs; // DoFs owned by current process in the velocity and pressure blocks.
        std::vector<IndexSet> block_relevant_dofs; // DoFs relevant to current process in the velocity and pressure blocks.

        // Dirichlet conditions (keep the system matrix symmetric).
        AffineConstraints<double> constraints;

        PETScWrappers::MPI::BlockSparseMatrix system_matrix;
        PETScWrappers::MPI::BlockSparseMatrix pressure_mass; // block (1,1) = M_p / nu, for the preconditioner
        PETScWrappers::MPI::BlockVector system_rhs;

        PETScWrappers::MPI::BlockVector solution_owned; // without ghost elements
        PETScWrappers::MPI::BlockVector solution;       // with ghost elements

        // Last solve: residual_history[k] = ||r^(k)|| (euclidean), k = 0, ..., n_iterations.
        std::vector<double> residual_history;
        unsigned int n_iterations = 0;
};

#endif
