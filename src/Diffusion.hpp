#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>

// The preconditioner is one V-cycle of BoomerAMG (hypre, via PETSc), as in the paper.
#ifndef DEAL_II_PETSC_WITH_HYPRE
#  error "BoomerAMG requires PETSc configured with hypre."
#endif

using namespace dealii; 

// Diffusion-reaction-transport problem -div(mu grad u) + b . grad u + sigma u = f (same
// assembly as DRT), set up for the AMG dataset of the paper (Sec. 5 and 6.4):
// - Q_r elements on quadrilaterals (Q1 in the paper); self-generated mesh on (-1, 1)^2;
// - PETSc linear algebra, CG preconditioned by one BoomerAMG V-cycle with strong threshold theta;
// - Dirichlet conditions through AffineConstraints, so that the matrix stays symmetric;
// - stopping criterion and approximate convergence factor rho of the paper.
// CG and BoomerAMG need a symmetric matrix: b = 0 (in the paper also sigma = 0).
class Diffusion
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

        // Constructor for self-generated mesh: (-1, 1)^2 with N_el x N_el square cells
        Diffusion
        (
            const unsigned int &N_el_,
            const unsigned int &r_, 
            const Value &mu_, 
            const Value &sigma_, 
            const Value &f_,
            const VectorField &b_
        )
        : selfMesh(true)
        , N_el(N_el_)
        , mesh_file_name("")
        , r(r_)
        , mu(mu_)
        , sigma(sigma_)
        , f(f_)
        , b(b_)
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , mesh(MPI_COMM_WORLD)
        , pcout(std::cout, mpi_rank == 0)
        {selfMesh = true;}

        // Constructor for externally generated mesh (.msh, quadrilaterals only)
        Diffusion
        (   const std::string &mesh_file_name_,
            const unsigned int &r_, 
            const Value &mu_, 
            const Value &sigma_, 
            const Value &f_,
            const VectorField &b_
        ) 
        : selfMesh(false)
        , N_el(0)
        , mesh_file_name(mesh_file_name_)
        , r(r_)
        , mu(mu_)
        , sigma(sigma_)
        , f(f_)
        , b(b_)
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , mesh(MPI_COMM_WORLD)
        , pcout(std::cout, mpi_rank == 0)
        {}

        void setup(); 
        void assemble(); 
        SolverResult solve(const double theta, const unsigned int n_repetitions = 1, const unsigned int max_iterations = 10000);
        double compute_convergence_factor(const unsigned int k = numbers::invalid_unsigned_int) const;
        void output(const std::string &output_file_name = "") const;
        double compute_error(const VectorTools::NormType &norm_type, const Function<dim> &exact_solution) const; 
        
        double get_h() const {return h_max;}
        types::global_dof_index n_dofs() const {return dof_handler.n_dofs();}

        // Boundary data by boundary id. Dirichlet: set before setup(); Neumann: before assemble().
        std::map<types::boundary_id, const Function<dim> *> dirichlet;
        std::map<types::boundary_id, const Function<dim> *> neumann;

    protected: 
        bool selfMesh; 
        const unsigned int N_el; 
        const std::string mesh_file_name; 
        
        // FE polynomial degree
        const unsigned int r; 

        // DRT functions
        Value mu; 
        Value sigma; 
        Value f; 
        VectorField b; 

        // Parallelization fields
        const unsigned int mpi_size;  
        const unsigned int mpi_rank;
        parallel::fullydistributed::Triangulation<dim> mesh; 
        double h_max = 0.0; // maximum cell diameter

        std::unique_ptr<FiniteElement<dim>> fe; 
        std::unique_ptr<Quadrature<dim>> quadrature; 
        std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary; 

        DoFHandler<dim> dof_handler; 

        IndexSet locally_owned_dofs; 
        IndexSet locally_relevant_dofs; 

        // Dirichlet conditions (keep the system matrix symmetric).
        AffineConstraints<double> constraints;

        PETScWrappers::MPI::SparseMatrix system_matrix;
        PETScWrappers::MPI::Vector system_rhs;

        PETScWrappers::MPI::Vector solution_owned; // without ghost elements
        PETScWrappers::MPI::Vector solution;       // with ghost elements

        // Last solve: residual_history[k] = ||r^(k)|| (euclidean), k = 0, ..., n_iterations.
        std::vector<double> residual_history;
        unsigned int n_iterations = 0;
        
        ConditionalOStream pcout; 
}; 

#endif