#ifndef STOKES_HPP
#define STOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "preconditioners/NavierStokesPreconditioner.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

// Class implementing a solver for the Stokes problem in the form: 
// sigma * u - nu * delta(u) + grad(p) = f
// div(u) = 0
class Stokes
{
    public:
        static constexpr unsigned int dim = 3;

        using Value = std::function<double(const Point<dim> &p)>; 
        using VectorField = std::function<Tensor<1, dim>(const Point<dim> &p)>; 

        class PreconditionIdentity
        {
            public:
                void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const { dst = src; }
        };
        // Constructor for self generated mesh
        Stokes
        (
            const unsigned int &N_el_,
            const unsigned int &degree_velocity_,
            const unsigned int &degree_pressure_,
            const Value &sigma_, 
            const Value &nu_, 
            const VectorField &f_
        )
        : selfMesh(true)
        , N_el(N_el_)
        , mesh_file_name("")
        , degree_velocity(degree_velocity_)
        , degree_pressure(degree_pressure_)
        , sigma(sigma_)
        , nu(nu_)
        , f(f_)
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , mesh(MPI_COMM_WORLD)
        , pcout(std::cout, mpi_rank == 0)
        {}

        // Constructor for externally generated mesh
        Stokes
        (
            const std::string  &mesh_file_name_,
            const unsigned int &degree_velocity_,
            const unsigned int &degree_pressure_,
            const Value &sigma_,
            const Value &nu_, 
            const VectorField &f_
        )
        : selfMesh(false)
        , N_el(0)
        , mesh_file_name(mesh_file_name_)
        , degree_velocity(degree_velocity_)
        , degree_pressure(degree_pressure_)
        , sigma(sigma_)
        , nu(nu_)
        , f(f_)
        , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , mesh(MPI_COMM_WORLD)
        , pcout(std::cout, mpi_rank == 0)
        {}

        void setup();
        void assemble();
        void solve();
        void output();

        void set_preconditioner(std::unique_ptr<NavierStokesPreconditioner> prec) {preconditioner = std::move(prec);}
        std::map<types::boundary_id, const Function<dim> *> dirichlet;
        std::map<types::boundary_id, const Function<dim> *> neumann;
    protected:
        bool selfMesh; 
        const unsigned int N_el; 
        const std::string mesh_file_name;
        const unsigned int degree_velocity;
        const unsigned int degree_pressure;
        const Value sigma; 
        const Value nu; // viscosity [m2/s].
        const VectorField f;

        const unsigned int mpi_size;
        const unsigned int mpi_rank;
        parallel::fullydistributed::Triangulation<dim> mesh;

        std::unique_ptr<FiniteElement<dim>> fe;
        std::unique_ptr<Quadrature<dim>> quadrature;
        std::unique_ptr<Quadrature<dim-1>> quadrature_boundary;

        // DoF handler.
        DoFHandler<dim> dof_handler;
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        std::vector<IndexSet> block_owned_dofs; // DoFs owned by current process in the velocity and pressure blocks.
        std::vector<IndexSet> block_relevant_dofs; // DoFs relevant to current process in the velocity and pressure blocks.

        TrilinosWrappers::BlockSparseMatrix system_matrix;
        TrilinosWrappers::MPI::BlockVector system_rhs;

        TrilinosWrappers::MPI::BlockVector solution_owned; // without ghost elements
        TrilinosWrappers::MPI::BlockVector solution;       // with ghost elements

        // Preconditioner
        TrilinosWrappers::BlockSparseMatrix velocity_mass; // Mu
        TrilinosWrappers::BlockSparseMatrix pressure_mass; // Mp
        AssemblyFlags flags;

        std::unique_ptr<NavierStokesPreconditioner> preconditioner;

        ConditionalOStream pcout;
};

#endif