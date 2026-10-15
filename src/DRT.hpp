#ifndef DRT_HPP
#define DRT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <filesystem>

using namespace dealii; 

class DRT
{
    public: 
        static constexpr unsigned int dim = 2; 

        using Value = std::function<double(const Point<dim> &p)>; 
        using VectorField = std::function<Tensor<1, dim>(const Point<dim> &p)>; 

        // Constructor for self-generated mesh
        DRT
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

        // Constructor for externally generated mesh
        DRT
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
        void solve(); 
        void output() const; 
        double compute_error(const VectorTools::NormType &norm_type, const Function<dim> &exact_solution) const; 
        
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

        std::unique_ptr<FiniteElement<dim>> fe; 
        std::unique_ptr<Quadrature<dim>> quadrature; 
        std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary; 

        DoFHandler<dim> dof_handler; 

        IndexSet locally_owned_dofs; 
        IndexSet locally_relevant_dofs; 

        TrilinosWrappers::SparseMatrix system_matrix; 
        TrilinosWrappers::MPI::Vector system_rhs; 

        TrilinosWrappers::MPI::Vector solution_owned; // without ghost elements
        TrilinosWrappers::MPI::Vector solution;       // with ghost elements
        
        ConditionalOStream pcout; 
}; 

#endif