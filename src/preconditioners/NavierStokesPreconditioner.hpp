#ifndef NAVIER_STOKES_PRECONDITIONER_HPP
#define NAVIER_STOKES_PRECONDITIONER_HPP

#include "RequiredMatrices.hpp"

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>

struct AssemblyFlags
{
    bool Mu = false;
    bool Mp = false;
};

class NavierStokesPreconditioner
{
    public:
        virtual ~NavierStokesPreconditioner() = default;

        virtual AssemblyFlags get_needed_matrices() const {return AssemblyFlags();}

        virtual void initialize(const RequiredMatrices &a) = 0;
        virtual void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const = 0;
};

#endif