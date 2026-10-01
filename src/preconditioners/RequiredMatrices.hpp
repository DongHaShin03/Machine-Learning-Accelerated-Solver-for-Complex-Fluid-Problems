#ifndef REQUIRED_MATRICES_HPP
#define REQUIRED_MATRICES_HPP

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

using namespace dealii;

struct RequiredMatrices
{
    const TrilinosWrappers::SparseMatrix *velocity_stiffness = nullptr; 
    const TrilinosWrappers::SparseMatrix *pressure_mass      = nullptr; 
    const TrilinosWrappers::SparseMatrix *velocity_mass      = nullptr; 
    const TrilinosWrappers::SparseMatrix *B                  = nullptr; 

    
    // PCD
    const TrilinosWrappers::SparseMatrix *pressure_convection_diffusion = nullptr; 
    const TrilinosWrappers::SparseMatrix *pressure_laplacian = nullptr; 
};

#endif