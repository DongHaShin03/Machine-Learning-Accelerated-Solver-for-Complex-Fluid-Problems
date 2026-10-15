cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.5.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

project(stokes_amg CXX)

# The solver uses PETSc matrices/vectors, BoomerAMG (hypre via PETSc) and MPI.
# PETSc with hypre is checked at compile time in src/preconditioners/BlockDiagonalAMG.hpp.
if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_PETSC)
  message(FATAL_ERROR "deal.II must be configured with MPI and PETSc "
                      "(DEAL_II_WITH_MPI=${DEAL_II_WITH_MPI}, "
                      "DEAL_II_WITH_PETSC=${DEAL_II_WITH_PETSC}).")
endif()

# Stokes solver
add_executable(stokes src/main.cpp src/Stokes.cpp)
deal_ii_setup_target(stokes)

# Diffusion solver (dataset of Sec. 6.4)
add_executable(diffusion src/main_diffusion.cpp src/Diffusion.cpp)
deal_ii_setup_target(diffusion)

# `make run`
set(NPROCS 1 CACHE STRING "Number of MPI processes used by `make run`")
find_program(MPIEXEC_PROGRAM NAMES mpiexec mpirun)

add_custom_target(run
  COMMAND ${MPIEXEC_PROGRAM} -n ${NPROCS} $<TARGET_FILE:stokes>
  DEPENDS stokes
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running stokes on ${NPROCS} MPI process(es)")

