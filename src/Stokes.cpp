#include "Stokes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

void Stokes::setup()
{
    AssertThrow(!dirichlet.empty(), ExcMessage("Set the Dirichlet boundary conditions before calling setup()."));

    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;

        // Channel with cylinder of the paper (Sec. 6.5, Fig. 13):
        // Omega = (0, 2.2) x (0, 0.41) \ B_0.05(0.2, 0.2).
        // colorize = true gives the boundary ids
        //   0 = inflow (x = 0), 1 = outflow (x = 2.2), 2 = cylinder, 3 = walls (y = 0, y = 0.41).
        // refine_global(i) gives h ~ 0.14 / 2^i, as in the paper.
        Triangulation<dim> mesh_serial;
        GridGenerator::channel_with_cylinder(mesh_serial, 0.03, 2, 2.0, true);
        mesh_serial.refine_global(n_refinements);

        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        double h_local = 0.0;
        for (const auto &cell : mesh.active_cell_iterators())
            if (cell->is_locally_owned())
                h_local = std::max(h_local, cell->diameter());
        h_max = Utilities::MPI::max(h_local, MPI_COMM_WORLD);

        pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
        pcout << "  h_max              = " << h_max << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        const FE_Q<dim> fe_scalar_velocity(degree_velocity);
        const FE_Q<dim> fe_scalar_pressure(degree_pressure);
        fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim, fe_scalar_pressure, 1);

        pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree << std::endl;
        pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->n_dofs_per_cell() << std::endl;

        quadrature = std::make_unique<QGauss<dim>>(fe->degree + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        // We want to reorder DoFs so that all velocity DoFs come first, and then
        // all pressure DoFs.
        std::vector<unsigned int> block_component(dim + 1, 0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise(dof_handler, block_component);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

        // Besides the locally owned and locally relevant indices for the whole
        // system (velocity and pressure), we will also need those for the
        // individual velocity and pressure blocks.
        std::vector<types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const types::global_dof_index n_u = dofs_per_block[0];
        const types::global_dof_index n_p = dofs_per_block[1];

        block_owned_dofs.resize(2);
        block_relevant_dofs.resize(2);
        block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
        block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
        block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
        block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

        pcout << "  Number of DoFs: " << std::endl;
        pcout << "    Velocity = " << n_u << std::endl;
        pcout << "    Pressure = " << n_p << std::endl;
        pcout << "    Total    = " << n_u + n_p << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Dirichlet boundary conditions on the velocity only. They are imposed through
    // AffineConstraints during assembly, which keeps the matrix symmetric (needed
    // by MINRES) and moves the non-homogeneous data (inflow profile) to the rhs.
    // Boundary ids not in `dirichlet` (id 1, outflow) get the natural do-nothing
    // condition nu du/dn - p n = 0.
    {
        pcout << "Initializing the constraints" << std::endl;

        const FEValuesExtractors::Vector velocity(0);
        const ComponentMask velocity_mask = fe->component_mask(velocity);

        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        for (const auto &[boundary_id, function] : dirichlet)
            VectorTools::interpolate_boundary_values(dof_handler, boundary_id, *function, constraints, velocity_mask);
        constraints.close();
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;

        pcout << "  Initializing the sparsity pattern" << std::endl;

        // Velocity DoFs interact with other velocity DoFs (the weak formulation has
        // terms involving u times v), and pressure DoFs interact with velocity DoFs
        // (there are terms involving p times v or u times q). However, pressure
        // DoFs do not interact with other pressure DoFs (there are no terms
        // involving p times q). We build a table to store this information, so that
        // the sparsity pattern can be built accordingly.
        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim) // pressure-pressure term
                    coupling[c][d] = DoFTools::none;
                else // other combinations
                    coupling[c][d] = DoFTools::always;
            }
        }

        BlockDynamicSparsityPattern dsp(block_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, coupling, dsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, MPI_COMM_WORLD, locally_relevant_dofs);

        // We also build a sparsity pattern for the pressure mass matrix (for the preconditioner)
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim) // pressure-pressure term
                    coupling[c][d] = DoFTools::always;
                else // other combinations
                    coupling[c][d] = DoFTools::none;
            }
        }

        BlockDynamicSparsityPattern dsp_pressure_mass(block_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, coupling, dsp_pressure_mass, constraints, false);
        SparsityTools::distribute_sparsity_pattern(dsp_pressure_mass, locally_owned_dofs, MPI_COMM_WORLD, locally_relevant_dofs);

        pcout << "  Initializing the matrices" << std::endl;
        system_matrix.reinit(block_owned_dofs, dsp, MPI_COMM_WORLD);
        pressure_mass.reinit(block_owned_dofs, dsp_pressure_mass, MPI_COMM_WORLD);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
    }
}

void Stokes::assemble()
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system" << std::endl;

    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
    const unsigned int n_q           = quadrature->size();

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    // Local matrices and vector.
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    // Shape functions, evaluated once per quadrature point.
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);

    system_matrix = 0.0;
    pressure_mass = 0.0;
    system_rhs    = 0.0;

    const FEValuesExtractors::Vector velocity(0);
    const FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);

        cell_matrix               = 0.0;
        cell_pressure_mass_matrix = 0.0;
        cell_rhs                  = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const Tensor<1, dim> f_loc  = f(fe_values.quadrature_point(q));
            const double         nu_loc = nu(fe_values.quadrature_point(q));
            const double         JxW    = fe_values.JxW(q);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                phi_u[k]      = fe_values[velocity].value(k, q);
                grad_phi_u[k] = fe_values[velocity].gradient(k, q);
                div_phi_u[k]  = fe_values[velocity].divergence(k, q);
                phi_p[k]      = fe_values[pressure].value(k, q);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Viscous term a(u, v) = (nu grad u, grad v).
                    cell_matrix(i, j) += nu_loc * scalar_product(grad_phi_u[j], grad_phi_u[i]) * JxW;

                    // Pressure term in the momentum equation: b(p, v) = -(p, div v).
                    cell_matrix(i, j) -= phi_p[j] * div_phi_u[i] * JxW;

                    // Continuity equation: b(q, u) = -(q, div u). Same sign as above,
                    // so that the system [A B^T; B 0] is symmetric (eq. 8).
                    cell_matrix(i, j) -= phi_p[i] * div_phi_u[j] * JxW;

                    // Pressure mass matrix scaled by 1/nu (preconditioner).
                    cell_pressure_mass_matrix(i, j) += phi_p[i] * phi_p[j] / nu_loc * JxW;
                }

                // Forcing term.
                cell_rhs(i) += f_loc * phi_u[i] * JxW;
            }
        }

        cell->get_dof_indices(dof_indices);

        // Matrix and rhs together, so that the inflow data ends up in the rhs.
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices, system_matrix, system_rhs);
        constraints.distribute_local_to_global(cell_pressure_mass_matrix, dof_indices, pressure_mass);
    }

    system_matrix.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

Stokes::SolverResult Stokes::solve(const double theta, const unsigned int n_repetitions)
{
    AssertThrow(n_repetitions > 0, ExcMessage("n_repetitions must be positive."));

    // Stopping criterion of the paper (Sec. 5): relative residual in the euclidean norm,
    //   N_it = min{k : ||r^(k)|| / ||r^(0)|| < 1e-8},  r^(k) = b - K x^(k).
    const unsigned int max_iterations = 10000;
    const double       tolerance      = 1.0e-8;

    SolverResult result;
    result.theta = theta;

    std::vector<double> times_setup(n_repetitions);
    std::vector<double> times_solve(n_repetitions);

    PETScWrappers::MPI::BlockVector residual(block_owned_dofs, MPI_COMM_WORLD);

    for (unsigned int r = 0; r < n_repetitions; ++r)
    {
        Timer timer;

        // Preconditioner: one BoomerAMG V-cycle on A (strong threshold theta), Jacobi on M_p/nu.
        timer.restart();
        BlockDiagonalAMG preconditioner;
        preconditioner.initialize(system_matrix.block(0, 0), pressure_mass.block(1, 1), theta);
        timer.stop();
        times_setup[r] = Utilities::MPI::max(timer.wall_time(), MPI_COMM_WORLD);

        // deal.II's MINRES only monitors the preconditioned residual norm. The euclidean
        // residual r^(k) is computed in a function connected to the solver, called at
        // every iteration with the current iterate x^(k); it stores ||r^(k)|| and decides
        // when to stop. The SolverControl always returns success (tolerance = +inf), so
        // the decision is taken by that function alone: deal.II keeps iterating as long
        // as one of the checks returns `iterate`.
        SolverControl solver_control(max_iterations, std::numeric_limits<double>::max(), false, false);
        SolverMinRes<PETScWrappers::MPI::BlockVector> solver(solver_control);

        solution_owned = 0.0; // x^(0) = 0, so r^(0) = b

        residual_history.assign(1, system_rhs.l2_norm());
        n_iterations = 0;

        solver.connect([&](const unsigned int iteration, const double, const PETScWrappers::MPI::BlockVector &x) -> SolverControl::State
        {
            if (iteration > 0)
            {
                timer.stop(); // the extra matrix-vector product is not part of the solver time
                system_matrix.vmult(residual, x);
                residual.sadd(-1.0, 1.0, system_rhs); // r^(k) = b - K x^(k)
                residual_history.resize(iteration + 1, 0.0);
                residual_history[iteration] = residual.l2_norm();
                n_iterations = iteration;
                timer.start();
            }

            const double relative_residual = residual_history[n_iterations] / residual_history[0];
            if (!std::isfinite(relative_residual))
                return SolverControl::failure;
            if (relative_residual < tolerance)
                return SolverControl::success;
            if (iteration >= max_iterations)
                return SolverControl::failure;
            return SolverControl::iterate;
        });

        timer.restart();
        try
        {
            solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
            result.converged = true;
        }
        catch (const SolverControl::NoConvergence &)
        {
            result.converged = false;
        }
        timer.stop();
        times_solve[r] = Utilities::MPI::max(timer.wall_time(), MPI_COMM_WORLD);

        result.iterations = n_iterations;
    }

    // Mean and standard deviation of the timings.
    for (unsigned int r = 0; r < n_repetitions; ++r)
    {
        result.time_amg_setup += times_setup[r] / n_repetitions;
        result.time_solve     += times_solve[r] / n_repetitions;
    }
    for (unsigned int r = 0; r < n_repetitions; ++r)
        result.time_solve_std += std::pow(times_solve[r] - result.time_solve, 2) / n_repetitions;
    result.time_solve_std = std::sqrt(result.time_solve_std);

    // Set the Dirichlet values exactly on the constrained DoFs.
    constraints.distribute(solution_owned);
    solution = solution_owned;

    pcout << "  theta = " << theta
          << ": " << result.iterations << " MINRES iterations"
          << (result.converged ? "" : " (NOT converged)")
          << ", AMG setup = " << result.time_amg_setup << " s"
          << ", solve = " << result.time_solve << " s" << std::endl;

    return result;
}

double Stokes::compute_convergence_factor(const unsigned int k) const
{
    AssertThrow(n_iterations > 0 && residual_history.size() == n_iterations + 1,
                ExcMessage("Call solve() before compute_convergence_factor()."));

    // rho = rho^(N_it) by default
    const unsigned int step = (k == numbers::invalid_unsigned_int) ? n_iterations : k;
    AssertThrow(step >= 1 && step <= n_iterations, ExcIndexRange(step, 1, n_iterations + 1));

    // rho^(k) = (||r^(k)|| / ||r^(0)||)^(1/k)
    return std::pow(residual_history[step] / residual_history[0], 1.0 / step);
}

void Stokes::output(const std::string &output_file_name) const
{
    pcout << "===============================================" << std::endl;

    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> names(dim, "velocity");
    names.push_back("pressure");

    data_out.add_data_vector(dof_handler, solution, names, interpretation);

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./",
                                        output_file_name,
                                        0,
                                        MPI_COMM_WORLD);

    pcout << "Output written to " << output_file_name << std::endl;
    pcout << "===============================================" << std::endl;
}
