#include "Diffusion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

void Diffusion::setup()
{
    pcout << "===============================================" << std::endl;
    
    // Create the mesh
    {
        pcout << "Initializing the mesh" << std::endl;

        // A parallel::fullydistributed::Triangulation cannot be filled directly by GridGenerator:
        // both meshes are built in serial, partitioned and then distributed.
        Triangulation<dim> mesh_serial;

        // Square (-1, 1)^2 of the paper, refined globally into N_el x N_el cells (N_el = 2^n).
        // Global refinement numbers the cells, and so the DoFs, hierarchically (Z-order), which
        // gives the sparsity pattern of Fig. 6 of the paper (a subdivided_hyper_cube would number
        // them by rows). The order matters: it is the order of the Gauss-Seidel smoother of
        // BoomerAMG and of the view of the matrix. colorize = true gives the boundary ids
        // 0 (x = -1), 1 (x = 1), 2 (y = -1), 3 (y = 1).
        unsigned int n_refinements = 0;
        while ((1u << n_refinements) < N_el)
            ++n_refinements;
        AssertThrow((1u << n_refinements) == N_el, ExcMessage("N_el must be a power of 2."));

        GridGenerator::hyper_cube(mesh_serial, -1.0, 1.0, true);
        mesh_serial.refine_global(n_refinements);

        GridTools::partition_triangulation(mpi_size, mesh_serial);

        const auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        // h = maximum cell diameter (the dataset uses h = 2 / N_el, as the paper: see main_diffusion.cpp).
        double h_local = 0.0;
        for (const auto &cell : mesh.active_cell_iterators())
            if (cell->is_locally_owned())
                h_local = std::max(h_local, cell->diameter());
        h_max = Utilities::MPI::max(h_local, MPI_COMM_WORLD);

        pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        fe = std::make_unique<FE_Q<dim>>(r);

        pcout << "  Degree                     = " << fe->degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

        quadrature = std::make_unique<QGauss<dim>>(r + 1);
        quadrature_boundary = std::make_unique<QGauss<dim - 1>>(r + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

        pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Dirichlet boundary conditions. They are imposed through AffineConstraints during the
    // assembly: the matrix stays symmetric (needed by CG and BoomerAMG) and the non-homogeneous
    // data end up in the right-hand side, so that x^(0) = 0 gives r^(0) = b.
    {
        pcout << "Initializing the constraints" << std::endl;

        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        VectorTools::interpolate_boundary_values(dof_handler, dirichlet, constraints);
        constraints.close();
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;

        DynamicSparsityPattern dsp(locally_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, MPI_COMM_WORLD, locally_relevant_dofs);

        pcout << "  Initializing the system matrix" << std::endl;
        system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, MPI_COMM_WORLD);
        system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    }
}

void Diffusion::assemble()
{
    pcout << "===============================================" << std::endl;
    pcout << "  Assembling the linear system" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();
    const unsigned int n_q_faces = quadrature_boundary -> size(); 

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_gradients | 
                            update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_values_boundary(*fe, *quadrature_boundary,
                                         update_values | update_quadrature_points | update_JxW_values);

    // Local matrix and vector.
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_matrix = 0.0;
    system_rhs    = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);

        cell_matrix = 0.0;
        cell_rhs    = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double mu_loc        = mu(fe_values.quadrature_point(q));
            const double f_loc         = f(fe_values.quadrature_point(q));
            
            // Checkpoint system assembly
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {

                    // Diffusion
                    cell_matrix(i, j) += mu_loc *                     
                                        fe_values.shape_grad(i, q) * 
                                        fe_values.shape_grad(j, q) * 
                                        fe_values.JxW(q);
                }

                cell_rhs(i) += f_loc *                       
                               fe_values.shape_value(i, q) *
                               fe_values.JxW(q);
            }
        }

        cell->get_dof_indices(dof_indices);

        // Dirichlet conditions
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices, system_matrix, system_rhs);
    }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

Diffusion::SolverResult Diffusion::solve(const double theta, const unsigned int n_repetitions, const unsigned int max_iterations)
{
    const double tolerance = 1.0e-8;

    SolverResult result;
    result.theta = theta;

    std::vector<double> times_setup(n_repetitions);
    std::vector<double> times_solve(n_repetitions);

    PETScWrappers::MPI::Vector residual(locally_owned_dofs, MPI_COMM_WORLD);

    // (rep and not r, which is the polynomial degree)
    for (unsigned int rep = 0; rep < n_repetitions; ++rep)
    {
        Timer timer;

        // Preconditioner: one V-cycle of BoomerAMG with strong threshold theta. All the other
        // parameters are left at their defaults (max_iter = 1, i.e. one V-cycle per application).
        timer.restart();
        PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
        data.symmetric_operator = true;  // symmetric smoothing: needed by CG
        data.strong_threshold   = theta; // default 0.25

        PETScWrappers::PreconditionBoomerAMG preconditioner;
        preconditioner.initialize(system_matrix, data);
        // Build the AMG hierarchy now, so that it is measured in the setup time and not in the
        // first application of the preconditioner inside the solve (no-op if already built).
        AssertThrow(PCSetUp(preconditioner.get_pc()) == 0, ExcMessage("PCSetUp failed."));
        timer.stop();
        times_setup[rep] = Utilities::MPI::max(timer.wall_time(), MPI_COMM_WORLD);

        // deal.II's CG only monitors its own, recursively updated, residual. The euclidean residual
        // r^(k) of the paper is computed in a function connected to the solver, called at every
        // iteration with the current iterate x^(k); it stores ||r^(k)|| and decides when to stop.
        // The SolverControl always returns success (tolerance = +inf), so the decision is taken by
        // that function alone: deal.II keeps iterating as long as one of the checks returns `iterate`.
        SolverControl solver_control(max_iterations, std::numeric_limits<double>::max(), false, false);
        SolverCG<PETScWrappers::MPI::Vector> solver(solver_control);

        solution_owned = 0.0; // x^(0) = 0 for every theta and repetition, so r^(0) = b

        residual_history.assign(1, system_rhs.l2_norm());
        n_iterations = 0;

        solver.connect([&](const unsigned int iteration, const double, const PETScWrappers::MPI::Vector &x) -> SolverControl::State
        {
            if (iteration > 0)
            {
                timer.stop(); // the extra matrix-vector product is not part of the solver time
                system_matrix.vmult(residual, x);
                residual.sadd(-1.0, 1.0, system_rhs); // r^(k) = b - A x^(k)
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
        times_solve[rep] = Utilities::MPI::max(timer.wall_time(), MPI_COMM_WORLD);

        result.iterations = n_iterations;
    }

    // Mean and standard deviation of the timings.
    for (unsigned int rep = 0; rep < n_repetitions; ++rep)
    {
        result.time_amg_setup += times_setup[rep] / n_repetitions;
        result.time_solve     += times_solve[rep] / n_repetitions;
    }
    for (unsigned int rep = 0; rep < n_repetitions; ++rep)
        result.time_solve_std += std::pow(times_solve[rep] - result.time_solve, 2) / n_repetitions;
    result.time_solve_std = std::sqrt(result.time_solve_std);

    // Set the Dirichlet values exactly on the constrained DoFs.
    constraints.distribute(solution_owned);
    solution = solution_owned;

    pcout << "  theta = " << theta
          << ": " << result.iterations << " CG iterations"
          << (result.converged ? "" : " (NOT converged)")
          << ", AMG setup = " << result.time_amg_setup << " s"
          << ", solve = " << result.time_solve << " s" << std::endl;

    return result;
}

double Diffusion::compute_convergence_factor(const unsigned int k) const
{
    // rho = rho^(N_it) by default
    const unsigned int step = (k == numbers::invalid_unsigned_int) ? n_iterations : k;
    AssertThrow(step >= 1 && step <= n_iterations, ExcIndexRange(step, 1, n_iterations + 1));

    // rho^(k) = (||r^(k)|| / ||r^(0)||)^(1/k)
    return std::pow(residual_history[step] / residual_history[0], 1.0 / step);
}

void Diffusion::output(const std::string &output_file_name_) const
{
    pcout << "===============================================" << std::endl;

    DataOut<dim> data_out;

    data_out.add_data_vector(dof_handler, solution, "solution");

    // Add vector for parallel partition.
    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();
    
    std::string output_file_name;
    if(selfMesh) output_file_name = "output-" + std::to_string(N_el);
    else 
    {
        const std::filesystem::path mesh_path(mesh_file_name);
        output_file_name = "output-" + mesh_path.stem().string();
    }
    // A name can be given, e.g. to keep the solutions of different problems on the same mesh.
    if(!output_file_name_.empty()) output_file_name = output_file_name_;

    // We choose VTU/PVTU files for parallel output format
    // @params : folder, basename, index
    data_out.write_vtu_with_pvtu_record("./", output_file_name, 0, MPI_COMM_WORLD);

    pcout << "Output written to " << output_file_name << std::endl;
    pcout << "===============================================" << std::endl;
}

double Diffusion::compute_error(const VectorTools::NormType &norm_type, const Function<dim> &exact_solution) const
{
    // Quadrilaterals: default (bilinear) mapping and Gauss quadrature.
    const QGauss<dim> quadrature_error(r + 2);

    Vector<double> error_per_cell(mesh.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

    const double error = VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

    return error;
}