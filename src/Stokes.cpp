#include "Stokes.hpp"

void Stokes::setup()
{
    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;
        if(selfMesh)
        {

        }
        else
        {
            Triangulation<dim> mesh_serial;

            GridIn<dim> grid_in;
            grid_in.attach_triangulation(mesh_serial);

            std::ifstream mesh_file(mesh_file_name);
            grid_in.read_msh(mesh_file);

            GridTools::partition_triangulation(mpi_size, mesh_serial);
            const auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
            mesh.create_triangulation(construction_data);
        }

        pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
        const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
        fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity, dim, fe_scalar_pressure, 1);

        pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree << std::endl;
        pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
        quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
        pcout << "  Quadrature points per face = " << quadrature_boundary->size() << std::endl;
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
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];

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

        TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
        sparsity.compress();

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
        TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity_pressure_mass);
        sparsity_pressure_mass.compress();

        pcout << "  Initializing the matrices" << std::endl;
        system_matrix.reinit(sparsity);
        pressure_mass.reinit(sparsity_pressure_mass);

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

    if(preconditioner)
        flags = preconditioner->get_needed_matrices();

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q           = quadrature->size();
    const unsigned int n_q_face      = quadrature_boundary->size();

    FEValues<dim> fe_values(*fe, *quadrature,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_values_boundary(*fe, *quadrature_boundary,
                                           update_values | update_normal_vectors | update_JxW_values);

    if(flags.Mu) velocity_mass = 0.0; 
    if(flags.Mp) pressure_mass = 0.0; 

    // Local matrix and vector. 
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_velocity_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_matrix = 0.0;
    system_rhs    = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);

        cell_matrix               = 0.0;
        cell_rhs                  = 0.0;

        if (flags.Mu) cell_velocity_mass_matrix = 0.0;
        if (flags.Mp) cell_pressure_mass_matrix = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const Tensor<1, dim> f_loc = f(fe_values.quadrature_point(q)); 
            const double sigma_loc     = sigma(fe_values.quadrature_point(q)); 
            const double nu_loc        = nu(fe_values.quadrature_point(q)); 

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const Tensor<1, dim> phi_vel_i = fe_values[velocity].value(i, q);
                const Tensor<2, dim> grad_phi_vel_i = fe_values[velocity].gradient(i,q);
                const double div_phi_vel_i = fe_values[velocity].divergence(i, q);

                const double psi_i = fe_values[pressure].value(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {   
                    const Tensor<1, dim> phi_vel_j = fe_values[velocity].value(j, q);
                    const Tensor<2, dim> grad_phi_vel_j = fe_values[velocity].gradient(j,q);
                    const double div_phi_vel_j = fe_values[velocity].divergence(j, q);

                    const double psi_j = fe_values[pressure].value(j, q);
                    
                    if(sigma_loc != 0)
                    {
                        cell_matrix(i, j) += sigma_loc * 
                                             scalar_product(phi_vel_j, phi_vel_i) * 
                                             fe_values.JxW(q); 
                    }

                    // Viscosity term.
                    if(nu_loc != 0)
                    {
                        cell_matrix(i, j) += nu_loc * 
                                             scalar_product(grad_phi_vel_j, grad_phi_vel_i) *
                                             fe_values.JxW(q);
                    }

                    // Pressure term in the momentum equation.
                    // div(v) * p
                    cell_matrix(i, j) -= (psi_i * div_phi_vel_j) * 
                                         fe_values.JxW(q);

                    // Pressure term in the continuity equation.
                    // div(u) * q
                    cell_matrix(i, j) -= (psi_j * div_phi_vel_i) * 
                                         fe_values.JxW(q);

                    if (flags.Mu) cell_velocity_mass_matrix(i, j) += scalar_product(phi_vel_i, phi_vel_i) * fe_values.JxW(q);
                    if (flags.Mp) cell_pressure_mass_matrix(i, j) += psi_i * psi_j / nu_loc * fe_values.JxW(q);
                }

                // Forcing term.
                cell_rhs(i) += f_loc * phi_vel_i * fe_values.JxW(q); 
            }
        }

        // Boundary integral for Neumann BCs.
        if (cell->at_boundary())
        {
            for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
                if (cell->face(f)->at_boundary())
                {
                    const types::boundary_id face_id = cell -> face(f) -> boundary_id(); 
                    
                    if(neumann.count(face_id))
                    {
                        fe_values_boundary.reinit(cell, f);
                        const Function<dim>* boundary_function = neumann[face_id];

                        for (unsigned int q = 0; q < n_q_face; ++q)
                        {
                            double h_loc = boundary_function -> value(fe_values_boundary.quadrature_point(q));
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                                cell_rhs(i) +=  h_loc * 
                                                scalar_product(fe_values_boundary.normal_vector(q), fe_values_boundary[velocity].value(i, q)) *
                                                fe_values_boundary.JxW(q);
                            }
                        }
                    }
                }
            }
        }

        cell->get_dof_indices(dof_indices);

        system_matrix.add(dof_indices, cell_matrix);
        system_rhs.add(dof_indices, cell_rhs);
        if(flags.Mu) velocity_mass.add(dof_indices, cell_velocity_mass_matrix);
        if(flags.Mp) pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    if(flags.Mu) velocity_mass.compress(VectorOperation::add);
    if(flags.Mp) pressure_mass.compress(VectorOperation::add);

    // Dirichlet boundary conditions
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;

        ComponentMask mask_velocity(dim + 1, true);
        mask_velocity.set(dim, false);

        boundary_functions[0] = dirichlet[0]; 
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions, 
                                                 boundary_values,
                                                 mask_velocity); 
        boundary_functions.clear(); 


        MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_owned, system_rhs, false);
    }
}

void Stokes::solve()
{
    pcout << "===============================================" << std::endl;

    SolverControl solver_control(2000, 1e-6 * system_rhs.l2_norm());

    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    pcout << "Solving the linear system" << std::endl;
    solution_owned = 0.0;

    RequiredMatrices data;

    data.velocity_stiffness = &system_matrix.block(0, 0);
    data.B = &system_matrix.block(1, 0);

    if(flags.Mu) data.velocity_mass = &velocity_mass.block(0, 0);
    if(flags.Mp) data.pressure_mass = &pressure_mass.block(1, 1);

    if (preconditioner)
    {
        preconditioner->initialize(data);
        solver.solve(system_matrix, solution_owned, system_rhs, *preconditioner);
    }
    else pcout << "Error: a preconditioner is needed in order to solve the problem." << std::endl; 

    pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;

    solution = solution_owned;
}

void Stokes::output()
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

    const std::string output_file_name = "output-stokes";
    data_out.write_vtu_with_pvtu_record("./",
                                        output_file_name,
                                        0,
                                        MPI_COMM_WORLD);

    pcout << "Output written to " << output_file_name << std::endl;
    pcout << "===============================================" << std::endl;
}   