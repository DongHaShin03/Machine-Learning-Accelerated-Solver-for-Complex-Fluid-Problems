#include "DRT.hpp"

void DRT::setup()
{
    pcout << "===============================================" << std::endl;
    
    // Create the mesh
    {
        pcout << "Initializing the mesh" << std::endl;

        if(selfMesh)
        {
            GridGenerator::subdivided_hyper_cube(mesh, N_el, -1.0, 1.0, true);
            const std::string mesh_file_name = "mesh-" + std::to_string(N_el) + ".vtk";
            GridOut           grid_out;
            std::ofstream     grid_out_file(mesh_file_name);
            grid_out.write_vtk(mesh, grid_out_file);
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

        fe = std::make_unique<FE_SimplexP<dim>>(r);

        pcout << "  Degree                     = " << fe->degree << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
        quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(r + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;

        TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, sparsity);

        sparsity.compress();

        pcout << "  Initializing the system matrix" << std::endl;
        system_matrix.reinit(sparsity);
        system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    }
}

void DRT::assemble()
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
            const double sigma_loc     = sigma(fe_values.quadrature_point(q));
            const double f_loc         = f(fe_values.quadrature_point(q));
            const Tensor<1, dim> b_loc = b(fe_values.quadrature_point(q)); 
            
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

                    // Transport
                    cell_matrix(i, j) += b_loc * 
                                         fe_values.shape_grad(j, q) * 
                                         fe_values.shape_value(i, q) * 
                                         fe_values.JxW(q); 

                    cell_matrix(i, j) += sigma_loc *                 
                                        fe_values.shape_value(i, q) *
                                        fe_values.shape_value(j, q) *
                                        fe_values.JxW(q);
                }

                cell_rhs(i) += f_loc *                       
                               fe_values.shape_value(i, q) *
                               fe_values.JxW(q);
            }
        }
        
        // Neumann Boundary Conditions
        if (cell->at_boundary())
        {
            for (unsigned int face_number = 0; face_number < cell->n_faces(); ++face_number)
            {
                if (cell->face(face_number)->at_boundary())
                {
                    const types::boundary_id face_id = cell -> face(face_number) -> boundary_id(); 

                    if(neumann.count(face_id))
                    {
                        fe_values_boundary.reinit(cell, face_number);

                        const Function<dim>* boundary_function = neumann[face_id]; 

                        for (unsigned int q = 0; q < n_q_faces; ++q)
                        {
                            double h_loc = boundary_function -> value(fe_values_boundary.quadrature_point(q));
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                                cell_rhs(i) += h_loc * 
                                               fe_values_boundary.shape_value(i, q) * 
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
    }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    // Dirichlet boundary conditions
    {
        std::map<types::global_dof_index, double> boundary_values;

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 dirichlet,
                                                 boundary_values);

        MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, true);
    }
}

void DRT::solve()
{
    pcout << "===============================================" << std::endl;

    TrilinosWrappers::PreconditionSSOR preconditioner;
    preconditioner.initialize(system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

    // @params : maxiter, tolerance, reduce
    ReductionControl solver_control(10000, 1.0e-16, 1.0e-6);

    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    pcout << "  Solving the linear system" << std::endl;

    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
    solution = solution_owned; 
    pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}

void DRT::output() const
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

    // We choose VTU/PVTU files for parallel output format
    // @params : folder, basename, index
    data_out.write_vtu_with_pvtu_record("./", output_file_name, 0, MPI_COMM_WORLD);

    pcout << "Output written to " << output_file_name << std::endl;
    pcout << "===============================================" << std::endl;
}

double DRT::compute_error(const VectorTools::NormType &norm_type, const Function<dim> &exact_solution) const
{
    const QGaussSimplex<dim> quadrature_error(r + 2);

    FE_SimplexP<dim> fe_linear(1);
    MappingFE        mapping(fe_linear);

    Vector<double> error_per_cell(mesh.n_active_cells());
    VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

    const double error = VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

    return error;
}