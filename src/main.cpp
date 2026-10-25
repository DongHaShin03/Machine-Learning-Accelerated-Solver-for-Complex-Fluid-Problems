#include "Stokes.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static constexpr unsigned int dim = Stokes::dim;

class Inlet : public Function<dim>
{
    public:
        Inlet(const double &U_max_) : Function<dim>(dim + 1)
        , U_max(U_max_)
        {}

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            double y = p[1];
            values[0] = 4.0 * U_max * y * (H - y) / (H * H); // u_x
            for (unsigned int i = 1; i < dim; ++i) values[i] = 0.0; // u_y = 0.0
            values[dim] = 0.0; // Pressure
        }
    private:
        const double H = 0.41;
        const double U_max;
};

// Usage: ./stokes [key=value ...], e.g. ./stokes max_level=2 n_repetitions=5
//   min_level, max_level  refinement levels i, h = 0.14 / 2^i (default 0 and 5, as in the paper)
//   n_repetitions         repetitions of each solve, to average the timings (default 1)
//   pressure_mass         pressure block of the preconditioner: paper (M_h, as written in the paper,
//                         default) or scaled (M_p / nu); in both cases it is inverted exactly
//   output                csv file with the results (default results.csv)
// Arguments without '=' are left to PETSc (e.g. -log_view).
// The program runs in serial, as in the paper: with more MPI processes the coarsening of BoomerAMG
// and the numbering of the DoFs would depend on the partition.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const bool is_root = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
                ExcMessage("The dataset is generated in serial, as in the paper: run without mpirun."));

    std::map<std::string, std::string> options = {
        {"min_level", "0"}, {"max_level", "5"}, {"n_repetitions", "1"},
        {"pressure_mass", "paper"}, {"output", "results.csv"}};
    for (int k = 1; k < argc; ++k)
    {
        const std::string arg   = argv[k];
        const auto        equal = arg.find('=');
        if (equal == std::string::npos)
            continue;
        AssertThrow(options.count(arg.substr(0, equal)) == 1, ExcMessage("Unknown option " + arg));
        options[arg.substr(0, equal)] = arg.substr(equal + 1);
    }
    AssertThrow(options["pressure_mass"] == "paper" || options["pressure_mass"] == "scaled",
                ExcMessage("pressure_mass must be paper or scaled."));

    // Taylor-Hood Q2-Q1
    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;

    // Dataset (Section 6.5)
    const unsigned int min_refinement = std::stoul(options["min_level"]);
    const unsigned int max_refinement = std::stoul(options["max_level"]);
    const std::vector<double> nu_values    = {0.001, 0.1, 10};
    const std::vector<double> U_max_values = {0.00001, 0.001, 0.1, 10};

    // Strong threshold theta. The paper does not list the values for Stokes, but its 3600 samples
    // over the 72 problems (nu, U, i) give 50 values: theta = 0.02 k, k = 1, ..., 49, and the default
    // 0.25, which is the reference value rho_0.25 of the paper. They include the 25 values of the
    // diffusion dataset.
    std::vector<double> theta_values = {0.25};
    for (unsigned int k = 1; k <= 49; ++k)
        theta_values.push_back(0.02 * k);
    std::sort(theta_values.begin(), theta_values.end());

    // Solution written to file only for this theta.
    const double theta_output = 0.25;

    // Repetitions of each solve to average the timings (Sec. 5.3);
    const unsigned int n_repetitions = std::stoul(options["n_repetitions"]);

    const bool scale_pressure_mass_by_nu = (options["pressure_mass"] == "scaled");

    // Forcing term f = 0
    const auto f = [](const Point<dim> &) { return Tensor<1, dim>(); };

    std::ofstream csv;
    if (is_root)
    {
        csv.open(options["output"]);
        csv << "i,h,n_dofs,nu,U,theta,converged,iterations,rho,"
               "t_amg_setup,t_solve,t_solve_std" << std::endl;
    }

    for (unsigned int i = min_refinement; i <= max_refinement; ++i)
    {
        // h of the paper (Sec. 6.5); -log2(h) is an input of the ANN. The measured maximum cell
        // diameter (Stokes::get_h()) halves less than h because of the curved cells around the cylinder.
        const double h = 0.14 / std::pow(2.0, i);

        for (const double nu_value : nu_values)
        {
            for (const double U_max : U_max_values)
            {
                // Viscosity's value
                const auto nu = [nu_value](const Point<dim> &) { return nu_value; };
                Stokes problem(i, degree_velocity, degree_pressure, nu, f, scale_pressure_mass_by_nu);

                // Boundary Conditions
                Inlet inlet_velocity(U_max);
                Functions::ZeroFunction<dim> zero_function(dim + 1);

                problem.dirichlet[0] = &inlet_velocity; // inlet
                problem.dirichlet[2] = &zero_function;  // cylinder
                problem.dirichlet[3] = &zero_function;  // walls

                problem.setup();
                problem.assemble();

                for (const double theta : theta_values)
                {
                    const Stokes::SolverResult result = problem.solve(theta, n_repetitions);
                    const double rho = problem.compute_convergence_factor();

                    if (is_root)
                    {
                        std::cout << "  rho = " << rho << std::endl;
                        csv << i << ',' << h << ',' << problem.n_dofs() << ','
                            << nu_value << ',' << U_max << ',' << theta << ','
                            << result.converged << ',' << result.iterations << ','
                            << rho << ','
                            << result.time_amg_setup << ',' << result.time_solve << ','
                            << result.time_solve_std << std::endl;
                    }

                    if (std::abs(theta - theta_output) < 1e-12)
                    {
                        std::ostringstream output_file_name;
                        output_file_name << "output-stokes_nu" << nu_value << "_U" << U_max << "_i" << i;
                        problem.output(output_file_name.str());
                    }
                }
            }
        }
    }

    return 0;
}
