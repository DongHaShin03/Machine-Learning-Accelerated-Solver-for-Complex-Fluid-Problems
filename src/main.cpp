#include "Stokes.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
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

// Usage: ./stokes [max_refinement] [n_repetitions]  (default 5, as in the paper)
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const bool is_root = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    // Taylor-Hood Q2-Q1 
    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;

    // Dataset (Section 6.5)
    const unsigned int max_refinement = (argc > 1) ? std::stoul(argv[1]) : 5;
    const std::vector<double> nu_values    = {0.001, 0.1, 10};
    const std::vector<double> U_max_values = {0.00001, 0.001, 0.1, 10};

    // Strong threshold theta: 25 values in [0.02, 0.9] (Sec. 5), including the
    // default 0.25, which is the reference value rho_0.25 of the paper.
    std::vector<double> theta_values = {0.02, 0.25, 0.9};
    for (unsigned int k = 1; k <= 22; ++k)
        theta_values.push_back(0.04 * k);
    std::sort(theta_values.begin(), theta_values.end());

    // Solution written to file only for this theta.
    const double theta_output = 0.25;

    // Repetitions of each solve to average the timings (Sec. 5.3); 
    const unsigned int n_repetitions = (argc > 2) ? std::stoul(argv[2]) : 1;

    // Forcing term f = 0 
    const auto f = [](const Point<dim> &) { return Tensor<1, dim>(); };

    std::ofstream csv;
    if (is_root)
    {
        csv.open("results.csv");
        csv << "i,h,n_dofs,nu,U,theta,converged,iterations,rho,"
               "t_amg_setup,t_solve,t_solve_std" << std::endl;
    }

    for (unsigned int i = 0; i <= max_refinement; ++i)
    {
        for (const double nu_value : nu_values)
        {
            for (const double U_max : U_max_values)
            {
                // Viscosity's value
                const auto nu = [nu_value](const Point<dim> &) { return nu_value; };
                Stokes problem(i, degree_velocity, degree_pressure, nu, f);

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
                        csv << i << ',' << problem.get_h() << ',' << problem.n_dofs() << ','
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
