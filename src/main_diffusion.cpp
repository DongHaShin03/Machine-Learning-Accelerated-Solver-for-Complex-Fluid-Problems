#include "Diffusion.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static constexpr unsigned int dim = Diffusion::dim;

// Diffusion coefficient of the paper (Sec. 6.4, Algorithm 5): the square (-1, 1)^2 is split
// into size^mode tiles, i.e. size stripes along x (mode = 1) or a size x size checkerboard
// (mode = 2), and mu = 10^eps[k] on tile k.
struct Pattern
{
    unsigned int size;
    unsigned int mode;
    std::vector<double> eps; // size^mode exponents

    double operator()(const Point<dim> &p) const
    {
        const auto tile = [this](const double x)
        {
            return std::min(size - 1, static_cast<unsigned int>(std::floor((x + 1.0) / 2.0 * size)));
        };
        const unsigned int k = (mode == 1) ? tile(p[0]) : tile(p[0]) + size * tile(p[1]);
        return std::pow(10.0, eps[k]);
    }
};

// Exact solution of the paper (Sec. 6.4), with a = size pi / 2:
//   u = sin(a x) sin(a y) if size is odd,  u = cos(a x) cos(a y) otherwise.
// Its normal derivative vanishes on the tile interfaces, so it solves -div(mu grad u) = f with
// f = 2 a^2 mu u for every mu constant on each tile (u does not depend on mu).
class ExactSolution : public Function<dim>
{
    public:
        ExactSolution(const unsigned int &size_) : Function<dim>()
        , size(size_)
        , a(size_ * numbers::PI / 2.0)
        {}

        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            if (size % 2 == 1)
                return std::sin(a * p[0]) * std::sin(a * p[1]);
            return std::cos(a * p[0]) * std::cos(a * p[1]);
        }

        double get_a() const {return a;}

    private:
        const unsigned int size;
        const double a;
};

// Usage: ./diffusion [key=value ...], e.g. ./diffusion max_level=4 n_patterns=2
//   min_level, max_level  levels i: N_el = 16 * 2^i cells per side, h = 2 / N_el = 0.125 / 2^i
//                         (default 0 and 7, the 8 mesh sizes of the paper; i = 7 has 4.2 million DoFs)
//   n_patterns            random coefficients for each pair (size, mode); default 4, i.e. 72 coefficients
//                         and, with 8 levels and 25 theta, 14400 samples (about the 15000 of Sec. 6.4.2)
//   eps                   exponents of mu uniform in (-eps, eps); default 8, so that two values of mu differ
//                         at most by 10^16 (double precision). The paper uses 20: then CG breaks down on some
//                         checkerboards (no theta converges, and in Debug mode deal.II stops at the breakdown)
//   max_iterations        CG iterations after which a solve stops as not converged (default 1000)
//   n_repetitions         repetitions of each solve, to average the timings (default 1)
//   seed                  seed of the random coefficients (default 0)
//   prefix                output files <prefix>_results.csv and <prefix>_patterns.csv (default diffusion)
// Arguments without '=' are left to PETSc (e.g. -log_view).
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    const bool is_root = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    std::map<std::string, std::string> options = {
        {"min_level", "0"}, {"max_level", "7"}, {"n_patterns", "4"}, {"eps", "8"},
        {"max_iterations", "1000"}, {"n_repetitions", "1"}, {"seed", "0"}, {"prefix", "diffusion"}};
    for (int k = 1; k < argc; ++k)
    {
        const std::string arg   = argv[k];
        const auto        equal = arg.find('=');
        if (equal == std::string::npos)
            continue;
        AssertThrow(options.count(arg.substr(0, equal)) == 1, ExcMessage("Unknown option " + arg));
        options[arg.substr(0, equal)] = arg.substr(equal + 1);
    }

    // Q1 elements (Sec. 5)
    const unsigned int degree = 1;

    // Dataset (Section 6.4)
    const unsigned int min_level      = std::stoul(options["min_level"]);
    const unsigned int max_level      = std::stoul(options["max_level"]);
    const unsigned int n_patterns     = std::stoul(options["n_patterns"]);
    const double       eps            = std::stod(options["eps"]);
    const unsigned int max_iterations = std::stoul(options["max_iterations"]);
    const unsigned int n_repetitions  = std::stoul(options["n_repetitions"]);
    const unsigned int seed           = std::stoul(options["seed"]);
    const std::string  prefix         = options["prefix"];

    const unsigned int N_el_0 = 16; // level 0: h = 2 / 16 = 0.125, the coarsest mesh of the paper
    const std::vector<unsigned int> size_values = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    const std::vector<unsigned int> mode_values = {1, 2}; // stripes, checkerboard

    // Strong threshold theta: 25 values in [0.02, 0.9] (Sec. 5), including the
    // default 0.25, which is the reference value rho_0.25 of the paper.
    std::vector<double> theta_values = {0.02, 0.25, 0.9};
    for (unsigned int k = 1; k <= 22; ++k)
        theta_values.push_back(0.04 * k);
    std::sort(theta_values.begin(), theta_values.end());

    // Error of the solution computed only for this theta.
    const double theta_check = 0.25;

    // Random coefficients, the same on every level and on every MPI process (same seed).
    std::vector<Pattern> patterns;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-eps, eps);
    for (const unsigned int size : size_values)
        for (const unsigned int mode : mode_values)
            for (unsigned int n = 0; n < n_patterns; ++n)
            {
                Pattern pattern{size, mode, std::vector<double>(mode == 1 ? size : size * size)};
                for (double &e : pattern.eps)
                    e = distribution(generator);
                patterns.push_back(pattern);
            }

    std::ofstream csv;
    if (is_root)
    {
        // Exponents written with all their digits, so that each mu can be rebuilt exactly.
        std::ofstream patterns_csv(prefix + "_patterns.csv");
        patterns_csv << "seed,pattern_id,size,mode,eps" << std::endl << std::setprecision(17);
        for (unsigned int p = 0; p < patterns.size(); ++p)
        {
            patterns_csv << seed << ',' << p << ',' << patterns[p].size << ',' << patterns[p].mode << ',';
            for (unsigned int k = 0; k < patterns[p].eps.size(); ++k)
                patterns_csv << (k > 0 ? ";" : "") << patterns[p].eps[k];
            patterns_csv << std::endl;
        }

        csv.open(prefix + "_results.csv");
        csv << "problem_id,seed,pattern_id,i,N_el,h,n_dofs,size,mode,theta,converged,iterations,rho,"
               "t_amg_setup,t_solve,t_solve_std,error_L2" << std::endl;
    }

    for (unsigned int i = min_level; i <= max_level; ++i)
    {
        const unsigned int N_el = N_el_0 << i; // 16 * 2^i

        for (unsigned int p = 0; p < patterns.size(); ++p)
        {
            // One id per matrix (seed, level, coefficient), the same in every run: it links the
            // rows of the same problem, also across runs (e.g. to the view of its matrix).
            const std::string problem_id = "s" + std::to_string(seed) + "_i" + std::to_string(i) + "_p" + std::to_string(p);
            const Pattern &pattern = patterns[p];

            if (is_root)
                std::cout << "#### Problem " << problem_id << " (N_el = " << N_el << ", size = " << pattern.size
                          << ", mode = " << pattern.mode << ")" << std::endl;

            const ExactSolution exact_solution(pattern.size);
            const double a = exact_solution.get_a();

            // -div(mu grad u) = f, no transport and no reaction
            const auto mu    = [&pattern](const Point<dim> &x) { return pattern(x); };
            const auto sigma = [](const Point<dim> &) { return 0.0; };
            const auto f     = [&](const Point<dim> &x) { return 2.0 * a * a * pattern(x) * exact_solution.value(x); };
            const auto b     = [](const Point<dim> &) { return Tensor<1, dim>(); };

            Diffusion problem(N_el, degree, mu, sigma, f, b);

            // Boundary Conditions: u = exact solution on the whole boundary (ids 0, 1, 2, 3)
            for (types::boundary_id id = 0; id < 4; ++id)
                problem.dirichlet[id] = &exact_solution;

            problem.setup();
            problem.assemble();

            // Rows of this problem, written at the end together with the error.
            std::vector<std::string> rows;
            double error_L2 = std::numeric_limits<double>::quiet_NaN();

            for (const double theta : theta_values)
            {
                const Diffusion::SolverResult result = problem.solve(theta, n_repetitions, max_iterations);
                const double rho = (result.iterations > 0) ? problem.compute_convergence_factor()
                                                           : std::numeric_limits<double>::quiet_NaN();

                // Check of the solver: u does not depend on mu, so the error must go to 0 with h when the
                // tiles are aligned with the mesh (size = 2, 4, 8). With the other sizes the cells across
                // the tile interfaces mix values of mu and the error can stay large: this changes the
                // discrete problem, not the linear solver, so rho is still a valid sample.
                if (std::abs(theta - theta_check) < 1e-12)
                    error_L2 = problem.compute_error(VectorTools::L2_norm, exact_solution);

                if (is_root)
                {
                    std::cout << "  rho = " << rho << std::endl;
                    std::ostringstream row;
                    row << problem_id << ',' << seed << ',' << p << ',' << i << ',' << N_el << ','
                        << problem.get_h() << ',' << problem.n_dofs() << ','
                        << pattern.size << ',' << pattern.mode << ',' << theta << ','
                        << result.converged << ',' << result.iterations << ','
                        << rho << ','
                        << result.time_amg_setup << ',' << result.time_solve << ','
                        << result.time_solve_std;
                    rows.push_back(row.str());
                }
            }

            if (is_root)
            {
                std::cout << "  L2 error (theta = " << theta_check << ") = " << error_L2 << std::endl;
                for (const std::string &row : rows)
                    csv << row << ',' << error_L2 << std::endl;
            }
        }
    }

    return 0;
}
