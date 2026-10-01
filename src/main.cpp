#include "Stokes.hpp"
#include "preconditioners/BlockTriangular.hpp"
#include <ctime>

/**
 * Test case: Laboratory session 7
 */

static constexpr unsigned int dim = Stokes::dim; 

using Value = std::function<double(const Point<dim> &p)>; 
using VectorField = std::function<Tensor<1, dim>(const Point<dim> &p)>; 
using namespace std; 

class Inlet : public Function<dim> 
{
    public: 
        Inlet() : Function<dim>(dim + 1) {}

        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = -1 * alpha * p[1] * (2 - p[1]) * (1 - p[2]) * (2 - p[2]);

            for (unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
        }
    protected: 
        constexpr static double alpha = 1.0; 
}; 

class Neumann : public Function<dim> 
{
    public: 
        Neumann(){}
        virtual double value(const Point<dim> &, const unsigned int = 0) const override
        {
            return -p_out; 
        }
    protected: 
        constexpr static double p_out = 10.0; 
}; 

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const std::string  mesh_file_name  = "../mesh/mesh-step-5.msh";

    // Select Velocity elements' degree
    const unsigned int degree_velocity = 2;

    // Select Pressure elements' degree
    const unsigned int degree_pressure = 1;

    const auto sigma = [](const Point<dim> &)
    {
        return 0.0; 
    }; 

    // Select Viscosity term's value
    const auto nu = [](const Point<dim> &)
    {
        return 1.0; 
    }; 

    // Select Forcing term's value
    const auto f = [](const Point<dim> &)
    {
        Tensor<1, dim> result; 
        result[0] = 0.0; 
        result[1] = 0.0; 
        result[2] = 0.0; 
        return result; 
    }; 


    Stokes problem(mesh_file_name, degree_velocity, degree_pressure, sigma, nu, f);

    // Select Preconditioner
    auto preconditioner = std::make_unique<BlockTriangular>(); 
    problem.set_preconditioner(std::move(preconditioner)); 

    // Set Boundary Conditions
    Inlet inlet_velocity; 
    Functions::ZeroFunction<dim> zero_function; 
    Neumann neumann_bc; 

    problem.dirichlet[0] = &inlet_velocity;  
    problem.dirichlet[1] = &zero_function;  
    problem.neumann[2]   = &neumann_bc;  

    problem.setup();
    problem.assemble();

    clock_t t0 = clock(); 
    problem.solve();
    clock_t t1 = clock() - t0; 
    problem.output();

    std::cout << "Solving time: " << (double)t1/CLOCKS_PER_SEC << " seconds." << std::endl; 
}