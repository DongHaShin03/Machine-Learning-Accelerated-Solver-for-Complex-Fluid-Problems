#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

using namespace dealii;

void make_cylinder_grid(Triangulation<2> &tria, const unsigned int n_refinements)
{
  GridGenerator::channel_with_cylinder(tria,
                                       /*shell_region_width=*/0.03,
                                       /*n_shells=*/2,
                                       /*skewness=*/2.0,
                                       /*colorize=*/true);
  tria.refine_global(n_refinements);
}

// Profilo di ingresso su Gamma_in: u_x = 4 U y (H - y) / H^2, u_y = 0,
// con massimo U a y = H/2. Ha 3 componenti (u_x, u_y, p) come un FESystem
// di Stokes in 2D; la componente di pressione vale 0 e va esclusa con una
// ComponentMask sulle sole velocità, ad esempio:
//
//   VectorTools::interpolate_boundary_values(dof_handler, 0, InflowVelocity(U),
//                                            constraints, fe.component_mask(velocities));
//   VectorTools::interpolate_boundary_values(dof_handler, 2,
//                                            Functions::ZeroFunction<2>(3),
//                                            constraints, fe.component_mask(velocities));
//   (idem per l'id 3; nessuna condizione da imporre sull'id 1)
class InflowVelocity : public Function<2>
{
public:
  explicit InflowVelocity(const double U, const double H = 0.41)
    : Function<2>(3), U(U), H(H)
  {}

  double value(const Point<2> &p, const unsigned int component) const override
  {
    if (component == 0)
      return 4.0 * U * p[1] * (H - p[1]) / (H * H);
    return 0.0;
  }

private:
  const double U, H;
};

// Conta le facce di bordo per ciascun boundary id.
std::map<types::boundary_id, unsigned int>
count_boundary_faces(const Triangulation<2> &tria)
{
  std::map<types::boundary_id, unsigned int> counts;
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary())
        ++counts[face->boundary_id()];
  return counts;
}

int main(int argc, char **argv)
{
  const unsigned int max_ref = (argc > 1) ? std::atoi(argv[1]) : 5;

  std::cout << "Canale con cilindro: (0,2.2)x(0,0.41) meno B_0.05(0.2,0.2)\n\n"
            << std::setw(4) << "i" << std::setw(10) << "celle"
            << std::setw(12) << "h_max" << std::setw(14) << "0.14/2^i"
            << std::setw(12) << "h_min" << "   facce di bordo per id\n";

  for (unsigned int i = 0; i <= max_ref; ++i)
    {
      Triangulation<2> tria;
      make_cylinder_grid(tria, i);

      std::cout << std::setw(4) << i << std::setw(10) << tria.n_active_cells()
                << std::setw(12) << std::setprecision(4)
                << GridTools::maximal_cell_diameter(tria) << std::setw(14)
                << 0.14 / std::pow(2.0, i) << std::setw(12)
                << GridTools::minimal_cell_diameter(tria) << "   ";
      for (const auto &[id, n] : count_boundary_faces(tria))
        std::cout << id << ":" << n << "  ";
      std::cout << "\n";

      // File da aprire in ParaView, solo per le mesh più piccole
      if (i <= 2)
        {
          std::ofstream vtu("mesh_ref" + std::to_string(i) + ".vtu");
          GridOut().write_vtu(tria, vtu);
        }
    }

  const InflowVelocity u0(/*U=*/1.0);
  std::cout << "\nBoundary id: 0 = Gamma_in, 1 = Gamma_out, "
               "2 = cilindro, 3 = pareti (Gamma_0 = 2 + 3)\n"
            << "Controllo profilo di ingresso con U = 1: u_x(0, 0) = "
            << u0.value(Point<2>(0, 0), 0)
            << ", u_x(0, 0.205) = " << u0.value(Point<2>(0, 0.205), 0)
            << ", u_x(0, 0.41) = " << u0.value(Point<2>(0, 0.41), 0) << "\n";
  return 0;
}