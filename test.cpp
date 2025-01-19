/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2022 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Author: Simon Sticko, Uppsala University, 2021
 */

// The majority of the #include are taken from step85.
// The other ones are highlighted
#include <deal.II/base/function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h> // from step38

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h> // from step38

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h> // from step38
#include <deal.II/grid/grid_tools.h>   // for point_value-like function

#include <deal.II/grid/grid_out.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h> // from step38

#include <deal.II/numerics/data_out_stack.h> // for point_value()-like function

#include <fstream>
#include <vector>

#include <deal.II/base/function_signed_distance.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

using namespace dealii;

template <int spacedim>
class LaplaceBeltramiProblem;

template <int dim>
class CutFEMSolver
{
private:
    void make_grid();

    void setup_discrete_level_set();

    void distribute_dofs();

    void initialize_matrices();

    void preparation_for_assembly();

    void assemble_system(LaplaceBeltramiProblem<dim> &LaplaceBeltrami_solver);

    void solve();

    bool face_has_ghost_penalty(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        const unsigned int face_index) const;

    const unsigned int fe_degree;

    const Functions::ConstantFunction<dim> rhs_function;
    const double kb, ks;
    const double bb, bs;

    bool first_step = true;

    Triangulation<dim> triangulation;

    const FE_Q<dim> fe_level_set;
    DoFHandler<dim> level_set_dof_handler;
    Vector<double> level_set;

    hp::FECollection<dim> fe_collection;
    DoFHandler<dim> dof_handler;
    Vector<double> solution_bulk;
    Vector<double> old_solution_bulk;

    Vector<double> u_S;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> stiffness_matrix;
    Vector<double> rhs;

public:
    void output_results() const;

    double compute_L2_error_bulk() const;

    CutFEMSolver();

    void run(LaplaceBeltramiProblem<dim> &LaplaceBeltrami_solver);

    // function for sharing the solution on the bulk with the Laplace-Beltrami problem.
    // The resulting vector is defined over the bulk unfitted mesh and must be adjusted in Laplace-Beltrami problem
    Vector<double> get_solution_bulk()
    {
        return solution_bulk;
    }

    // function to set a new vector for the bulk solution.
    void set_u_S(Vector<double> &solution_surf)
    {
        u_S = solution_surf;
    }

    // function to return the dof_handler of the mesh associaed to the bulk.
    // Will be used in the Laplace-Beltrami problem.
    DoFHandler<dim> &get_bulk_dof_handler()
    {
        return dof_handler;
    }

    // function to set the vector of the bulk solution computed at the previous iteration.
    void set_old_u_B(Vector<double> &u_B_old)
    {
        old_solution_bulk = u_B_old;
    }

    // changes the value of the flag. Used to understand if we already own a surface solution (first_step = false)
    // or not (first_step = true).
    void set_first_step_false()
    {
        if (first_step)
            first_step = false;
    }
};

template <int spacedim>
class LaplaceBeltramiProblem
{
private:
    static constexpr unsigned int dim = spacedim - 1;

    // to be handled better
    const double ks, bs;
    const double kb, bb;

    // flag to understand if we are performing the first step. If true, we set u_B = 0,
    // otherwise we get it from the solution of the bulk problem.
    bool first_step = true;

    void make_grid_and_dofs();
    void solve();
    /*
    void assemble_system(CutFEMSolver<spacedim> &CutFEM_solver);
    void output_results() const;
    void compute_L2_error_surf() const;
    */

    Triangulation<dim, spacedim> triangulation;
    const FE_Q<dim, spacedim> fe;
    DoFHandler<dim, spacedim> dof_handler;
    MappingQ<dim, spacedim> mapping;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution_surf;
    Vector<double> old_solution_surf;
    Vector<double> system_rhs;

    // vector of the nodal values of solution_bulk. Must be interpolated over the circumference, hence it is not const
    Vector<double> u_B;

public:
    // public just to test
    void assemble_system(CutFEMSolver<spacedim> &CutFEM_solver);
    void output_results() const;
    double compute_L2_error_surf() const;

    LaplaceBeltramiProblem(const unsigned degree = 2);
    void run(CutFEMSolver<spacedim> &CutFEM_solver);

    // function for sharing the solution on the surface with the CutFEM bulk problem.
    // The resulting vector is defined over the mesh of the circumference and must adjusted in the bulk problem
    Vector<double> get_solution_surf()
    {
        return solution_surf;
    }

    // function to set a new vector for the bulk solution.
    void set_u_B(Vector<double> &solution_bulk)
    {
        u_B = solution_bulk;
    }

    // function to return the dof_handler of the mesh associaed to the circumferernce.
    // Will be used in the bulk problem.
    DoFHandler<dim, spacedim> &get_surf_dof_handler()
    {
        return dof_handler;
    }

    // changes the value of the flag. Used to understand if we already own a bulk solution (first_step = false)
    // or not (first_step = true).
    void set_first_step_false()
    {
        if (first_step)
            first_step = false;
    }

    // function to set the vector of the bulk solution computed at the previous iteration.
    void set_old_u_S(Vector<double> &u_S_old)
    {
        old_solution_surf = u_S_old;
    }

    // function to return the dof_handler of the mesh associaed to the bulk.
    // Will be used in the Laplace-Beltrami problem.
    MappingQ<dim, spacedim> &get_surf_mapping()
    {
        return mapping;
    }
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;
};

template <>
double RightHandSide<2>::value(const Point<2> &p,
                               const unsigned int /*component*/) const
{
    // return (p[0] * p[0] * p[1] * p[1]);
    // return std::sin(2 * numbers::PI * p[0]) * std::cos(2 * numbers::PI * p[1]);
    // return std::exp(-p[0] - p[1]);
    return (-8.0 * p[0] * p[1]);
}

template <int spacedim>
LaplaceBeltramiProblem<spacedim>::LaplaceBeltramiProblem(
    const unsigned degree)
    : ks(1.0),
      bs(1.0),
      kb(1.0),
      bb(1.0),
      fe(degree),
      dof_handler(triangulation),
      mapping(degree)
{
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::make_grid_and_dofs()
{
    {
        Triangulation<spacedim> volume_mesh;
        GridGenerator::hyper_ball(volume_mesh);

        const std::set<types::boundary_id> boundary_ids = {0};

        GridGenerator::extract_boundary_mesh(volume_mesh,
                                             triangulation,
                                             boundary_ids);
    }
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold(0, SphericalManifold<dim, spacedim>());

    triangulation.refine_global(3);

    std::cout << "Surface mesh has " << triangulation.n_active_cells()
              << " cells." << std::endl;

    dof_handler.distribute_dofs(fe);

    std::cout << "Surface mesh has " << dof_handler.n_dofs()
              << " degrees of freedom." << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution_surf.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::assemble_system(CutFEMSolver<spacedim> &CutFEM_solver)
{
    system_matrix = 0;
    system_rhs = 0;

    const QGauss<dim> quadrature_formula(2 * fe.degree);
    FEValues<dim, spacedim> fe_values(mapping,
                                      fe,
                                      quadrature_formula,
                                      update_values | update_gradients |
                                          update_quadrature_points |
                                          update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<double> rhs_values(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    RightHandSide<spacedim> rhs;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);

        rhs.value_list(fe_values.get_quadrature_points(), rhs_values);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                    cell_matrix(i, j) +=
                        (ks *
                             fe_values.shape_grad(i, q_point) *
                             fe_values.shape_grad(j, q_point) +
                         bs *
                             fe_values.shape_value(i, q_point) *
                             fe_values.shape_value(j, q_point)) *
                        fe_values.JxW(q_point);

        if (first_step)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    cell_rhs(i) +=
                        fe_values.shape_value(i, q_point) *
                        rhs_values[q_point] *
                        fe_values.JxW(q_point);
                }
        }

        else
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    const Point<spacedim> &point = fe_values.quadrature_point(q_point);
                    const double u_B_value = VectorTools::point_value(CutFEM_solver.get_bulk_dof_handler(),
                                                                      CutFEM_solver.get_solution_bulk(),
                                                                      point);

                    cell_rhs(i) +=
                        (fe_values.shape_value(i, q_point) *
                             rhs_values[q_point] +
                         bb *
                             u_B_value *
                             fe_values.shape_value(i, q_point)) *
                        fe_values.JxW(q_point);
                }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::solve()
{
    SolverControl solver_control(solution_surf.size(), 1e-7 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, solution_surf, system_rhs, preconditioner);
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::output_results() const
{
    DataOut<dim, spacedim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_surf,
                             "solution_surf",
                             DataOut<dim, spacedim>::type_dof_data);
    data_out.build_patches(mapping, mapping.get_degree());

    const std::string filename =
        "solution_surf-" + std::to_string(spacedim) + "d.vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
}

////////////// surface L2 error between consecutive steps //////////////
template <int spacedim>
double LaplaceBeltramiProblem<spacedim>::compute_L2_error_surf() const
{
    double l2_error_squared = 0.0;

    const QGauss<dim> quadrature_formula(2 * fe.degree);
    FEValues<dim, spacedim> fe_values(mapping,
                                      fe,
                                      quadrature_formula,
                                      update_values |
                                          update_quadrature_points |
                                          update_JxW_values);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        std::vector<double> solution_surf_values(fe_values.n_quadrature_points);
        fe_values.get_function_values(solution_surf, solution_surf_values);

        std::vector<double> old_solution_surf_values(fe_values.n_quadrature_points);
        fe_values.get_function_values(old_solution_surf, old_solution_surf_values);

        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
            const double error_at_point =
                solution_surf_values.at(q) - old_solution_surf_values.at(q);
            l2_error_squared +=
                Utilities::fixed_power<2>(error_at_point) * fe_values.JxW(q);
        }
    }

    return std::sqrt(l2_error_squared);
}

////////////////////////////// run the Laplace-Beltrami problem ////////////////////////////////////
template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::run(CutFEMSolver<spacedim> &CutFEM_solver)
{
    if (first_step)
        make_grid_and_dofs();

    assemble_system(CutFEM_solver);
    solve();
}

template <int dim>
CutFEMSolver<dim>::CutFEMSolver()
    : fe_degree(1),
      rhs_function(4.0),
      kb(1.0),
      ks(1.0),
      bb(1.0),
      bs(1.0),
      fe_level_set(fe_degree),
      level_set_dof_handler(triangulation),
      dof_handler(triangulation),
      mesh_classifier(level_set_dof_handler, level_set)
{
}

template <int dim>
void CutFEMSolver<dim>::make_grid()
{
    GridGenerator::hyper_cube(triangulation, -1.21, 1.21);
    triangulation.refine_global(3);
    std::cout << "Bulk mesh has " << triangulation.n_active_cells()
              << " cells." << std::endl;
}

template <int dim>
void CutFEMSolver<dim>::setup_discrete_level_set()
{
    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.n_dofs());

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
                             level_set);
}

enum ActiveFEIndex
{
    lagrange = 0, // intersected and inside cells
    nothing = 1   // outside cells
};

template <int dim>
void CutFEMSolver<dim>::distribute_dofs()
{
    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        const NonMatching::LocationToLevelSet cell_location =
            mesh_classifier.location_to_level_set(cell);

        if (cell_location == NonMatching::LocationToLevelSet::outside)
            cell->set_active_fe_index(ActiveFEIndex::nothing);
        else
            cell->set_active_fe_index(ActiveFEIndex::lagrange);
    }

    dof_handler.distribute_dofs(fe_collection);
    std::cout << "Bulk mesh has " << dof_handler.n_dofs()
              << " degrees of freedom." << std::endl;
}

template <int dim>
void CutFEMSolver<dim>::initialize_matrices()
{
    const auto face_has_flux_coupling = [&](const auto &cell,
                                            const unsigned int face_index)
    {
        return this->face_has_ghost_penalty(cell, face_index);
    };

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    const unsigned int n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;

    const AffineConstraints<double> constraints;
    const bool keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_flux_coupling);
    sparsity_pattern.copy_from(dsp);

    stiffness_matrix.reinit(sparsity_pattern);
    solution_bulk.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());
}

////////////////////////////// preparation for the assembly of the CutFEM problem ////////////////////////////////////
template <int dim>
void CutFEMSolver<dim>::preparation_for_assembly()
{
    make_grid();
    setup_discrete_level_set();
    mesh_classifier.reclassify();
    distribute_dofs();
    initialize_matrices();
}

template <int dim>
bool CutFEMSolver<dim>::face_has_ghost_penalty(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int face_index) const
{
    if (cell->at_boundary(face_index))
        return false;

    const NonMatching::LocationToLevelSet cell_location =
        mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
        mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
        return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
        return true;

    return false;
}

template <int dim>
void CutFEMSolver<dim>::assemble_system(LaplaceBeltramiProblem<dim> &LaplaceBeltrami_solver)
{
    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double> local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    const double ghost_parameter = 0.5;
    const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;

    const QGauss<dim - 1> face_quadrature(fe_degree + 1);
    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                                   update_JxW_values |
                                                   update_normal_vectors);

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
             IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
    {
        local_stiffness = 0;
        local_rhs = 0;

        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        const std::optional<FEValues<dim>> &inside_fe_values =
            non_matching_fe_values.get_inside_fe_values();

        if (inside_fe_values)
            for (const unsigned int q :
                 inside_fe_values->quadrature_point_indices())
            {
                const Point<dim> &point = inside_fe_values->quadrature_point(q);
                for (const unsigned int i : inside_fe_values->dof_indices())
                {
                    for (const unsigned int j : inside_fe_values->dof_indices())
                    {
                        local_stiffness(i, j) +=
                            kb *
                            inside_fe_values->shape_grad(i, q) *
                            inside_fe_values->shape_grad(j, q) *
                            inside_fe_values->JxW(q);
                    }
                    local_rhs(i) += rhs_function.value(point) / kb *
                                    inside_fe_values->shape_value(i, q) *
                                    inside_fe_values->JxW(q);
                }
            }

        const std::optional<NonMatching::FEImmersedSurfaceValues<dim>>
            &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
        {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
            {
                const Point<dim> &point =
                    surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                    surface_fe_values->normal_vector(q);

                double u_S_value;

                // first find the cell in which this point
                // is, initialize a quadrature rule with
                // it, and then a FEValues object
                const std::pair<typename DoFHandler<dim - 1, dim>::active_cell_iterator, Point<dim - 1>>
                    cell_point =
                        GridTools::find_active_cell_around_point(LaplaceBeltrami_solver.get_surf_mapping(),
                                                                 LaplaceBeltrami_solver.get_surf_dof_handler(),
                                                                 point);

                const Quadrature<dim - 1> quadrature(
                    cell_point.first->reference_cell().closest_point(cell_point.second));

                const FiniteElement<dim - 1, dim> &fe_S = LaplaceBeltrami_solver.get_surf_dof_handler().get_fe();

                FEValues<dim - 1, dim> fe_values_point(LaplaceBeltrami_solver.get_surf_mapping(), fe_S, quadrature, update_values);
                fe_values_point.reinit(cell_point.first);

                // then use this to get at the values of
                // the given fe_function at this point
                std::vector<Vector<double>> u_value(1, Vector<double>(fe_S.n_components()));
                fe_values_point.get_function_values(LaplaceBeltrami_solver.get_solution_surf(), u_value);

                u_S_value = u_value[0][0];

                for (const unsigned int i : surface_fe_values->dof_indices())
                {
                    for (const unsigned int j :
                         surface_fe_values->dof_indices())
                    {
                        local_stiffness(i, j) +=
                            (-((nitsche_parameter * cell_side_length) / (kb / bb + nitsche_parameter * cell_side_length)) *
                                 (normal * surface_fe_values->shape_grad(i, q) *
                                      surface_fe_values->shape_value(j, q) +
                                  normal * surface_fe_values->shape_grad(j, q) *
                                      surface_fe_values->shape_value(i, q)) +
                             (1.0 / (kb / bb + nitsche_parameter * cell_side_length)) *
                                 surface_fe_values->shape_value(i, q) *
                                 surface_fe_values->shape_value(j, q) -
                             ((kb / bb * nitsche_parameter * cell_side_length) / (kb / bb + nitsche_parameter * cell_side_length)) *
                                 (normal * surface_fe_values->shape_grad(i, q) *
                                  normal * surface_fe_values->shape_grad(j, q))) *
                            surface_fe_values->JxW(q);
                    }

                    local_rhs(i) +=
                        ((1.0 / (kb / bb + nitsche_parameter * cell_side_length)) *
                             bs / bb * u_S_value *
                             surface_fe_values->shape_value(i, q) -
                         ((nitsche_parameter * cell_side_length) / (kb / bb + nitsche_parameter * cell_side_length)) *
                             bs / bb * u_S_value *
                             normal * surface_fe_values->shape_grad(i, q)) *
                        surface_fe_values->JxW(q);
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);

        stiffness_matrix.add(local_dof_indices, local_stiffness);
        rhs.add(local_dof_indices, local_rhs);

        for (const unsigned int f : cell->face_indices())
            if (face_has_ghost_penalty(cell, f))
            {
                const unsigned int invalid_subface =
                    numbers::invalid_unsigned_int;

                fe_interface_values.reinit(cell,
                                           f,
                                           invalid_subface,
                                           cell->neighbor(f),
                                           cell->neighbor_of_neighbor(f),
                                           invalid_subface);

                const unsigned int n_interface_dofs =
                    fe_interface_values.n_current_interface_dofs();
                FullMatrix<double> local_stabilization(n_interface_dofs,
                                                       n_interface_dofs);
                for (unsigned int q = 0;
                     q < fe_interface_values.n_quadrature_points;
                     ++q)
                {
                    const Tensor<1, dim> normal = fe_interface_values.normal(q);
                    for (unsigned int i = 0; i < n_interface_dofs; ++i)
                        for (unsigned int j = 0; j < n_interface_dofs; ++j)
                        {
                            local_stabilization(i, j) +=
                                .5 * ghost_parameter * cell_side_length * normal *
                                fe_interface_values.jump_in_shape_gradients(i, q) *
                                normal *
                                fe_interface_values.jump_in_shape_gradients(j, q) *
                                fe_interface_values.JxW(q);
                        }
                }

                const std::vector<types::global_dof_index>
                    local_interface_dof_indices =
                        fe_interface_values.get_interface_dof_indices();

                stiffness_matrix.add(local_interface_dof_indices,
                                     local_stabilization);
            }
    }
}

template <int dim>
void CutFEMSolver<dim>::solve()
{
    const unsigned int max_iterations = 1e4;
    SolverControl solver_control(max_iterations);
    SolverCG<> solver(solver_control);
    solver.solve(stiffness_matrix, solution_bulk, rhs, PreconditionIdentity());
}

template <int dim>
void CutFEMSolver<dim>::output_results() const
{
    std::cout << "Writing vtu file" << std::endl;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution_bulk, "solution_bulk");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    data_out.set_cell_selection(
        [this](const typename Triangulation<dim>::cell_iterator &cell)
        {
            return cell->is_active() &&
                   mesh_classifier.location_to_level_set(cell) !=
                       NonMatching::LocationToLevelSet::outside;
        });

    data_out.build_patches();
    std::ofstream output("bulk_solution.vtu");
    data_out.write_vtu(output);
}

////////////// bulk L2 error between consecutive steps //////////////
template <int dim>
double CutFEMSolver<dim>::compute_L2_error_bulk() const
{
    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
        update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);
    double error_L2_squared = 0;

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
             IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
    {
        non_matching_fe_values.reinit(cell);

        const std::optional<FEValues<dim>> &fe_values =
            non_matching_fe_values.get_inside_fe_values();

        if (fe_values)
        {
            std::vector<double> solution_bulk_values(fe_values->n_quadrature_points);
            fe_values->get_function_values(solution_bulk, solution_bulk_values);

            std::vector<double> old_solution_bulk_values(fe_values->n_quadrature_points);
            fe_values->get_function_values(old_solution_bulk, old_solution_bulk_values);

            for (const unsigned int q : fe_values->quadrature_point_indices())
            {
                const double error_at_point =
                    solution_bulk_values.at(q) - old_solution_bulk_values.at(q);
                error_L2_squared +=
                    Utilities::fixed_power<2>(error_at_point) * fe_values->JxW(q);
            }
        }
    }

    return std::sqrt(error_L2_squared);
}

////////////////////////////// run the CutFEM problem ////////////////////////////////////
template <int dim>
void CutFEMSolver<dim>::run(LaplaceBeltramiProblem<dim> &LaplaceBeltrami_solver)
{
    if (first_step)
        preparation_for_assembly();
    assemble_system(LaplaceBeltrami_solver);
    solve();
}

int main()
{
    const int dim = 2;

    ////////////// iterative algorithm //////////////
    const unsigned int max_steps = 1500;
    const double tol = 1e-2;

    double err_u_B = tol + 1.0;
    double err_u_S = tol + 1.0;

    Vector<double> u_B_old, u_B_new;
    Vector<double> u_S_old, u_S_new;

    LaplaceBeltramiProblem<dim> LaplaceBeltrami_solver;
    CutFEMSolver<dim> CutFEM_solver;

    for (unsigned int step = 0; (err_u_B >= tol || err_u_S >= tol) && step < max_steps; step++)
    {
        std::cout << "Step " << step << std::endl;

        if (step > 0)
        {
            LaplaceBeltrami_solver.set_first_step_false();
            CutFEM_solver.set_first_step_false();
        }

        LaplaceBeltrami_solver.set_u_B(u_B_old);
        LaplaceBeltrami_solver.run(CutFEM_solver);
        u_S_new = LaplaceBeltrami_solver.get_solution_surf();

        if (step == 0)
        {
            u_S_old.reinit(u_S_new.size());
            u_S_old = 0.0;
        }

        LaplaceBeltrami_solver.set_old_u_S(u_S_old);
        err_u_S = LaplaceBeltrami_solver.compute_L2_error_surf();
        std::cout << "Surface residual: " << err_u_S << std::endl;

        CutFEM_solver.set_u_S(u_S_new);
        CutFEM_solver.run(LaplaceBeltrami_solver);
        u_B_new = CutFEM_solver.get_solution_bulk();

        if (step == 0)
        {
            u_B_old.reinit(u_B_new.size());
            u_B_old = 0.0;
        }

        CutFEM_solver.set_old_u_B(u_B_old);
        err_u_B = CutFEM_solver.compute_L2_error_bulk();
        std::cout << "Bulk residual: " << err_u_B << std::endl;
        std::cout << std::endl;

        u_S_old = u_S_new;
        u_B_old = u_B_new;
    }

    LaplaceBeltrami_solver.output_results();
    CutFEM_solver.output_results();
}
