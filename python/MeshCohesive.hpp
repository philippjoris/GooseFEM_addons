/**
 * @file pyGooseFEM_MeshCohesive.h
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 *
 * Pybind11 bindings for GooseFEM cohesive mesh classes.
 */

#ifndef PYGOOSEFEM_MESH_COHESIVE_H
#define PYGOOSEFEM_MESH_COHESIVE_H

#include <GooseFEM/MeshCohesive.h> 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

namespace py = pybind11;

/**
 * @brief Helper function to register common properties of CohesiveMeshBase.
 * @tparam C The concrete C++ class (e.g., GooseFEM::Mesh::Quad4::RegularCohesive).
 * @tparam P The pybind11::class_ object for the concrete class.
 */
template <class C, class P>
void register_Mesh_CohesiveBase2d(P& cls)
{
    cls.def_property_readonly("nelx", &C::nelx,
                              "Get the number of elements in the horizontal (x) direction.");
    cls.def_property_readonly("nely_lower", &C::nely_lower,
                              "Get the number of elements in the vertical (y) direction for the lower bulk part.");
    cls.def_property_readonly("nely_upper", &C::nely_upper,
                              "Get the number of elements in the vertical (y) direction for the upper bulk part.");
    cls.def_property_readonly("nnode", &C::nnode,
                              "Get the total number of unique nodes in the mesh.");
    cls.def_property_readonly("ndim", &C::ndim,
                              "Get the spatial dimension of the mesh (e.g., 2 for 2D).");
    cls.def_property_readonly("dofs", &C::dofs,
                              "Get the dof vector of the mesh.");                              
    cls.def_property_readonly("nne_bulk", &C::nne_bulk,
                              "Get the number of nodes per bulk element.");
    cls.def_property_readonly("nne_cohesive", &C::nne_cohesive,
                              "Get the number of nodes per cohesive element.");
    cls.def_property_readonly("nelem_bulk", &C::nelem_bulk,
                              "Get the total number of bulk elements.");
    cls.def_property_readonly("nelem_cohesive", &C::nelem_cohesive,
                              "Get the total number of cohesive elements.");
    cls.def_property_readonly("h", &C::h,
                              "Get the characteristic edge size of the elements.");
    cls.def_property_readonly("coor", &C::coor,
                              "Get the global nodal coordinates.");
    cls.def_property_readonly("conn_bulk", &C::conn_bulk,
                              "Get the connectivity array for all bulk elements.");
    cls.def_property_readonly("conn_cohesive", &C::conn_cohesive,
                              "Get the connectivity array for all cohesive elements.");

    // Boundary Node Sets
    cls.def_property_readonly("nodesBottomEdge", &C::nodesBottomEdge,
                              "Get the global node IDs on the very bottom edge of the mesh.");
    cls.def_property_readonly("nodesTopEdge", &C::nodesTopEdge,
                              "Get the global node IDs on the very top edge of the mesh.");
    cls.def_property_readonly("nodesLeftEdge", &C::nodesLeftEdge,
                              "Get the global node IDs on the entire left edge of the mesh.");
    cls.def_property_readonly("nodesRightEdge", &C::nodesRightEdge,
                              "Get the global node IDs on the entire right edge of the mesh.");
    cls.def_property_readonly("nodesCohesiveLowerInterface", &C::nodesCohesiveLowerInterface,
                              "Get the global node IDs on the lower side of the cohesive interface.");
    cls.def_property_readonly("nodesCohesiveUpperInterface", &C::nodesCohesiveUpperInterface,
                              "Get the global node IDs on the upper side of the cohesive interface.");
    cls.def_property_readonly("elementsCohesive", &C::elementsCohesive,
                              "Get the global element IDs for all cohesive elements.");

    // Additional useful functions
    cls.def_property_readonly("elementgrid_bulk_lower", &C::elementgrid_bulk_lower,
                              "Get the element numbers of the lower bulk part as a 2D grid.");
    cls.def_property_readonly("elementgrid_bulk_upper", &C::elementgrid_bulk_upper,
                              "Get the element numbers of the upper bulk part as a 2D grid.");
}

#endif // PYGOOSEFEM_MESH_COHESIVE_H