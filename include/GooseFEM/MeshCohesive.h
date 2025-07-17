/**
 * Generic mesh operations meshes including bulk and cohesive elements.
 * 
 * (GooseFEM::Mesh::MeshCohesive).
 *
 * @file MeshCohesive.h
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 *
 */

#pragma once

#include "config.h"
#include "Mesh.h"

namespace GooseFEM {
namespace MeshCohesive {

/**
 * @brief Base class for meshes that include cohesive elements, using CRTP.
 *
 * This base class defines the common interface for mesh generation,
 * ensuring derived classes implement the necessary methods to provide
 * coordinates, bulk and cohesive connectivity, and various node sets.
 * It uses the Curiously Recurring Template Pattern (CRTP).
 *
 * @tparam Derived The derived class inheriting from CohesiveMeshBase.
 */
template <class Derived>
class CohesiveMeshBase2d {
protected:
    /**
     * @brief Casts 'this' pointer to the derived type (non-const).
     * @return Reference to the derived object.
     */
    Derived& derived_cast() { return static_cast<Derived&>(*this); }

    /**
     * @brief Casts 'this' pointer to the derived type (const).
     * @return Const reference to the derived object.
     */
    const Derived& derived_cast() const { return static_cast<const Derived&>(*this); }

public:
    // --- Public Interface Methods ---

    auto dofs() const
    {
        return derived_cast().dofs_impl();
    }

    /** @brief Get the number of elements in the horizontal (x) direction. */
    auto nelx() const 
    {
        return derived_cast().nelx_impl();
    }

    /** @brief Get the number of elements in the vertical (y) direction for the lower bulk part. */
    auto nely_lower() const
    {
        return derived_cast().nely_lower_impl();
    }

    /** @brief Get the number of elements in the vertical (y) direction for the upper bulk part. */
    auto nely_upper() const
    {
        return derived_cast().nely_upper_impl();
    }

    /** @brief Get the total number of unique nodes in the mesh. */
    auto nnode() const
    {
        return derived_cast().nnode_total_impl();
    }

    /** @brief Get the spatial dimension of the mesh (e.g., 2 for 2D). */
    auto ndim() const
    {
        return derived_cast().ndim_impl();
    }

    /** @brief Get the number of nodes per bulk element. */
    auto nne_bulk() const
    {
        return derived_cast().nne_bulk_impl();
    }

    /** @brief Get the number of nodes per cohesive element. */
    auto nne_cohesive() const
    {
        return derived_cast().nne_cohesive_impl();
    }

    /** @brief Get the total number of bulk elements. */
    auto nelem_bulk() const
    {
        return derived_cast().nelem_bulk_impl();
    }

    /** @brief Get the total number of cohesive elements. */
    auto nelem_cohesive() const
    {
        return derived_cast().nelem_cohesive_impl();
    }

    /** @brief Get the characteristic edge size of the elements. */
    auto h() const
    {
        return derived_cast().h_impl();
    }

    /** @brief Get the global nodal coordinates. */
    auto coor() const
    {
        return derived_cast().coor_impl();
    }

    /** @brief Get the connectivity array for all bulk elements. */
    auto conn_bulk() const
    {
        return derived_cast().conn_bulk_impl();
    }

    /** @brief Get the connectivity array for all cohesive elements. */
    auto conn_cohesive() const
    {
        return derived_cast().conn_cohesive_impl();
    }

    // --- Boundary Node Sets ---

    /** @brief Get the global node IDs on the very bottom edge of the mesh. */
    auto nodesBottomEdge() const
    {
        return derived_cast().nodesBottomEdge_impl();
    }

    /** @brief Get the global node IDs on the very top edge of the mesh. */
    auto nodesTopEdge() const
    {
        return derived_cast().nodesTopEdge_impl();
    }

    /** @brief Get the global node IDs on the entire left edge of the mesh. */
    auto nodesLeftEdge() const
    {
        return derived_cast().nodesLeftEdge_impl();
    }

    /** @brief Get the global node IDs on the entire right edge of the mesh. */
    auto nodesRightEdge() const
    {
        return derived_cast().nodesRightEdge_impl();
    }

    /** @brief Get the global node IDs on the lower side of the cohesive interface. */
    auto nodesCohesiveLowerInterface() const
    {
        return derived_cast().nodesCohesiveLowerInterface_impl();
    }

    /** @brief Get the global node IDs on the upper side of the cohesive interface. */
    auto nodesCohesiveUpperInterface() const
    {
        return derived_cast().nodesCohesiveUpperInterface_impl();
    }

    /** @brief Get the global element IDs for all cohesive elements. */
    auto elementsCohesive() const
    {
        return derived_cast().elementsCohesive_impl();
    }

    // --- Additional useful functions ---

    /** @brief Get the element numbers of the lower bulk part as a 2D grid. */
    auto elementgrid_bulk_lower() const
    {
        return derived_cast().elementgrid_bulk_lower_impl();
    }

    /** @brief Get the element numbers of the upper bulk part as a 2D grid. */
    auto elementgrid_bulk_upper() const
    {
        return derived_cast().elementgrid_bulk_upper_impl();
    }
};

} // namespace Mesh
} // namespace GooseFEM