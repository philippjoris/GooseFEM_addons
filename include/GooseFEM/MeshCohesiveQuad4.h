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
#include "Element.h"
#include "MeshCohesive.h"
#include "Element.h"
#include <array>

namespace GooseFEM {
namespace MeshCohesive {
namespace Quad4 {

/**
 * @brief Generates a regular 2D mesh with a horizontal layer of cohesive elements in the middle.
 * This class includes functionality to add a small initial opening (kink) at the left
 * side of the cohesive interface.
 */
class RegularCohesive : public CohesiveMeshBase2d<RegularCohesive> {
public:
    RegularCohesive() = default;

    /**
     * @brief Constructor for a regular 2D mesh with a horizontal layer of cohesive elements.
     *
     * @param nelx Number of elements in horizontal (x) direction for both bulk parts and cohesive interface.
     * @param nely_lower Number of elements in vertical (y) direction for the lower bulk part.
     * @param nely_upper Number of elements in vertical (y) direction for the upper bulk part.
     * @param h Edge size (width == height) for each quadrilateral element.
     */
    RegularCohesive(size_t nelx, size_t nely_lower, size_t nely_upper, double h = 1.0)
    {
        m_h = h;
        m_nelx = nelx;
        m_nely_lower = nely_lower;
        m_nely_upper = nely_upper;
        m_ndim = 2;
        m_nne_bulk = 4;
        m_nne_cohesive = 4;

        GOOSEFEM_ASSERT(m_nelx >= 1);
        GOOSEFEM_ASSERT(m_nely_lower >= 1);
        GOOSEFEM_ASSERT(m_nely_upper >= 1);

        // Calculate total nodes and elements
        m_nnode_lower_block = (m_nelx + 1) * (m_nely_lower + 1);
        m_nnode_upper_block = (m_nelx + 1) * (m_nely_upper + 1);
        m_nnode_total = m_nnode_lower_block + m_nnode_upper_block;

        m_nelem_bulk = m_nelx * m_nely_lower + m_nelx * m_nely_upper;
        m_nelem_cohesive = m_nelx;
    }

private:
    friend class CohesiveMeshBase2d<RegularCohesive>;

    // --- Member variables ---
    double m_h;
    size_t m_nelx;
    size_t m_nely_lower;
    size_t m_nely_upper;
    size_t m_ndim;
    size_t m_nne_bulk;
    size_t m_nne_cohesive;
    size_t m_nelem_bulk;
    size_t m_nelem_cohesive;
    size_t m_nnode_total;
    size_t m_nnode_lower_block;
    size_t m_nnode_upper_block;

    // --- Implementations for CohesiveMeshBase interface methods ---
    size_t nelx_impl() const
    {
        return m_nelx;
    }
    size_t nely_lower_impl() const
    {
        return m_nely_lower;
    }
    size_t nely_upper_impl() const
    {
        return m_nely_upper;
    }
    size_t nnode_total_impl() const
    {
        return m_nnode_total;
    }
    size_t ndim_impl() const
    {
        return m_ndim;
    }
    size_t nne_bulk_impl() const
    {
        return m_nne_bulk;
    }
    size_t nne_cohesive_impl() const
    {
        return m_nne_cohesive;
    }
    size_t nelem_bulk_impl() const
    {
        return m_nelem_bulk;
    }
    size_t nelem_cohesive_impl() const
    {
        return m_nelem_cohesive;
    }
    double h_impl() const
    {
        return m_h;
    }

    array_type::tensor<size_t, 2> dofs_impl() const
    {
        array_type::tensor<size_t, 2> dof_tensor = xt::empty<size_t>({m_nnode_total, m_ndim});

        for (size_t i = 0; i < m_nnode_total; ++i) {
            for (size_t j = 0; j < m_ndim; ++j) {
                dof_tensor(i, j) = static_cast<size_t>(i * m_ndim + j);
            }
        }
        return dof_tensor;
    }

    array_type::tensor<double, 2> coor_impl() const
    {
        array_type::tensor<double, 2> ret = xt::empty<double>({m_nnode_total, m_ndim});

        // Coordinates for lower bulk block (from y=0 to y=m_h*m_nely_lower)
        array_type::tensor<double, 1> x_coords = xt::linspace<double>(0.0, m_h * static_cast<double>(m_nelx), m_nelx + 1);
        array_type::tensor<double, 1> y_coords_lower = xt::linspace<double>(0.0, m_h * static_cast<double>(m_nely_lower), m_nely_lower + 1);

        size_t inode = 0;
        for (size_t iy = 0; iy < m_nely_lower + 1; ++iy) {
            for (size_t ix = 0; ix < m_nelx + 1; ++ix) {
                ret(inode, 0) = x_coords(ix);
                ret(inode, 1) = y_coords_lower(iy);
                ++inode;
            }
        }

        // Coordinates for upper bulk block (from y=interface_y to y=interface_y + m_h*m_nely_upper)
        array_type::tensor<double, 1> y_coords_upper = xt::linspace<double>(0.0, m_h * static_cast<double>(m_nely_upper), m_nely_upper + 1);
        double interface_y_level = m_h * static_cast<double>(m_nely_lower);

        for (size_t iy = 0; iy < m_nely_upper + 1; ++iy) {
            for (size_t ix = 0; ix < m_nelx + 1; ++ix) {
                ret(inode, 0) = x_coords(ix);
                ret(inode, 1) = y_coords_upper(iy) + interface_y_level;
                ++inode;
            }
        }
        // No kink offset applied
        return ret;
    }

    array_type::tensor<size_t, 2> conn_bulk_impl() const
    {
        array_type::tensor<size_t, 2> ret = xt::empty<size_t>({m_nelem_bulk, m_nne_bulk});
        size_t ielem = 0;

        // Connectivity for lower bulk block elements
        for (size_t iy = 0; iy < m_nely_lower; ++iy) {
            for (size_t ix = 0; ix < m_nelx; ++ix) {
                ret(ielem, 0) = (iy) * (m_nelx + 1) + (ix);
                ret(ielem, 1) = (iy) * (m_nelx + 1) + (ix + 1);
                ret(ielem, 2) = (iy + 1) * (m_nelx + 1) + (ix + 1);
                ret(ielem, 3) = (iy + 1) * (m_nelx + 1) + (ix);
                ++ielem;
            }
        }

        // Connectivity for upper bulk block elements
        for (size_t iy = 0; iy < m_nely_upper; ++iy) {
            for (size_t ix = 0; ix < m_nelx; ++ix) {
                ret(ielem, 0) = (iy) * (m_nelx + 1) + (ix) + m_nnode_lower_block;
                ret(ielem, 1) = (iy) * (m_nelx + 1) + (ix + 1) + m_nnode_lower_block;
                ret(ielem, 2) = (iy + 1) * (m_nelx + 1) + (ix + 1) + m_nnode_lower_block;
                ret(ielem, 3) = (iy + 1) * (m_nelx + 1) + (ix) + m_nnode_lower_block;
                ++ielem;
            }
        }
        return ret;
    }

    array_type::tensor<size_t, 2> conn_cohesive_impl() const
    {
        array_type::tensor<size_t, 2> ret = xt::empty<size_t>({m_nelem_cohesive, m_nne_cohesive});

        for (size_t ix = 0; ix < m_nelx; ++ix) {
            size_t node_lower_left = (m_nely_lower) * (m_nelx + 1) + ix;
            size_t node_lower_right = (m_nely_lower) * (m_nelx + 1) + (ix + 1);

            size_t node_upper_left = ix + m_nnode_lower_block;
            size_t node_upper_right = (ix + 1) + m_nnode_lower_block;

            ret(ix, 0) = node_lower_left;
            ret(ix, 1) = node_lower_right;
            ret(ix, 2) = node_upper_right;
            ret(ix, 3) = node_upper_left;
        }
        return ret;
    }

    array_type::tensor<size_t, 1> nodesBottomEdge_impl() const
    {
        return xt::arange<size_t>(m_nelx + 1);
    }

    array_type::tensor<size_t, 1> nodesTopEdge_impl() const
    {
        return xt::arange<size_t>(m_nelx + 1) + (m_nely_upper * (m_nelx + 1)) + m_nnode_lower_block;
    }

    array_type::tensor<size_t, 1> nodesLeftEdge_impl() const
    {
        array_type::tensor<size_t, 1> nodes_lower = xt::arange<size_t>(m_nely_lower + 1) * (m_nelx + 1);
        array_type::tensor<size_t, 1> nodes_upper = xt::arange<size_t>(m_nely_upper + 1) * (m_nelx + 1) + m_nnode_lower_block;
        return xt::concatenate(xt::xtuple(nodes_lower, nodes_upper));
    }

    array_type::tensor<size_t, 1> nodesRightEdge_impl() const
    {
        array_type::tensor<size_t, 1> nodes_lower = xt::arange<size_t>(m_nely_lower + 1) * (m_nelx + 1) + m_nelx;
        array_type::tensor<size_t, 1> nodes_upper = xt::arange<size_t>(m_nely_upper + 1) * (m_nelx + 1) + m_nelx + m_nnode_lower_block;
        return xt::concatenate(xt::xtuple(nodes_lower, nodes_upper));
    }

    array_type::tensor<size_t, 1> nodesCohesiveLowerInterface_impl() const
    {
        return xt::arange<size_t>(m_nelx + 1) + m_nely_lower * (m_nelx + 1);
    }

    array_type::tensor<size_t, 1> nodesCohesiveUpperInterface_impl() const
    {
        return xt::arange<size_t>(m_nelx + 1) + m_nnode_lower_block;
    }

    array_type::tensor<size_t, 1> elementsCohesive_impl() const
    {
        return xt::arange<size_t>(m_nelem_cohesive);
    }

    array_type::tensor<size_t, 2> elementgrid_bulk_lower_impl() const
    {
        return xt::arange<size_t>(m_nelx * m_nely_lower).reshape({m_nely_lower, m_nelx});
    }

    array_type::tensor<size_t, 2> elementgrid_bulk_upper_impl() const
    {
        return xt::arange<size_t>(m_nelx * m_nely_upper) + xt::arange<size_t>(m_nelx * m_nely_lower).reshape({m_nely_upper, m_nelx});
    }
};

} // namespace Quad4
} // namespace Mesh
} // namespace GooseFEM