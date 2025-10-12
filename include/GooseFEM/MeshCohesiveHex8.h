/**
 * Generic mesh operations for 3D meshes including bulk and cohesive elements.
 *
 * (GooseFEM::MeshCohesive::Hex8).
 *
 * @file RegularCohesive3d.h
 * @copyright Copyright 2025. All rights reserved.
 */

#pragma once

#include "config.h"
#include "Element.h"
#include "MeshCohesive.h"
#include <array>

namespace GooseFEM {
namespace MeshCohesive {
namespace Hex8 {

/**
 * @brief Generates a regular 3D mesh with a planar layer of cohesive elements in the xy-plane.
 * This class includes functionality to add a small initial opening (kink) at one side of the cohesive interface (not implemented here).
 */
class RegularCohesive : public CohesiveMeshBase3d<RegularCohesive> {
public:
    RegularCohesive() = default;

    /**
     * @brief Constructor for a regular 3D mesh with a planar layer of cohesive elements.
     *
     * @param nelx Number of elements in x-direction for both bulk parts and cohesive interface.
     * @param nely Number of elements in y-direction for both bulk parts and cohesive interface.
     * @param nelz_lower Number of elements in z-direction for the lower bulk part.
     * @param nelz_upper Number of elements in z-direction for the upper bulk part.
     * @param h Edge size (width == height == depth) for each hexahedral element.
     */
    RegularCohesive(size_t nelx, size_t nely, size_t nelz_lower, size_t nelz_upper, double h = 1.0)
    {
        m_h = h;
        m_nelx = nelx;
        m_nely = nely;
        m_nelz_lower = nelz_lower;
        m_nelz_upper = nelz_upper;
        m_ndim = 3;
        m_nne_bulk = 8; // Hex8 elements
        m_nne_cohesive = 8; // Czm8 elements

        GOOSEFEM_ASSERT(m_nelx >= 1);
        GOOSEFEM_ASSERT(m_nely >= 1);
        GOOSEFEM_ASSERT(m_nelz_lower >= 1);
        GOOSEFEM_ASSERT(m_nelz_upper >= 1);

        // Calculate total nodes and elements
        m_nnode_lower_block = (m_nelx + 1) * (m_nely + 1) * (m_nelz_lower + 1);
        m_nnode_upper_block = (m_nelx + 1) * (m_nely + 1) * (m_nelz_upper + 1);
        m_nnode_total = m_nnode_lower_block + m_nnode_upper_block;

        m_nelem_bulk = m_nelx * m_nely * m_nelz_lower + m_nelx * m_nely * m_nelz_upper;
        m_nelem_cohesive = m_nelx * m_nely;
    }

private:
    friend class CohesiveMeshBase3d<RegularCohesive>;

    // --- Member variables ---
    double m_h;
    size_t m_nelx;
    size_t m_nely;
    size_t m_nelz_lower;
    size_t m_nelz_upper;
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
    size_t nely_impl() const
    {
        return m_nely;
    }
    size_t nelz_lower_impl() const
    {
        return m_nelz_lower;
    }
    size_t nelz_upper_impl() const
    {
        return m_nelz_upper;
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

        // Coordinates for lower bulk block (from z=0 to z=m_h*m_nelz_lower)
        array_type::tensor<double, 1> x_coords = xt::linspace<double>(0.0, m_h * static_cast<double>(m_nelx), m_nelx + 1);
        array_type::tensor<double, 1> y_coords = xt::linspace<double>(0.0, m_h * static_cast<double>(m_nely), m_nely + 1);
        array_type::tensor<double, 1> z_coords_lower = xt::linspace<double>(0.0, m_h * static_cast<double>(m_nelz_lower), m_nelz_lower + 1);

        size_t inode = 0;
        for (size_t iz = 0; iz < m_nelz_lower + 1; ++iz) {
            for (size_t iy = 0; iy < m_nely + 1; ++iy) {
                for (size_t ix = 0; ix < m_nelx + 1; ++ix) {
                    ret(inode, 0) = x_coords(ix);
                    ret(inode, 1) = y_coords(iy);
                    ret(inode, 2) = z_coords_lower(iz);
                    ++inode;
                }
            }
        }

        // Coordinates for upper bulk block (from z=interface_z to z=interface_z + m_h*m_nelz_upper)
        array_type::tensor<double, 1> z_coords_upper = xt::linspace<double>(0.0, m_h * static_cast<double>(m_nelz_upper), m_nelz_upper + 1);
        double interface_z_level = m_h * static_cast<double>(m_nelz_lower);

        for (size_t iz = 0; iz < m_nelz_upper + 1; ++iz) {
            for (size_t iy = 0; iy < m_nely + 1; ++iy) {
                for (size_t ix = 0; ix < m_nelx + 1; ++ix) {
                    ret(inode, 0) = x_coords(ix);
                    ret(inode, 1) = y_coords(iy);
                    ret(inode, 2) = z_coords_upper(iz) + interface_z_level;
                    ++inode;
                }
            }
        }
        // No kink offset applied
        return ret;
    }

    array_type::tensor<size_t, 2> conn_bulk_impl() const
    {
        array_type::tensor<size_t, 2> ret = xt::empty<size_t>({m_nelem_bulk, m_nne_bulk});
        size_t ielem = 0;

        // Connectivity for lower bulk block elements (Hex8)
        for (size_t iz = 0; iz < m_nelz_lower; ++iz) {
            for (size_t iy = 0; iy < m_nely; ++iy) {
                for (size_t ix = 0; ix < m_nelx; ++ix) {
                    size_t base = iz * (m_nelx + 1) * (m_nely + 1) + iy * (m_nelx + 1) + ix;
                    ret(ielem, 0) = base;
                    ret(ielem, 1) = base + 1;
                    ret(ielem, 2) = base + (m_nelx + 1) + 1;
                    ret(ielem, 3) = base + (m_nelx + 1);
                    ret(ielem, 4) = base + (m_nelx + 1) * (m_nely + 1);
                    ret(ielem, 5) = base + (m_nelx + 1) * (m_nely + 1) + 1;
                    ret(ielem, 6) = base + (m_nelx + 1) * (m_nely + 1) + (m_nelx + 1) + 1;
                    ret(ielem, 7) = base + (m_nelx + 1) * (m_nely + 1) + (m_nelx + 1);
                    ++ielem;
                }
            }
        }

        // Connectivity for upper bulk block elements (Hex8)
        for (size_t iz = 0; iz < m_nelz_upper; ++iz) {
            for (size_t iy = 0; iy < m_nely; ++iy) {
                for (size_t ix = 0; ix < m_nelx; ++ix) {
                    size_t base = m_nnode_lower_block + iz * (m_nelx + 1) * (m_nely + 1) + iy * (m_nelx + 1) + ix;
                    ret(ielem, 0) = base;
                    ret(ielem, 1) = base + 1;
                    ret(ielem, 2) = base + (m_nelx + 1) + 1;
                    ret(ielem, 3) = base + (m_nelx + 1);
                    ret(ielem, 4) = base + (m_nelx + 1) * (m_nely + 1);
                    ret(ielem, 5) = base + (m_nelx + 1) * (m_nely + 1) + 1;
                    ret(ielem, 6) = base + (m_nelx + 1) * (m_nely + 1) + (m_nelx + 1) + 1;
                    ret(ielem, 7) = base + (m_nelx + 1) * (m_nely + 1) + (m_nelx + 1);
                    ++ielem;
                }
            }
        }
        return ret;
    }

    array_type::tensor<size_t, 2> conn_cohesive_impl() const
    {
        array_type::tensor<size_t, 2> ret = xt::empty<size_t>({m_nelem_cohesive, m_nne_cohesive});

        size_t ielem = 0;
        for (size_t iy = 0; iy < m_nely; ++iy) {
            for (size_t ix = 0; ix < m_nelx; ++ix) {
                size_t base_lower = m_nelz_lower * (m_nelx + 1) * (m_nely + 1) + iy * (m_nelx + 1) + ix;
                size_t base_upper = m_nnode_lower_block + iy * (m_nelx + 1) + ix;

                // Lower surface nodes (0-3)
                ret(ielem, 0) = base_lower;
                ret(ielem, 1) = base_lower + 1;
                ret(ielem, 2) = base_lower + (m_nelx + 1) + 1;
                ret(ielem, 3) = base_lower + (m_nelx + 1);

                // Upper surface nodes (4-7)
                ret(ielem, 4) = base_upper;
                ret(ielem, 5) = base_upper + 1;
                ret(ielem, 6) = base_upper + (m_nelx + 1) + 1;
                ret(ielem, 7) = base_upper + (m_nelx + 1);
                ++ielem;
            }
        }
        return ret;
    }

    array_type::tensor<size_t, 1> nodesBottomFace_impl() const
    {
        return xt::arange<size_t>((m_nelx + 1) * (m_nely + 1));
    }

    array_type::tensor<size_t, 1> nodesTopFace_impl() const
    {
        return xt::arange<size_t>((m_nelx + 1) * (m_nely + 1)) + m_nnode_lower_block + m_nelz_upper * (m_nelx + 1) * (m_nely + 1);
    }

    array_type::tensor<size_t, 1> nodesLeftFace_impl() const
    {
        array_type::tensor<size_t, 1> nodes_lower = xt::arange<size_t>(m_nelz_lower + 1) * (m_nelx + 1) * (m_nely + 1);
        array_type::tensor<size_t, 1> nodes_upper = xt::arange<size_t>(m_nelz_upper + 1) * (m_nelx + 1) * (m_nely + 1) + m_nnode_lower_block;
        return xt::concatenate(xt::xtuple(nodes_lower, nodes_upper));
    }

    array_type::tensor<size_t, 1> nodesRightFace_impl() const
    {
        array_type::tensor<size_t, 1> nodes_lower = xt::arange<size_t>(m_nelz_lower + 1) * (m_nelx + 1) * (m_nely + 1) + m_nelx;
        array_type::tensor<size_t, 1> nodes_upper = xt::arange<size_t>(m_nelz_upper + 1) * (m_nelx + 1) * (m_nely + 1) + m_nelx + m_nnode_lower_block;
        return xt::concatenate(xt::xtuple(nodes_lower, nodes_upper));
    }

    array_type::tensor<size_t, 1> nodesFrontFace_impl() const
    {
        array_type::tensor<size_t, 1> nodes_lower = xt::arange<size_t>(m_nelz_lower + 1) * (m_nelx + 1) * (m_nely + 1);
        array_type::tensor<size_t, 1> nodes_upper = xt::arange<size_t>(m_nelz_upper + 1) * (m_nelx + 1) * (m_nely + 1) + m_nnode_lower_block;
        return xt::concatenate(xt::xtuple(nodes_lower, nodes_upper));
    }

    array_type::tensor<size_t, 1> nodesBackFace_impl() const
    {
        array_type::tensor<size_t, 1> nodes_lower = xt::arange<size_t>(m_nelz_lower + 1) * (m_nelx + 1) * (m_nely + 1) + m_nely * (m_nelx + 1);
        array_type::tensor<size_t, 1> nodes_upper = xt::arange<size_t>(m_nelz_upper + 1) * (m_nelx + 1) * (m_nely + 1) + m_nely * (m_nelx + 1) + m_nnode_lower_block;
        return xt::concatenate(xt::xtuple(nodes_lower, nodes_upper));
    }

    array_type::tensor<size_t, 1> nodesCohesiveLowerInterface_impl() const
    {
        return xt::arange<size_t>((m_nelx + 1) * (m_nely + 1)) + m_nelz_lower * (m_nelx + 1) * (m_nely + 1);
    }

    array_type::tensor<size_t, 1> nodesCohesiveUpperInterface_impl() const
    {
        return xt::arange<size_t>((m_nelx + 1) * (m_nely + 1)) + m_nnode_lower_block;
    }

    array_type::tensor<size_t, 1> elementsCohesive_impl() const
    {
        return xt::arange<size_t>(m_nelem_cohesive);
    }

    array_type::tensor<size_t, 3> elementgrid_bulk_lower_impl() const
    {
        return xt::arange<size_t>(m_nelx * m_nely * m_nelz_lower).reshape({m_nelz_lower, m_nely, m_nelx});
    }

    array_type::tensor<size_t, 3> elementgrid_bulk_upper_impl() const
    {
        return xt::arange<size_t>(m_nelx * m_nely * m_nelz_upper).reshape({m_nelz_upper, m_nely, m_nelx});
    }
};

} // namespace Hex8
} // namespace MeshCohesive
} // namespace GooseFEM