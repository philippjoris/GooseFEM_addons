/**
 * Quadrature for 8-noded cohesive zone element in 3d (GooseFEM::Mesh::ElementType::Czm8),
 * in a Cartesian coordinate system.
 *
 * @file ElementCzm8.h
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 *
 * This class extends the concepts from Czm4 to 3D.
 *
 */

#ifndef GOOSEFEM_ELEMENTCZM8_H
#define GOOSEFEM_ELEMENTCZM8_H

#include "Element.h"
#include "config.h"
#include "detail.h"
#include <cmath> // For std::sqrt
#include <cstddef>
#include <array> // For std::array

namespace GooseFEM {
namespace Element {

/**
 * 8-noded cohesive zone element in 3d (GooseFEM::Mesh::ElementType::Czm8).
 *
 * Node ordering:
 *
 * Upper surface (z > 0, or "top"):
 * 7 -------- 6
 * |          |
 * |          |
 * 4 -------- 5
 *
 * Lower surface (z < 0, or "bottom"):
 * 3 -------- 2
 * |          |
 * |          |
 * 0 -------- 1
 *
 *
 * The local coordinates (xi, eta) define the surface, with zeta (normal direction) being implicit.
 */
namespace Czm8 {

/**
 * Gauss quadrature for 8-noded cohesive zone element.
 * Integration is performed over the area of the interface (2D integration).
 */
namespace Gauss {

/**
 * Number of integration points.
 * For 8-noded linear element (bilinear interpolation over the interface area),
 * 2x2 = 4 Gauss points are sufficient for exact integration.
 * @return unsigned int
 */
inline size_t nip()
{
    return 4;
}

/**
 * Integration point coordinates (local coordinates, on the interface plane).
 * These are 2D coordinates (xi, eta) in the range [-1, 1] x [-1, 1].
 * @return Coordinates [#nip, `ndim_local`], with `ndim_local = 2`.
 */
inline array_type::tensor<double, 2> xi()
{
    size_t nip = 4;
    size_t ndim_local = 2; // For 2D integration (xi, eta)

    array_type::tensor<double, 2> xi = xt::empty<double>({nip, ndim_local});

    // 2x2 Gauss quadrature for 2D (quadrilateral integration)
    double sqrt3_inv = 1.0 / std::sqrt(3.0);

    xi(0, 0) = -sqrt3_inv; xi(0, 1) = -sqrt3_inv; // Bottom-left
    xi(1, 0) = +sqrt3_inv; xi(1, 1) = -sqrt3_inv; // Bottom-right
    xi(2, 0) = +sqrt3_inv; xi(2, 1) = +sqrt3_inv; // Top-right
    xi(3, 0) = -sqrt3_inv; xi(3, 1) = +sqrt3_inv; // Top-left

    return xi;
}

/**
 * Integration point weights.
 * @return Weights [#nip].
 */
inline array_type::tensor<double, 1> w()
{
    size_t nip = 4; // For 2x2 Gauss quadrature

    array_type::tensor<double, 1> w = xt::empty<double>({nip});

    w(0) = 1.0;
    w(1) = 1.0;
    w(2) = 1.0;
    w(3) = 1.0;

    return w;
}

} // namespace Gauss

/**
 * Interpolation and quadrature for an 8-noded cohesive zone element.
 *
 * Fixed dimensions:
 * - `ndim = 3`: number of dimensions (for nodal displacements/coordinates).
 * - `nne = 8`: number of nodes per element.
 *
 * Naming convention:
 * - `elemmat`: element stiffness matrix, [#nelem, #nne * #ndim, #nne * #ndim]
 * - `elemvec`: nodal vectors (e.g., forces), [#nelem, #nne, #ndim]
 * - `qtensor`: integration point tensor (e.g., relative displacement, tractions),
 * [#nelem, #nip, #ndim] for relative displacement
 * [#nelem, #nip, #ndim, #ndim] for tangent stiffness of CZM.
 * - `qscalar`: integration point scalar (e.g., damage variable), [#nelem, #nip]
 */
class Quadrature : public QuadratureBase<Quadrature> {
public:
    Quadrature() = default;

    /**
     * Constructor: use default Gauss integration for cohesive elements (2x2 points over area).
     * @param x nodal coordinates (`elemvec`).
     */
    template <class T>
    Quadrature(const T& x) : Quadrature(x, Gauss::xi(), Gauss::w())
    {
    }

    /**
     * Constructor with custom integration.
     * @param x nodal coordinates (`elemvec`).
     * @param xi Integration point coordinates (local coordinates) [#nip, 2].
     * @param w Integration point weights [#nip].
     */
    template <class T, class X, class W>
    Quadrature(const T& x, const X& xi, const W& w)
    {
        m_x = x;
        m_w = w;
        m_xi = xi;
        m_nip = w.size();
        m_nelem = m_x.shape(0);
        m_N = xt::empty<double>({m_nip, s_nne});
        m_dNxi = xt::empty<double>({m_nip, s_nne, s_ndimxi}); // dN/dxi, dN/deta

        for (size_t q = 0; q < m_nip; ++q) {
            double loc_xi = xi(q, 0);
            double loc_eta = xi(q, 1);

            // Shape functions for a 4-noded quadrilateral (bilinear)
            // N_i(xi, eta) = 0.25 * (1 +/- xi) * (1 +/- eta)
            // Lower surface nodes (0, 1, 2, 3) map to (xi, eta)
            m_N(q, 0) = 0.25 * (1.0 - loc_xi) * (1.0 - loc_eta); 
            m_N(q, 1) = 0.25 * (1.0 + loc_xi) * (1.0 - loc_eta); 
            m_N(q, 2) = 0.25 * (1.0 + loc_xi) * (1.0 + loc_eta); 
            m_N(q, 3) = 0.25 * (1.0 - loc_xi) * (1.0 + loc_eta); 

            // Upper surface nodes (4, 5, 6, 7) also map to (xi, eta)
            m_N(q, 4) = 0.25 * (1.0 - loc_xi) * (1.0 - loc_eta); 
            m_N(q, 5) = 0.25 * (1.0 + loc_xi) * (1.0 - loc_eta); 
            m_N(q, 6) = 0.25 * (1.0 + loc_xi) * (1.0 + loc_eta); 
            m_N(q, 7) = 0.25 * (1.0 - loc_xi) * (1.0 + loc_eta); 
        }

        // Derivatives of shape functions with respect to local coordinates xi and eta
        for (size_t q = 0; q < m_nip; ++q) {
            double loc_xi = xi(q, 0);
            double loc_eta = xi(q, 1);

            // dN/dxi
            m_dNxi(q, 0, 0) = -0.25 * (1.0 - loc_eta); // Node 0
            m_dNxi(q, 1, 0) = +0.25 * (1.0 - loc_eta); // Node 1
            m_dNxi(q, 2, 0) = +0.25 * (1.0 + loc_eta); // Node 2
            m_dNxi(q, 3, 0) = -0.25 * (1.0 + loc_eta); // Node 3
            m_dNxi(q, 4, 0) = -0.25 * (1.0 - loc_eta); // Node 4
            m_dNxi(q, 5, 0) = +0.25 * (1.0 - loc_eta); // Node 5
            m_dNxi(q, 6, 0) = +0.25 * (1.0 + loc_eta); // Node 6
            m_dNxi(q, 7, 0) = -0.25 * (1.0 + loc_eta); // Node 7

            // dN/deta
            m_dNxi(q, 0, 1) = -0.25 * (1.0 - loc_xi); // Node 0
            m_dNxi(q, 1, 1) = -0.25 * (1.0 + loc_xi); // Node 1
            m_dNxi(q, 2, 1) = +0.25 * (1.0 + loc_xi); // Node 2
            m_dNxi(q, 3, 1) = +0.25 * (1.0 - loc_xi); // Node 3
            m_dNxi(q, 4, 1) = -0.25 * (1.0 - loc_xi); // Node 4
            m_dNxi(q, 5, 1) = -0.25 * (1.0 + loc_xi); // Node 5
            m_dNxi(q, 6, 1) = +0.25 * (1.0 + loc_xi); // Node 6
            m_dNxi(q, 7, 1) = +0.25 * (1.0 - loc_xi); // Node 7
        }

        GOOSEFEM_ASSERT(m_x.shape(1) == s_nne);
        GOOSEFEM_ASSERT(m_x.shape(2) == s_ndim);
        GOOSEFEM_ASSERT(xt::has_shape(m_xi, {m_nip, s_ndimxi}));
        GOOSEFEM_ASSERT(xt::has_shape(m_w, {m_nip}));
        GOOSEFEM_ASSERT(xt::has_shape(m_N, {m_nip, s_nne}));
        GOOSEFEM_ASSERT(xt::has_shape(m_dNxi, {m_nip, s_nne, s_ndimxi})); // Two local derivatives needed

        m_vol = xt::empty<double>(this->shape_qscalar()); 

        m_rotation_matrix = xt::empty<double>({m_nelem, m_nip, s_ndim, s_ndim});

        this->compute_kinematics();
    }

    /**
    * @brief Computes the relative displacement and transforms it into normal and tangential components.
    *
    * @param elem_u Nodal displacements for elements (`elem_tensor<1>`), [#nelem, #nne, #ndim].
    * @param q_delta_u_norm_tan Output: Normal and two tangential relative displacements
    * (`qtensor<1>`), [#nelem, #nip, #ndim].
    * q_delta_u_norm_tan(e, q, 0) = delta_u_normal
    * q_delta_u_norm_tan(e, q, 1) = delta_u_tangential1
    * q_delta_u_norm_tan(e, q, 2) = delta_u_tangential2
    */
    template <class T, class R, class O>
    void relative_disp(const T& elem_u, R& q_delta_u_norm_tan, O& rot_mat) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(elem_u, this->shape_elemvec()));
        GOOSEFEM_ASSERT(xt::has_shape(q_delta_u_norm_tan, this->shape_qvector()));
        GOOSEFEM_ASSERT(xt::has_shape(rot_mat, this->shape_rotmatrix()));

        // 1. Calculate the global relative displacement
        // Create a temporary tensor to hold the global relative displacement
        xt::pytensor<double, 3> q_delta_u_global = xt::zeros<double>(this->shape_qvector());

    #pragma omp parallel for
        for (ptrdiff_t e = 0; e < (ptrdiff_t)m_nelem; ++e) {
            auto u = xt::adapt(&elem_u(e, 0, 0), xt::xshape<s_nne, s_ndim>());

            for (ptrdiff_t q = 0; q < (ptrdiff_t)m_nip; ++q) {
                // Interpolate displacements for lower surface nodes (0,1,2,3)
                array_type::tensor<double, 1> u_lower = xt::empty<double>({s_ndim});
                u_lower(0) = m_N(q, 0) * u(0, 0) + m_N(q, 1) * u(1, 0) + m_N(q, 2) * u(2, 0) + m_N(q, 3) * u(3, 0);
                u_lower(1) = m_N(q, 0) * u(0, 1) + m_N(q, 1) * u(1, 1) + m_N(q, 2) * u(2, 1) + m_N(q, 3) * u(3, 1);
                u_lower(2) = m_N(q, 0) * u(0, 2) + m_N(q, 1) * u(1, 2) + m_N(q, 2) * u(2, 2) + m_N(q, 3) * u(3, 2);

                // Interpolate displacements for upper surface nodes (4,5,6,7)
                array_type::tensor<double, 1> u_upper = xt::empty<double>({s_ndim});
                u_upper(0) = m_N(q, 4) * u(4, 0) + m_N(q, 5) * u(5, 0) + m_N(q, 6) * u(6, 0) + m_N(q, 7) * u(7, 0);
                u_upper(1) = m_N(q, 4) * u(4, 1) + m_N(q, 5) * u(5, 1) + m_N(q, 6) * u(6, 1) + m_N(q, 7) * u(7, 1);
                u_upper(2) = m_N(q, 4) * u(4, 2) + m_N(q, 5) * u(5, 2) + m_N(q, 6) * u(6, 2) + m_N(q, 7) * u(7, 2);

                q_delta_u_global(e, q, 0) = u_upper(0) - u_lower(0);
                q_delta_u_global(e, q, 1) = u_upper(1) - u_lower(1);
                q_delta_u_global(e, q, 2) = u_upper(2) - u_lower(2);
            }
        }

        // 2. Transform the global relative displacement into normal and tangential components
        q_delta_u_norm_tan.fill(0.0); 
        rot_mat.fill(0.0);

    #pragma omp parallel for
        for (ptrdiff_t e = 0; e < (ptrdiff_t)m_nelem; ++e) {
            for (ptrdiff_t q = 0; q < (ptrdiff_t)m_nip; ++q) {
                auto delta_u = xt::adapt(&q_delta_u_global(e, q, 0), xt::xshape<s_ndim>());
                auto rot = xt::adapt(&m_rotation_matrix(e, q, 0, 0), xt::xshape<s_ndim, s_ndim>());
                auto rot_output = xt::adapt(&rot_mat(e, q, 0, 0), xt::xshape<s_ndim, s_ndim>());

                // Apply rotation: q_delta_u_norm_tan = R * delta_u_global
                for (ptrdiff_t i = 0; i < (ptrdiff_t)s_ndim; ++i) { 
                    for (ptrdiff_t j = 0; j < (ptrdiff_t)s_ndim; ++j) { 
                        q_delta_u_norm_tan(e, q, i) += rot(i, j) * delta_u(j);
                    }
                }

                rot_output = rot;
            }
        }
    }

    /**
     * @brief Computes the element force vector (residual).
     * This function integrates tractions over the interface area to get nodal forces.
     *
     * @param q_tractions Tractions at integration points (`qtensor<1>`), [#nelem, #nip, #ndim].
     * q_tractions(e, q, :) are the traction components in global coordinates.
     * @param elem_f Output: Nodal force vector (`elemvec`), [#nelem, #nne, #ndim].
     */
    template <class T, class R>
    void int_N_dot_traction_dA(const T& q_tractions, R& elem_f) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(q_tractions, this->shape_qvector()));
        GOOSEFEM_ASSERT(xt::has_shape(elem_f, this->shape_elemvec()));

        elem_f.fill(0.0);

    #pragma omp parallel for
        for (ptrdiff_t e = 0; e < (ptrdiff_t)m_nelem; ++e) {
            auto f = xt::adapt(&elem_f(e, 0, 0), xt::xshape<s_nne, s_ndim>());

            for (ptrdiff_t q = 0; q < (ptrdiff_t)m_nip; ++q) {
                auto tractions = xt::adapt(&q_tractions(e, q, 0), xt::xshape<s_ndim>());
                auto& dA = m_vol(e, q);

                // CORRECTED: Calculate the contribution to the force vector using the B-matrix logic.
                // For a cohesive element, the force is correctly balanced between the top and bottom faces.
                // The nodal force integral is F_int = Integral(B^T * T * dA).
                // This is implemented by summing contributions from each node.
                
                // For each of the 4 pairs of nodes (0-4, 1-5, 2-6, 3-7)
                for (ptrdiff_t pair_idx = 0; pair_idx < 4; ++pair_idx) {
                    size_t bottom_node_idx = pair_idx;
                    size_t top_node_idx = pair_idx + 4;
                    
                    // N_bottom is the shape function for the bottom node.
                    double N_bottom = m_N(q, bottom_node_idx);
                    // N_top is the shape function for the top node.
                    double N_top = m_N(q, top_node_idx);
                    
                    // For each spatial dimension (x, y, z)
                    for (ptrdiff_t dim = 0; dim < (ptrdiff_t)s_ndim; ++dim) {
                        // The force on the bottom face is negative.
                        f(bottom_node_idx, dim) += -N_bottom * tractions(dim) * dA;
                        
                        // The force on the top face is positive.
                        f(top_node_idx, dim) += N_top * tractions(dim) * dA;
                    }
                }
            }
        }
    }

    template <class T>
    array_type::tensor<double, 3> Int_N_dot_traction_dA(const T& q_tractions) const
    {
        auto elem_f = array_type::tensor<double, 3>::from_shape(this->shape_elemvec());
        this->int_N_dot_traction_dA(q_tractions, elem_f);
        return elem_f;
    }

    /**
     * @param q_tangent_stiffness_global [#nelem, #nip, #ndim, #ndim]
     * @return elemmat [#nelem, #nne * #ndim, #nne * #ndim]
     */
    template <class T>
    auto Int_BT_D_B_dA(const T& q_tangent_stiffness_global) const
        -> array_type::tensor<double, 3>
    {
        auto elemmat = array_type::tensor<double, 3>::from_shape({
        static_cast<std::ptrdiff_t>(m_nelem),
        static_cast<std::ptrdiff_t>(s_nne * s_ndim),
        static_cast<std::ptrdiff_t>(s_nne * s_ndim)
        });
        this->int_BT_D_B_dA(q_tangent_stiffness_global, elemmat);
        return elemmat;
    }

    /**
     * @brief Computes the element stiffness matrix.
     *
     * @param q_tangent_stiffness_global Tangent stiffness of the cohesive law at integration points
     * in global coordinates (`qtensor<2>`), [#nelem, #nip, #ndim, #ndim].
     * This matrix directly maps global relative displacement increment
     * to global traction increment.
     * @param elem_K Output: Element stiffness matrix (`elemmat`), [#nelem, #nne * #ndim, #nne * #ndim].
     */
    template <class T, class R>
    void int_BT_D_B_dA(const T& q_tangent_stiffness_global, R& elem_K) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(q_tangent_stiffness_global, this->shape_qtensor<2>()));
        GOOSEFEM_ASSERT(xt::has_shape(elem_K, this->shape_elemmat()));

        elem_K.fill(0.0);

    #pragma omp parallel for
        for (ptrdiff_t e = 0; e < (ptrdiff_t)m_nelem; ++e) {
            auto K_elem = xt::adapt(&elem_K(e, 0, 0), xt::xshape<s_nne * s_ndim, s_nne * s_ndim>());

            for (ptrdiff_t q = 0; q < (ptrdiff_t)m_nip; ++q) {
                auto D_T_global = xt::adapt(&q_tangent_stiffness_global(e, q, 0, 0), xt::xshape<s_ndim, s_ndim>());
                auto& dA = m_vol(e, q);

                // Define the local 'B_coh' matrix for this integration point
                // This matrix maps global nodal displacements to global separation (delta_x, delta_y, delta_z)
                // It's a 3x(8*3) matrix: [delta_x; delta_y; delta_z] = B_coh * [u0x, u0y, u0z, ..., u7z]^T

                // Initialize B_coh (3 rows, 24 columns) with zeros
                array_type::tensor<double, 2> B_coh = xt::zeros<double>({s_ndim, s_nne * s_ndim});

                // CORRECTED: Populate B_coh based on shape functions for lower (0-3) and upper (4-7) nodes
                // The separation is computed as the displacement of the top nodes minus the displacement of the corresponding bottom nodes.
                // For example, top node 4 corresponds to bottom node 0.

                for (ptrdiff_t pair_idx = 0; pair_idx < 4; ++pair_idx) {
                    size_t bottom_node_idx = pair_idx;
                    size_t top_node_idx = pair_idx + 4;

                    for (ptrdiff_t dim = 0; dim < (ptrdiff_t)s_ndim; ++dim) {
                        // Contribution from the bottom nodes (negative sign)
                        B_coh(dim, bottom_node_idx * s_ndim + dim) = -m_N(q, bottom_node_idx);
                        
                        // Contribution from the top nodes (positive sign)
                        B_coh(dim, top_node_idx * s_ndim + dim) = m_N(q, top_node_idx);
                    }
                }

                // The rest of the function remains the same, as it correctly performs the matrix multiplication and summation.
                // Calculate K_elem contribution for this integration point
                // K_elem += (B_coh.transpose() * D_T_global * B_coh) * dA;

                xt::xtensor_fixed<double, xt::xshape<s_ndim, s_nne * s_ndim>> Temp;
                Temp.fill(0.0);

                for (ptrdiff_t i = 0; i < (ptrdiff_t)s_ndim; ++i) { 
                    for (ptrdiff_t k = 0; k < (ptrdiff_t)s_nne * s_ndim; ++k) { 
                        for (ptrdiff_t j = 0; j < (ptrdiff_t)s_ndim; ++j) { 
                            Temp(i, k) += D_T_global(i, j) * B_coh(j, k);
                        }
                    }
                }

                for (ptrdiff_t m = 0; m < (ptrdiff_t)s_nne * s_ndim; ++m) { 
                    for (ptrdiff_t n = 0; n < (ptrdiff_t)s_nne * s_ndim; ++n) { 
                        double sum_val = 0.0;
                        for (ptrdiff_t p = 0; p < (ptrdiff_t)s_ndim; ++p) { 
                            sum_val += B_coh(p, m) * Temp(p, n);
                        }
                        K_elem(m, n) += sum_val * dA;
                    }
                }
            }
        }
    }

    template <class T>
    void update_x(const T& x)
    {
        GOOSEFEM_ASSERT(xt::has_shape(x, this->m_x.shape()));
        xt::noalias(this->m_x) = x;
        this->compute_kinematics();
    }

    /**
     * Get the full shape of the rotation matrix (a "qtensor" of rank 2).
     * @returns [#nelem, #nip, #ndim, #ndim].
     */
    auto shape_rotmatrix() const -> std::array<size_t, 4>
    {
        return std::array<size_t, 4>{this->m_nelem, this->m_nip, this->s_ndim, this->s_ndim};
    }

private:
    friend QuadratureBase<Quadrature>;
    friend QuadratureBaseCartesian<Quadrature>;

    void compute_kinematics()
    {
    #pragma omp parallel
        {
            // Jacobian matrix J = [dX/dxi, dX/deta; dY/dxi, dY/deta; dZ/dxi, dZ/deta]
            // This maps local (xi, eta) to global (X, Y, Z) on the surface
            array_type::tensor<double, 2> J_surface = xt::empty<double>({s_ndim, s_ndimxi}); 

    #pragma omp for
            for (ptrdiff_t e = 0; e < (ptrdiff_t)m_nelem; ++e) {
                auto x = xt::adapt(&m_x(e, 0, 0), xt::xshape<s_nne, s_ndim>());

                for (ptrdiff_t q = 0; q < (ptrdiff_t)m_nip; ++q) {

                    J_surface.fill(0.0);

                    // Loop over all 8 nodes
                    for (ptrdiff_t node_idx = 0; node_idx < (ptrdiff_t)s_nne; ++node_idx) {
                        // dX/dxi, dY/dxi, dZ/dxi 
                        J_surface(0, 0) += m_dNxi(q, node_idx, 0) * x(node_idx, 0);
                        J_surface(1, 0) += m_dNxi(q, node_idx, 0) * x(node_idx, 1);
                        J_surface(2, 0) += m_dNxi(q, node_idx, 0) * x(node_idx, 2);

                        // dX/deta, dY/deta, dZ/deta 
                        J_surface(0, 1) += m_dNxi(q, node_idx, 1) * x(node_idx, 0);
                        J_surface(1, 1) += m_dNxi(q, node_idx, 1) * x(node_idx, 1);
                        J_surface(2, 1) += m_dNxi(q, node_idx, 1) * x(node_idx, 2);
                    }

                    // Calculate normal vector and area Jacobian from J_surface
                    // Normal vector is proportional to the cross product of the tangent vectors
                    // t1 = (J_surface(0,0), J_surface(1,0), J_surface(2,0)) -> dX/dxi vector
                    // t2 = (J_surface(0,1), J_surface(1,1), J_surface(2,1)) -> dX/deta vector

                    double t1_x = J_surface(0, 0);
                    double t1_y = J_surface(1, 0);
                    double t1_z = J_surface(2, 0);

                    double t2_x = J_surface(0, 1);
                    double t2_y = J_surface(1, 1);
                    double t2_z = J_surface(2, 1);

                    // Normal vector (un-normalized) from cross product t1 x t2
                    double n_x_unnorm = t1_y * t2_z - t1_z * t2_y;
                    double n_y_unnorm = t1_z * t2_x - t1_x * t2_z;
                    double n_z_unnorm = t1_x * t2_y - t1_y * t2_x;

                    double normal_mag = std::sqrt(n_x_unnorm * n_x_unnorm +
                                                  n_y_unnorm * n_y_unnorm +
                                                  n_z_unnorm * n_z_unnorm);
                    // std::cout << "nx, ny, nz: " << n_x_unnorm << ", " << n_y_unnorm << ", " << n_z_unnorm << std::endl;
                    GOOSEFEM_ASSERT(normal_mag > 1e-12); 

                    // Unit normal vector
                    double n_x = n_x_unnorm / normal_mag;
                    double n_y = n_y_unnorm / normal_mag;
                    double n_z = n_z_unnorm / normal_mag;

                    m_vol(e, q) = m_w(q) * normal_mag;

                    double t1_mag = std::sqrt(t1_x * t1_x + t1_y * t1_y + t1_z * t1_z);
                    GOOSEFEM_ASSERT(t1_mag > 1e-12);
                    double t1_x_unit = t1_x / t1_mag;
                    double t1_y_unit = t1_y / t1_mag;
                    double t1_z_unit = t1_z / t1_mag;

                    double t2_x_ortho = n_y * t1_z_unit - n_z * t1_y_unit;
                    double t2_y_ortho = n_z * t1_x_unit - n_x * t1_z_unit;
                    double t2_z_ortho = n_x * t1_y_unit - n_y * t1_x_unit;

                    double t2_mag = std::sqrt(t2_x_ortho * t2_x_ortho + t2_y_ortho * t2_y_ortho + t2_z_ortho * t2_z_ortho);
                    GOOSEFEM_ASSERT(t2_mag > 1e-12);
                    t2_x_ortho /= t2_mag;
                    t2_y_ortho /= t2_mag;
                    t2_z_ortho /= t2_mag;

                    // Rotation matrix R = [[n_x,   n_y,  n_z],
                    //                      [t1_x, t1_y, t1_z],
                    //                      [t2_x, t2_y, t2_z]]
                    m_rotation_matrix(e, q, 0, 0) = n_x;
                    m_rotation_matrix(e, q, 0, 1) = n_y;
                    m_rotation_matrix(e, q, 0, 2) = n_z;

                    m_rotation_matrix(e, q, 1, 0) = t1_x_unit;
                    m_rotation_matrix(e, q, 1, 1) = t1_y_unit;
                    m_rotation_matrix(e, q, 1, 2) = t1_z_unit;

                    m_rotation_matrix(e, q, 2, 0) = t2_x_ortho;
                    m_rotation_matrix(e, q, 2, 1) = t2_y_ortho;
                    m_rotation_matrix(e, q, 2, 2) = t2_z_ortho;
                }
            }
        }
    }

    constexpr static size_t s_nne = 8;    ///< Number of nodes per element.
    constexpr static size_t s_ndim = 3;   ///< Number of dimensions for nodal vectors (x,y,z).
    constexpr static size_t s_ndimxi = 2; ///< Number of dimensions for shape function derivative (xi, eta).
    constexpr static size_t s_tdim = 3;   ///< For tensors like tangent stiffness in global coords (3x3).
    size_t m_tdim = 3;                    ///< Dynamic alias of s_tdim
    size_t m_nelem;                       ///< Number of elements.
    size_t m_nip;                         ///< Number of integration points per element.
    array_type::tensor<double, 3> m_x;    ///< nodal positions stored per element [#nelem, #nne, #ndim]
    array_type::tensor<double, 1> m_w;    ///< weight of each integration point [nip]
    array_type::tensor<double, 2> m_xi;   ///< local coordinate per integration point [#nip, #ndimxi]
    array_type::tensor<double, 2> m_N;    ///< shape functions [#nip, #nne]
    array_type::tensor<double, 3> m_dNxi; ///< local shape func grad [#nip, #nne, #ndimxi] (two local coordinates)
    array_type::tensor<double, 2> m_vol;  ///< integration point "volume" (effective area * weight) [#nelem, #nip]
    array_type::tensor<double, 4> m_rotation_matrix; ///< rotation matrix of local element [#nelem, #nip, 3, 3]
};

} // namespace Czm8
} // namespace Element
} // namespace GooseFEM

#endif // GOOSEFEM_ELEMENTCZM8_H