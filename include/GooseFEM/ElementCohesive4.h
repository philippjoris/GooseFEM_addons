/**
 * Quadrature for 4-noded cohesive zone element in 2d (GooseFEM::Mesh::ElementType::Czm4),
 * in a Cartesian coordinate system.
 *
 * @file ElementCzm4.h
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.
 *
 * TO DO:
 * - RENAME _IMPL FUNCTIONS IF USED EXPLICITLY
 * - RENAME CLASS TO COHESIVE ZONE ELEMENT NAME
 * 
 * 
 * 
 * 
 */

#ifndef GOOSEFEM_ELEMENTCZM4_H
#define GOOSEFEM_ELEMENTCZM4_H

#include "Element.h"
#include "config.h"
#include "detail.h"
#include <cmath> // For std::sqrt

namespace GooseFEM {
namespace Element {

/**
 * 4-noded cohesive zone element in 2d (GooseFEM::Mesh::ElementType::Czm4).
 *
 * Node ordering:
 *
 * 3 -------- 2
 * |          |
 * |  (Cohesive Zone)
 * |          |
 * 0 -------- 1
 *
 * Nodes (0,3) form one side of the interface, and (1,2) form the other.
 * It's common for 0 and 1 to be initially coincident, and 3 and 2 to be initially coincident.
 * The element models the interface between the top and bottom pairs of nodes.
 */
namespace Czm4 {

/**
 * Gauss quadrature for 4-noded cohesive zone element.
 * Integration is performed along the length of the interface (1D integration).
 */
namespace Gauss {

/**
 * Number of integration points.
 * For 4-noded linear element, 2 Gauss points are sufficient for exact integration
 * along the length.
 * @return unsigned int
 */
inline size_t nip()
{
    return 2;
}

/**
 * Integration point coordinates (local coordinates, along the interface).
 * These are 1D coordinates in the range [-1, 1].
 * The second dimension is kept for consistency with the Quad4 structure, but the
 * second component will be zero as it's a 1D integration.
 * @return Coordinates [#nip, `ndim`], with `ndim = 2`.
 */
inline array_type::tensor<double, 2> xi()
{
    size_t nip = 2;
    size_t ndim = 2; // Keep ndim=2 for consistency with Quad4 definition, but only xi(q,0) is used

    array_type::tensor<double, 2> xi = xt::empty<double>({nip, ndim});

    // 2-point Gauss quadrature for 1D
    xi(0, 0) = -1.0 / std::sqrt(3.0);
    xi(0, 1) = 0.0; // Not used for 1D integration
    xi(1, 0) = +1.0 / std::sqrt(3.0);
    xi(1, 1) = 0.0; // Not used for 1D integration

    return xi;
}

/**
 * Integration point weights.
 * @return Weights [#nip].
 */
inline array_type::tensor<double, 1> w()
{
    size_t nip = 2; // For 2-point Gauss quadrature

    array_type::tensor<double, 1> w = xt::empty<double>({nip});

    w(0) = 1.0;
    w(1) = 1.0;

    return w;
}

} // namespace Gauss

/**
 * Interpolation and quadrature for a 4-noded cohesive zone element.
 *
 * Fixed dimensions:
 * - `ndim = 2`: number of dimensions (for nodal displacements/coordinates).
 * - `nne = 4`: number of nodes per element.
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
     * Constructor: use default Gauss integration for cohesive elements (2 points along length).
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
        m_dNxi = xt::empty<double>({m_nip, s_nne, s_ndimxi}); // dN/dxi (only 1 local coordinate)

        // Shape functions for a 4-noded cohesive element (linear interpolation along length)
        // These are effectively 1D linear shape functions applied to pairs of nodes.
        for (size_t q = 0; q < m_nip; ++q) {
            double loc_xi = xi(q, 0); // Only use the first local coordinate for 1D integration

            // Nodes 0 and 3 are at xi = -1 (left side)
            // Nodes 1 and 2 are at xi = +1 (right side)
            m_N(q, 0) = 0.5 * (1.0 - loc_xi); // N_left for lower surface
            m_N(q, 1) = 0.5 * (1.0 + loc_xi); // N_right for lower surface
            m_N(q, 2) = 0.5 * (1.0 + loc_xi); // N_right for upper surface
            m_N(q, 3) = 0.5 * (1.0 - loc_xi); // N_left for upper surface
        }

        // Derivatives of shape functions with respect to local coordinate xi
        for (size_t q = 0; q < m_nip; ++q) {
            // dN/dxi_0 (only 0th component is relevant, corresponding to local_xi)
            m_dNxi(q, 0, 0) = -0.5;
            m_dNxi(q, 1, 0) = +0.5;
            m_dNxi(q, 2, 0) = +0.5;
            m_dNxi(q, 3, 0) = -0.5;
        }

        GOOSEFEM_ASSERT(m_x.shape(1) == s_nne);
        GOOSEFEM_ASSERT(m_x.shape(2) == s_ndim);
        GOOSEFEM_ASSERT(xt::has_shape(m_xi, {m_nip, s_ndim})); // ndim=2 for input consistency, only first component used
        GOOSEFEM_ASSERT(xt::has_shape(m_w, {m_nip}));
        GOOSEFEM_ASSERT(xt::has_shape(m_N, {m_nip, s_nne}));
        GOOSEFEM_ASSERT(xt::has_shape(m_dNxi, {m_nip, s_nne, s_ndimxi})); // Only 1 local derivative needed

        // Integration "volume" (length)
        m_vol = xt::empty<double>(this->shape_qscalar()); // This will be "effective length" * weight      

        m_rotation_matrix = xt::empty<double>({m_nelem, m_nip, s_ndim, s_ndim});

        this->compute_kinematics(); // Renamed and modified for CZEs
    }

    /**
    * @brief Computes the relative displacement and transforms it into normal and tangential components.
    *    *
    * @param elem_u Nodal displacements for elements (`elem_tensor<1>`), [#nelem, #nne, #ndim].
    * @param q_delta_u_norm_tan Output: Normal and tangential relative displacement
    * (`qtensor<1>`), [#nelem, #nip, #ndim].
    * q_delta_u_norm_tan(e, q, 0) = delta_u_normal
    * q_delta_u_norm_tan(e, q, 1) = delta_u_tangential
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
        for (size_t e = 0; e < m_nelem; ++e) {
            auto u = xt::adapt(&elem_u(e, 0, 0), xt::xshape<s_nne, s_ndim>());

            for (size_t q = 0; q < m_nip; ++q) {
                double N0 = m_N(q, 0);
                double N1 = m_N(q, 1);
                double N2 = m_N(q, 2);
                double N3 = m_N(q, 3);

                array_type::tensor<double, 1> u_lower = xt::empty<double>({s_ndim});
                u_lower(0) = N0 * u(0, 0) + N1 * u(1, 0);
                u_lower(1) = N0 * u(0, 1) + N1 * u(1, 1);

                array_type::tensor<double, 1> u_upper = xt::empty<double>({s_ndim});
                u_upper(0) = N3 * u(3, 0) + N2 * u(2, 0);
                u_upper(1) = N3 * u(3, 1) + N2 * u(2, 1);

                q_delta_u_global(e, q, 0) = u_upper(0) - u_lower(0);
                q_delta_u_global(e, q, 1) = u_upper(1) - u_lower(1);
            }
        }

        // 2. Transform the global relative displacement into normal and tangential components
        q_delta_u_norm_tan.fill(0.0); // Initialize the output array
        rot_mat.fill(0.0);

    #pragma omp parallel for
        for (size_t e = 0; e < m_nelem; ++e) {
            for (size_t q = 0; q < m_nip; ++q) {
                // Use the internally calculated q_delta_u_global
                auto delta_u = xt::adapt(&q_delta_u_global(e, q, 0), xt::xshape<s_ndim>());
                auto rot = xt::adapt(&m_rotation_matrix(e, q, 0, 0), xt::xshape<s_ndim, s_ndim>());
                auto rot_output = xt::adapt(&rot_mat(e, q, 0, 0), xt::xshape<s_ndim, s_ndim>());

                // Normal component: (delta_u . n)
                q_delta_u_norm_tan(e, q, 0) = rot(0, 0) * delta_u(0) + rot(0, 1) * delta_u(1);
                // Tangential component: (delta_u . t)
                q_delta_u_norm_tan(e, q, 1) = rot(1, 0) * delta_u(0) + rot(1, 1) * delta_u(1);
                // Fill rotmat
                rot_output = rot;
            }
        }
    }

    /**
     * @brief Computes the element force vector (residual).
     * This function is analogous to `int_gradN_dot_tensor2_dV_impl` but for CZEs.
     * It integrates tractions over the interface length to get nodal forces.
     *
     * @param q_tractions Tractions at integration points (`qtensor<1>`), [#nelem, #nip, #ndim].
     * q_tractions(e, q, :) are the traction components in global coordinates.
     * @param elem_f Output: Nodal force vector (`elemvec`), [#nelem, #nne, #ndim].
     */
    template <class T, class R>
    void int_N_dot_traction_dL(const T& q_tractions, R& elem_f) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(q_tractions, this->shape_qvector()));
        GOOSEFEM_ASSERT(xt::has_shape(elem_f, this->shape_elemvec()));

        elem_f.fill(0.0);

#pragma omp parallel for
        for (size_t e = 0; e < m_nelem; ++e) {
            auto f = xt::adapt(&elem_f(e, 0, 0), xt::xshape<s_nne, s_ndim>());

            for (size_t q = 0; q < m_nip; ++q) {
                auto tractions = xt::adapt(&q_tractions(e, q, 0), xt::xshape<s_ndim>());
                auto& dL = m_vol(e, q); 

                // Lower nodes (0, 1)
                f(0, 0) -= m_N(q, 0) * tractions(0) * dL; 
                f(0, 1) -= m_N(q, 0) * tractions(1) * dL; 

                f(1, 0) -= m_N(q, 1) * tractions(0) * dL; 
                f(1, 1) -= m_N(q, 1) * tractions(1) * dL; 

                // Upper nodes (3, 2)
                f(3, 0) += m_N(q, 3) * tractions(0) * dL; 
                f(3, 1) += m_N(q, 3) * tractions(1) * dL; 

                f(2, 0) += m_N(q, 2) * tractions(0) * dL; 
                f(2, 1) += m_N(q, 2) * tractions(1) * dL;
            }
        }
    }

    template <class T>
    array_type::tensor<double, 3> Int_N_dot_traction_dL(const T& q_tractions) const
    {
        auto elemmat = array_type::tensor<double, 3>::from_shape(this->shape_elemvec());
        this->int_N_dot_traction_dL(q_tractions, elemmat);
        return elemmat;
    }
    

    /**
     * @param qtensor [#nelem, #nip, #ndim, #ndim, #ndim, #ndim]
     * @return elemmat [#nelem, #nne * #ndim, #nne * #ndim]
     */
    template <class T>
    auto Int_BT_D_B_dL(const T& q_tangent_stiffness_global) const
        -> array_type::tensor<double, 3>
    {
        auto elemmat = array_type::tensor<double, 3>::from_shape({
        m_nelem, s_nne * s_ndim, s_nne * s_ndim
        });
        this->int_BT_D_B_dL(q_tangent_stiffness_global, elemmat);
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
    void int_BT_D_B_dL(const T& q_tangent_stiffness_global, R& elem_K) const
    {
        GOOSEFEM_ASSERT(xt::has_shape(q_tangent_stiffness_global, this->shape_qtensor<2>()));
        GOOSEFEM_ASSERT(xt::has_shape(elem_K, this->shape_elemmat()));

        elem_K.fill(0.0);

#pragma omp parallel for
        for (size_t e = 0; e < m_nelem; ++e) {
            auto K_elem = xt::adapt(&elem_K(e, 0, 0), xt::xshape<s_nne * s_ndim, s_nne * s_ndim>());

            for (size_t q = 0; q < m_nip; ++q) {
                auto D_T_global = xt::adapt(&q_tangent_stiffness_global(e, q, 0, 0), xt::xshape<s_ndim, s_ndim>());
                auto& dL = m_vol(e, q);
                // For each integration point 'q'
                // D_T_global (2x2) is the tangent stiffness of the traction-separation law (e.g., K_s, K_n)

                // Define the local 'B_coh' matrix for this integration point
                // This matrix maps global nodal displacements to local separation (delta_x, delta_y)
                // It's a 2x8 matrix: [delta_x; delta_y] = B_coh * [u0x, u0y, u1x, u1y, u2x, u2y, u3x, u3y]^T

                // Initialize B_coh (2 rows, 8 columns) with zeros
                array_type::tensor<double, 2> B_coh = xt::zeros<double>({s_ndim, s_nne * s_ndim});

                // Populate B_coh based on shape functions for lower (0,1) and upper (2,3) nodes
                // row 0: delta_x
                B_coh(0, 0) = -m_N(q, 0); // u0x
                B_coh(0, 2) = -m_N(q, 1); // u1x
                B_coh(0, 4) = +m_N(q, 2); // u2x
                B_coh(0, 6) = +m_N(q, 3); // u3x

                // row 1: delta_y
                B_coh(1, 1) = -m_N(q, 0); // u0y
                B_coh(1, 3) = -m_N(q, 1); // u1y
                B_coh(1, 5) = +m_N(q, 2); // u2y
                B_coh(1, 7) = +m_N(q, 3); // u3y

                // Calculate K_elem contribution for this integration point
                // K_elem += (B_coh.transpose() * D_T_global * B_coh) * dL;

                xt::xtensor_fixed<double, xt::xshape<s_ndim, s_nne * s_ndim>> Temp;
                Temp.fill(0.0);

                for (size_t i = 0; i < s_ndim; ++i) { 
                    for (size_t k = 0; k < s_nne * s_ndim; ++k) { 
                        for (size_t j = 0; j < s_ndim; ++j) {
                            Temp(i, k) += D_T_global(i, j) * B_coh(j, k);
                        }
                    }
                }

                for (size_t m = 0; m < s_nne * s_ndim; ++m) { 
                    for (size_t n = 0; n < s_nne * s_ndim; ++n) {
                        double sum_val = 0.0;
                        for (size_t p = 0; p < s_ndim; ++p) {
                            sum_val += B_coh(p, m) * Temp(p, n);
                        }
                        K_elem(m, n) += sum_val * dL; 
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
            array_type::tensor<double, 1> dX_dxi = xt::empty<double>({s_ndim}); // dX/dxi (X,Y)

#pragma omp for
            for (size_t e = 0; e < m_nelem; ++e) {
                auto x = xt::adapt(&m_x(e, 0, 0), xt::xshape<s_nne, s_ndim>());

                for (size_t q = 0; q < m_nip; ++q) {
                    // Get derivatives of shape functions w.r.t. local_xi
                    double dN0_dxi = m_dNxi(q, 0, 0);
                    double dN1_dxi = m_dNxi(q, 1, 0);
                    double dN2_dxi = m_dNxi(q, 2, 0);
                    double dN3_dxi = m_dNxi(q, 3, 0);

                    // Consider the lower line (nodes 0, 1) to define the tangent
                    double dx_dxi_lower = dN0_dxi * x(0, 0) + dN1_dxi * x(1, 0);
                    double dy_dxi_lower = dN0_dxi * x(0, 1) + dN1_dxi * x(1, 1);

                    // Consider the upper line (nodes 3, 2) to define the tangent
                    double dx_dxi_upper = dN3_dxi * x(3, 0) + dN2_dxi * x(2, 0);
                    double dy_dxi_upper = dN3_dxi * x(3, 1) + dN2_dxi * x(2, 1);

                    // Average tangent vector in global coordinates
                    dX_dxi(0) = 0.5 * (dx_dxi_lower + dx_dxi_upper);
                    dX_dxi(1) = 0.5 * (dy_dxi_lower + dy_dxi_upper);

                    double Jdet_1D = std::sqrt(dX_dxi(0) * dX_dxi(0) + dX_dxi(1) * dX_dxi(1)); // Length Jacobian
                    GOOSEFEM_ASSERT(Jdet_1D > 1e-12); // Check for degenerate elements

                    // Tangent vector (unit vector)
                    double t_x = dX_dxi(0) / Jdet_1D;
                    double t_y = dX_dxi(1) / Jdet_1D;

                    // Rotation matrix R = [[n_x, n_y],
                    //                     [t_x, t_y]]
                    m_rotation_matrix(e, q, 0, 0) = -t_y;
                    m_rotation_matrix(e, q, 0, 1) = t_x;
                    m_rotation_matrix(e, q, 1, 0) = t_x;
                    m_rotation_matrix(e, q, 1, 1) = t_y;

                    // The "volume" for CZEs is the integration point's effective length.
                    // This is the determinant of the 1D Jacobian times the integration weight.
                    m_vol(e, q) = m_w(q) * Jdet_1D;
                }
            }
        }
    }

    // Constants and members, similar to Quad4
    constexpr static size_t s_nne = 4;    ///< Number of nodes per element.
    constexpr static size_t s_ndim = 2;   ///< Number of dimensions for nodal vectors (x,y).
    constexpr static size_t s_ndimxi = 1; ///< Number of dimensions for shape function derivative.
    constexpr static size_t s_tdim = 2;   ///< For tensors like tangent stiffness in global coords (2x2).
    size_t m_tdim = 2;                    ///< Dynamic alias of s_tdim
    size_t m_nelem;                       ///< Number of elements.
    size_t m_nip;                         ///< Number of integration points per element.
    array_type::tensor<double, 3> m_x;    ///< nodal positions stored per element [#nelem, #nne, #ndim]
    array_type::tensor<double, 1> m_w;    ///< weight of each integration point [nip]
    array_type::tensor<double, 2> m_xi;   ///< local coordinate per integration point [#nip, #ndim] (only first component used)
    array_type::tensor<double, 2> m_N;    ///< shape functions [#nip, #nne]
    array_type::tensor<double, 3> m_dNxi; ///< local shape func grad [#nip, #nne, 1] (only one local coordinate)
    array_type::tensor<double, 2> m_vol;  ///< integration point "volume" (effective length * weight) [#nelem, #nip]
    array_type::tensor<double, 4> m_rotation_matrix; ///< rotation matrix of local element [#nelem, #nip, 2, 2]
};

} // namespace Czm4
} // namespace Element
} // namespace GooseFEM

#endif