/**
 * Constitutive relationship for 4-noded cohesive zone element in 2d,
 *                         and a 8-noded cohesive zone element in 3d,
 * in a Cartesian coordinate system.
 *
 * @file Cohesive3d.h
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.

 */

#ifndef GOOSEFEM_COHESIVE_H
#define GOOSEFEM_COHESIVE_H

#include <GMatTensor/Cartesian2d.h>
#include <GMatTensor/Cartesian3d.h>
#include <GooseFEM/config.h>
#include <GooseFEM/detail.h>
#include <cmath>
#include <numeric>
#include <type_traits> 

namespace GooseFEM {
namespace ConstitutiveModels {

template <size_t DIM>
using CohesiveBase = typename std::conditional<
    DIM == 2,
    GMatTensor::Cartesian2d,
    GMatTensor::Cartesian3d
>::type;

template <size_t N, size_t DIM>
class CohesiveBilinear : public CohesiveBase::Array<N> {
protected:
    using Base = CohesiveBase<DIM>;

    array_type::tensor<double, N> m_beta; ///< weighting for tangential separation per item.
    array_type::tensor<double, N> m_Kn; ///< initial normal separation stiffness per item.
    array_type::tensor<double, N> m_Kt; ///< initial tangential separation stiffness per item.
    array_type::tensor<double, N> m_delta0; ///< initiation displacement per item.
    array_type::tensor<double, N> m_deltafrac; ///< displacement displacement per item.
    array_type::tensor<double, N + 1> m_delta; ///< displacement jump per item.
    array_type::tensor<double, N + 1> m_T_local; ///< Traction per item in local coordinate system.
    array_type::tensor<double, N + 2> m_C_local; ///< Tangent per item in local coordinate system.
    array_type::tensor<double, N + 1> m_T; ///< Traction per item.
    array_type::tensor<double, N + 2> m_C; ///< Tangent per item.
    array_type::tensor<double, N + 2> m_P_matrix; ///< Tangent per item.
    array_type::tensor<bool, N> m_failed;
    array_type::tensor<double, N> m_delta_eff;

    array_type::tensor<double, N> m_Damage; ///< Accumulated damage variable per item.
    array_type::tensor<double, N> m_Damage_t; ///< Accumulated damage variable at previous increment per item.

    // member variables for viscous regularization
    array_type::tensor<double, N> m_Damage_v;
    array_type::tensor<double, N> m_Damage_v_t;
    array_type::tensor<double, N> m_eta;

    using Base::Array<N>::m_ndim;
    using Base::Array<N>::m_stride_tensor1;
    using Base::Array<N>::m_stride_tensor2;
    using Base::Array<N>::m_stride_tensor4;
    using Base::Array<N>::m_size;
    using Base::Array<N>::m_shape;
    using Base::Array<N>::m_shape_tensor1;
    using Base::Array<N>::m_shape_tensor2;
    using Base::Array<N>::m_shape_tensor4;

public:
    using Base::Array<N>::rank;

    CohesiveBilinear() = default;

    /**
    Construct system.
    \param Kn Normal separation stiffness per item.
    \param Kt Tangential separation stiffness per item.
    \param delta0 Initial displacement per item.
    \param beta Initial displacement per item.
    */
    template <class T>
    CohesiveBilinear(
        const T& Kn,
        const T& Kt,
        const T& delta0,
        const T& deltafrac,
        const T& beta,
        const T& eta
    ) {
        GOOSEFEM_ASSERT(Kn.dimension() == N);
        GOOSEFEM_ASSERT(Kt.dimension() == N);
        GOOSEFEM_ASSERT(delta0.dimension() == N);
        GOOSEFEM_ASSERT(deltafrac.dimension() == N);
        GOOSEFEM_ASSERT(beta.dimension() == N);
        GOOSEFEM_ASSERT(eta.dimension() == N);
        GOOSEFEM_ASSERT(xt::has_shape(Kn, Kt.shape()));
        std::copy(Kn.shape().cbegin(), Kn.shape().cend(), m_shape.begin());
        this->init(m_shape);

        m_Kn = Kn;
        m_Kt = Kt;
        m_delta0 = delta0;
        m_deltafrac = deltafrac;
        m_beta = beta;
        m_eta = eta;

        m_delta = xt::zeros<double>(m_shape_tensor1);
        m_delta_eff = xt::zeros<double>(m_shape);
        m_T_local = xt::empty<double>(m_shape_tensor1);
        m_C_local = xt::empty<double>(m_shape_tensor2);
        m_Damage = xt::zeros<double>(m_shape);
        m_Damage_t = m_Damage;
        m_Damage_v = xt::zeros<double>(m_shape);
        m_Damage_v_t = m_Damage_v;

        m_failed = xt::zeros<bool>(m_shape);
        m_P_matrix = xt::empty<double>(m_shape_tensor2);

        m_T = xt::empty<double>(m_shape_tensor1);
        m_C = xt::empty<double>(m_shape_tensor2);

        for (size_t i = 0; i < m_size; ++i) {
            auto C_local_i = xt::adapt(&m_C_local.flat(i * m_ndim * m_ndim), {m_ndim, m_ndim});
            auto C_i = xt::adapt(&m_C.flat(i * m_ndim * m_ndim), {m_ndim, m_ndim});

            C_local_i(0, 0) = m_Kn.flat(i);
            C_local_i(0, 1) = 0.0;
            C_local_i(1, 0) = 0.0;
            C_local_i(1, 1) = m_Kt.flat(i);

            if constexpr (DIM == 3) {
                C_local_i(0, 2) = 0.0;
                C_local_i(2, 0) = 0.0;
                C_local_i(1, 2) = 0.0;
                C_local_i(2, 1) = 0.0;
                C_local_i(2, 2) = m_Kt.flat(i);
            }

            array_type::tensor<double, m_ndim> P_matrix_i = Base::I2();

            xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> P_T_local;
            xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> Temp;

            Base::pointer::A2_dot_B2(C_local_i.data(), P_T_local.data(), Temp.data());
            Base::pointer::A2_dot_B2(Temp.data(), P_matrix_i.data(), C_i.data());
        }
    }

    /**
    Normal separation stiffness coefficient per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& Kn() const { return m_Kn; }

    /**
    Tangential separation stiffness coefficient per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& Kt() const { return m_Kt; }

    /**
    Current global tangential stiffness matrix per item
    \return [shape()].
    */
    const array_type::tensor<double, N + 2>& C() const { return m_C; }

    /**
    Current local tangential stiffness matrix per item
    \return [shape()].
    */
    const array_type::tensor<double, N + 2>& C_local() const { return m_C_local; }

    /**
    Current global traction vector per item
    \return [shape()].
    */
    const array_type::tensor<double, N + 1>& T() const { return m_T; }


    /**
    Current local traction vector per item
    \return [shape()].
    */
    const array_type::tensor<double, N + 1>& T_local() const { return m_T_local; }

    /**
    Current rotation matrix local-global per item
    \return [shape()].
    */
    const array_type::tensor<double, N + 2>& ori() const { return m_P_matrix; }

    /**
    Current rotation matrix local-global per item
    The user is responsible for calling refresh() after modifying entries.
    \return [shape()].
    */
    array_type::tensor<double, N + 2>& ori() { return m_P_matrix; }

    /**
    Current local displacement jump per item
    \return [shape()].
    */
    const array_type::tensor<double, N + 1>& delta() const { return m_delta; }

    /**
    Current local displacement jump per item
    The user is responsible for calling refresh() after modifying entries.
    \return [shape()].
    */
    array_type::tensor<double, N + 1>& delta() { return m_delta; }

    /**
    Initial displacement jump per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& delta0() const { return m_delta0; }

    /**
    Accumulated damage variable per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& Damage() const { return m_Damage; }

    /**
    Bool 'has failed' per item.
    \return [shape()].
    */
    const array_type::tensor<bool, N>& failed() const { return m_failed; }

    /**
    Effective relative displacement variable per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& delta_eff() const { return m_delta_eff; }

    const array_type::tensor<double, N>& eta() const { return m_eta; }

    // "this->refresh()" is not called automatically anymore! User has to call the refresh function him/herself!
    template <class T>
    void set_delta(const T& arg) {
        GOOSEFEM_ASSERT(xt::has_shape(arg, m_shape_tensor1));
        std::copy(arg.cbegin(), arg.cend(), m_delta.begin());
        // this->refresh(dt);
    }

    // "this->refresh()" is not called automatically anymore! User has to call the refresh function him/herself!
    template <class T>
    void set_ori(const T& arg) {
        GOOSEFEM_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_P_matrix.begin());
        // this->refresh(dt);
    }

    /*
    This refresh function returns the traction vector based on the displacement jump.
    */
    void refresh(double dt, bool compute_tangent = true, bool element_erosion = true) {
        namespace GT = Base::pointer;

        // In 2D, m_shape_tensor1 is [shape, 2] and m_shape_tensor2 is [shape, 2, 2]
        // In 3D, m_shape_tensor1 is [shape, 3] and m_shape_tensor2 is [shape, 3, 3]

#pragma omp parallel
        {
            auto T_local = xt::adapt(m_T_local.data(), {m_ndim});
            auto T = xt::adapt(m_T.data(), {m_ndim});
            auto C_local = xt::adapt(m_C_local.data(), {m_ndim, m_ndim});
            auto C = xt::adapt(m_C.data(), {m_ndim, m_ndim});
            auto P_matrix = xt::adapt(m_P_matrix.data(), {m_ndim, m_ndim});

#pragma omp for
            for (size_t i = 0; i < m_size; ++i) {

                double Kn = m_Kn.flat(i);
                double Kt = m_Kt.flat(i);
                double delta0 = m_delta0.flat(i);
                double deltafrac = m_deltafrac.flat(i);
                double beta = m_beta.flat(i);
                double eta = m_eta.flat(i);
                double G = deltafrac - delta0;
                GOOSEFEM_ASSERT(G > 1e-12);

                double damage_t = m_Damage_t.flat(i);
                double damage_v_t = m_Damage_v_t.flat(i);
                bool failed_prev = m_failed.flat(i);

                auto delta_i = xt::view(m_delta, i, xt::all());

                T_local.reset_buffer(&m_T_local.flat(i * m_stride_tensor1), m_stride_tensor1);
                T.reset_buffer(&m_T.flat(i * m_stride_tensor1), m_stride_tensor1);
                C_local.reset_buffer(&m_C_local.flat(i * m_stride_tensor2), m_stride_tensor2);
                C.reset_buffer(&m_C.flat(i * m_stride_tensor2), m_stride_tensor2);
                P_matrix.reset_buffer(&m_P_matrix.flat(i * m_stride_tensor2), m_stride_tensor2);

                if (element_erosion && failed_prev) {
                    T.fill(0.0);
                    C.fill(0.0);
                    m_Damage.flat(i) = 1.0;
                    m_Damage_v.flat(i) = 1.0;
                    continue;
                }

                double delta_n = delta_i(0);
                double delta_t_sq = 0.0;
                for (size_t k = 1; k < m_ndim; ++k) {
                    delta_t_sq += delta_i(k) * delta_i(k);
                }
                double delta_t = std::sqrt(delta_t_sq);
                
                // Compression logic
                if (delta_n < 0.0) {
                    T_local(0) = Kn * delta_n;
                    for (size_t k = 1; k < m_ndim; ++k) {
                        T_local(k) = Kt * delta_i(k);
                    }
                    C_local.fill(0.0);
                    C_local(0, 0) = Kn;
                    for (size_t k = 1; k < m_ndim; ++k) {
                        C_local(k, k) = Kt;
                    }
                    m_delta_eff.flat(i) = 0.0;
                    m_Damage.flat(i) = 0.0;
                    m_Damage_v.flat(i) = 0.0;
                } else { // Tension or shear

                    m_delta_eff.flat(i) = std::sqrt(delta_n * delta_n + beta * delta_t_sq);
                    double delta_eff = m_delta_eff.flat(i);
                    double D_instant_trial;
                    if (delta_eff <= delta0) {
                        D_instant_trial = 0.0;
                    } else if (delta_eff >= deltafrac) {
                        D_instant_trial = 1.0;
                    } else {
                        if (std::abs(G) < 1e-12) {
                            D_instant_trial = 1.0;
                        } else {
                            D_instant_trial = (deltafrac * (delta_eff - delta0)) / (delta_eff * G);
                        }
                    }

                    D_instant_trial = std::max(D_instant_trial, damage_t);
                    D_instant_trial = std::min(D_instant_trial, 1.0);

                    double current_D_v;
                    if (eta < 1e-12) {
                        current_D_v = D_instant_trial;
                    } else {
                        current_D_v = damage_v_t + (dt / eta) * (D_instant_trial - damage_v_t);
                    }
                    current_D_v = std::max(0.0, std::min(1.0, current_D_v));

                    m_Damage.flat(i) = current_D_v;
                    m_Damage_v.flat(i) = current_D_v;

                    if (element_erosion && m_Damage.flat(i) >= 1.0) {
                        m_failed.flat(i) = true;
                    }

                    T_local(0) = (1 - current_D_v) * Kn * delta_n;
                    for (size_t k = 1; k < m_ndim; ++k) {
                        T_local(k) = (1 - current_D_v) * Kt * delta_i(k);
                    }
                    
                    if (compute_tangent) {
                        double dD_d_delta_eff = 0.0;
                        if (delta_eff > delta0 && delta_eff < deltafrac && std::abs(G) > 1e-12) {
                            dD_d_delta_eff = (deltafrac * delta0) / (G * delta_eff * delta_eff);
                        }

                        double d_current_D_v_d_D_instant_trial = (eta > 1e-12) ? (dt / eta) : 1.0;

                        double K_n_eff_secant = (1.0 - current_D_v) * Kn;
                        double K_t_eff_secant = (1.0 - current_D_v) * Kt;

                        C_local.fill(0.0);
                        if (current_D_v < 1.0 - 1e-6) {
                            
                            double d_dv_d_delta_n = (delta_eff > 1e-12) ? d_current_D_v_d_D_instant_trial * dD_d_delta_eff * (delta_n / delta_eff) : 0.0;

                            C_local(0, 0) = K_n_eff_secant - Kn * delta_n * d_dv_d_delta_n;
                            for (size_t k = 1; k < m_ndim; ++k) {
                                double d_dv_d_delta_t = (delta_eff > 1e-12) ? d_current_D_v_d_D_instant_trial * dD_d_delta_eff * (beta * delta_i(k) / delta_eff) : 0.0;
                                C_local(0, k) = -Kn * delta_n * d_dv_d_delta_t;
                                C_local(k, 0) = -Kt * delta_i(k) * d_dv_d_delta_n;
                                C_local(k, k) = K_t_eff_secant - Kt * delta_i(k) * d_dv_d_delta_t;
                            }
                        }
                    }
                }

                // Transform local tractions to global tractions
                GT::A2_dot_B1(P_matrix.data(), T_local.data(), T.data());

                if (compute_tangent) {
                    xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> P_T;
                    for (size_t i = 0; k < m_ndim; i++) {
                        for(size_t j = 0; j < m_ndim; j++){
                            P_T(i,j) = P_matrix(i,j);
                        }
                    }
                    xt::tensor_fixed<double, xt::xshape<m_ndim, m_ndim>> Temp;
                    GT::A2_dot_B2(C_local.data(), P_T.data(), Temp.data());
                    GT::A2_dot_B2(Temp.data(), P_matrix.data(), C.data());
                }
            }
        }
    }

    void increment() {
        std::copy(m_Damage.cbegin(), m_Damage.cend(), m_Damage_t.begin());
        std::copy(m_Damage_v.cbegin(), m_Damage_v.cend(), m_Damage_v_t.begin());
    }
};

} // namespace ConstitutiveModels
} // namespace GooseFEM

#endif