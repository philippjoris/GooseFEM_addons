/**
 * Constitutive relationship for 4-noded cohesive zone element in 2d (GooseFEM::Cohesive3d),
 * in a Cartesian coordinate system.
 *
 * @file Cohesive3d.h
 * @copyright Copyright 2025. Philipp van der Loos. All rights reserved.

The model is probably only valid for 2d simulations at the moment. If i want to make it versatile for 3d i need 
to make the dimensions of the variables more general I think.

 */

#ifndef GOOSEFEM_COHESIVE2D_H
#define GOOSEFEM_COHESIVE2D_H

#include <GMatTensor/Cartesian2d.h>

#include <GooseFEM/config.h>
#include <GooseFEM/detail.h>
#include <cmath> // For std::sqrt

namespace GooseFEM {
namespace ConstitutiveModels{
namespace Cartesian2d{

template <size_t N>
class CohesiveBilinear : public GMatTensor::Cartesian2d::Array<N> {
protected:
    array_type::tensor<double, N> m_beta; ///< weighting for tangential separation per item.
    array_type::tensor<double, N> m_Kn; ///< initial normal separation stiffness per item.
    array_type::tensor<double, N> m_Kt; ///< initial tangential separation stiffness per item.    
    array_type::tensor<double, N> m_delta0; ///< initiation displacement per item. This is a vector?
    array_type::tensor<double, N> m_deltafrac; ///< displacement displacement per item. This is a vector?
    array_type::tensor<double, N + 1> m_delta; ///< displacement jump per item. This is a vector?
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
    array_type::tensor<double, N> m_Damage_v;   ///< Viscously regularized damage variable per item (current).
    array_type::tensor<double, N> m_Damage_v_t; ///< Viscously regularized damage variable at previous increment per item.
    array_type::tensor<double, N> m_eta;        ///< Viscosity parameter (relaxation time) per item.       

    using GMatTensor::Cartesian2d::Array<N>::m_ndim;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor1;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian2d::Array<N>::m_size;
    using GMatTensor::Cartesian2d::Array<N>::m_shape;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor1;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian2d::Array<N>::rank;

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
    )
    {
        GOOSEFEM_ASSERT(Kn.dimension() == N);
        GOOSEFEM_ASSERT(Kt.dimension() == N);        
        GOOSEFEM_ASSERT(delta0.dimension() == N);                
        GOOSEFEM_ASSERT(beta.dimension() == N);                        
        GOOSEFEM_ASSERT(eta.dimension() == N);     
        GOOSEFEM_ASSERT(xt::has_shape(Kn, Kt.shape()));
        std::copy(Kn.shape().cbegin(), Kn.shape().cend(), m_shape.begin());
        // std::copy(Kt.shape().cbegin(), Kt.shape().cend(), m_shape.begin());        
        this->init(m_shape);

        m_Kn = Kn;
        m_Kt= Kt;
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

        // rotation matrix between local and global coordinates
        m_P_matrix = xt::empty<double>(m_shape_tensor2);;

        m_T = xt::empty<double>(m_shape_tensor1); // Global tractions (results of refresh)
        m_C = xt::empty<double>(m_shape_tensor2); // Global tangent stiffness        

        // Initialize initial stiffness matrix explicitly
        for (size_t i = 0; i < m_size; ++i) {
            auto C_local_i = xt::adapt(&m_C_local.flat(i * m_ndim * m_ndim), {m_ndim, m_ndim});
            auto C_i = xt::adapt(&m_C.flat(i * m_ndim * m_ndim), {m_ndim, m_ndim});
            C_local_i(0, 0) = m_Kn.flat(i);
            C_local_i(0, 1) = 0.0;
            C_local_i(1, 0) = 0.0;
            C_local_i(1, 1) = m_Kt.flat(i);

            // Also initialize global stiffness C using P_matrix (if P_matrix is initialized here,
            // which it should be if you need C at construction)
            // For now, let's assume P_matrix is also initialized to identity or specific initial orientation.
            // If P_matrix is typically set *after* construction, then m_C can be set to zero or uninitialized
            // and computed in the first refresh call in the simulation.
            // If P_matrix is not set yet, you cannot compute m_C here.
            // For simplicity, let's just initialize m_C_local to initial stiffness.
            // The global C can be set by the first refresh in the simulation with the actual dt and delta.

            // You would also initialize P_matrix here if it's constant or has a default.
            // If it's variable and set via `set_ori`, then m_C would need to be computed later.
            // Assuming P_matrix is set to identity initially:
            namespace GT = GMatTensor::Cartesian2d::pointer;

            array_type::tensor<double, m_ndim> P_matrix_i = GMatTensor::Cartesian2d::I2();
            xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> P_T_local;
            P_T_local(0,0) = P_matrix_i(0,0); P_T_local(0,1) = P_matrix_i(1,0);
            P_T_local(1,0) = P_matrix_i(0,1); P_T_local(1,1) = P_matrix_i(1,1);
            xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> Temp;
            GT::A2_dot_B2(C_local_i.data(), P_T_local.data(), Temp.data());
            GT::A2_dot_B2(Temp.data(), P_matrix_i.data(), C_i.data());
        }
    }

    /**
    Normal separation stiffness coefficient per item.
    \return [shape()].
    */
    const array_type::tensor<double, N>& Kn() const
    {
        return m_Kn;
    }

    /**
    Tangential separation stiffness coefficient per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& Kt() const
    {
        return m_Kt;
    }

    /**
    Current global tangential stiffness matrix per item
    \return [shape()].
    */
    const array_type::tensor<double, N+2>& C() const
    {
        return m_C;
    }    


    /**
    Current local tangential stiffness matrix per item
    \return [shape()].
    */
    const array_type::tensor<double, N+2>& C_local() const
    {
        return m_C_local;
    } 

    /**
    Current global traction vector per item
    \return [shape()].
    */
    const array_type::tensor<double, N+1>& T() const
    {
        return m_T;
    }    


    /**
    Current local traction vector per item
    \return [shape()].
    */
    const array_type::tensor<double, N+1>& T_local() const
    {
        return m_T_local;
    }    

    /**
    Current rotation matrix local-global per item
    \return [shape()].
    */
    const array_type::tensor<double, N+2>& ori() const
    {
        return m_P_matrix;
    }    

    /**
    Current rotation matrix local-global per item
    The user is responsible for calling refresh() after modifying entries.
    \return [shape()].
    */
    array_type::tensor<double, N+2>& ori()
    {
        return m_P_matrix;
    }    

    /**
    Current local displacement jump per item
    \return [shape()].
    */
    const array_type::tensor<double, N+1>& delta() const
    {
        return m_delta;
    }    

    /**
    Current local displacement jump per item
    The user is responsible for calling refresh() after modifying entries.
    \return [shape()].
    */
    array_type::tensor<double, N+1>& delta()
    {
        return m_delta;
    }       

    /**
    Initial displacement jump per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& delta0() const
    {
        return m_delta0;
    }    

    /**
    Accumulated damage variable per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& Damage() const
    {
        return m_Damage;
    }    

    /**
    Bool 'has failed' per item.
    \return [shape()].
    */
    const array_type::tensor<bool, N>& failed() const
    {
        return m_failed;
    }  
    
    /**
    Effective relative displacement variable per item
    \return [shape()].
    */
    const array_type::tensor<double, N>& delta_eff() const
    {
        return m_delta_eff;
    }      

    const array_type::tensor<double, N>& eta() const
    {
        return m_eta;
    }

    // this->refresh() is not called automatically anymore! User has to call the refresh function him/herself!
    template <class T>
    void set_delta(const T& arg)
    {
        GOOSEFEM_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_delta.begin());
        // this->refresh(dt);
    }

    // this->refresh() is not called automatically anymore! User has to call the refresh function him/herself!
    template <class T>
    void set_ori(const T& arg)
    {
        GOOSEFEM_ASSERT(xt::has_shape(arg, m_shape_tensor4));
        std::copy(arg.cbegin(), arg.cend(), m_P_matrix.begin());
        // this->refresh(dt);
    }

    /*
    This refresh function returns the traction vector based on the displacement jump.

    Note though that you can call this function as often as you like, you will only loose time.
    */
    void refresh(double dt, bool compute_tangent = true, bool element_erosion = true)
    {        
        namespace GT = GMatTensor::Cartesian2d::pointer;

#pragma omp parallel
        {    
            // In this case, T_local, T, C_local, C, P_matrix are declared *inside* the parallel region,
            // making them thread-private. Each thread will have its own copy of these `xt::adapt` objects.

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
                double G = m_deltafrac.flat(i) - m_delta0.flat(i); GOOSEFEM_ASSERT(G > 1e-12);
                                
                double damage_t = m_Damage_t.flat(i);
                double damage_v_t = m_Damage_v_t.flat(i);
                bool failed_prev = m_failed.flat(i);  
                
                // creates a slice of m_delta and treads it as a xtensor itself
                auto delta = xt::view(m_delta, i, xt::all());
                
                // reset_buffer is called on an existing xt::adapt object. It changes the underlying memory pointer
                // to the specific element i
                T_local.reset_buffer(&m_T_local.flat(i * m_ndim), m_ndim);
                T.reset_buffer(&m_T.flat(i * m_ndim), m_ndim);
                C_local.reset_buffer(&m_C_local.flat(i * m_ndim * m_ndim), m_ndim * m_ndim);  
                C.reset_buffer(&m_C.flat(i * m_ndim * m_ndim), m_ndim * m_ndim);
                P_matrix.reset_buffer(&m_P_matrix.flat(i * m_ndim * m_ndim), m_ndim * m_ndim);

                if (element_erosion && failed_prev) {
                    T.fill(0.0);
                    C.fill(0.0);
                    m_Damage.flat(i) = 1.0; 
                    m_Damage_v.flat(i) = 1.0; 
                    continue;
                }                
                
                // Step 1: decompose delta into delta_n and delta_t
                double delta_n = std::max(0.0, delta[0]);
                double delta_t = delta[1];

                // Step 2: calculate effective displacement jump delta_eff = sqrt{ pow(max(delta_n), 2) + beta * pow(max(delta_t), 2) }
                m_delta_eff.flat(i) = std::sqrt(delta_n * delta_n + beta * delta_t * delta_t);

                double delta_eff = m_delta_eff.flat(i);
                // Step 3: calculate Damage value
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

                // Viscous regularization
                double current_D_v;
                if (eta < 1e-12) {
                    current_D_v = D_instant_trial;
                } else {
                    // Backwards Euler update for viscous damage 
                    current_D_v = damage_v_t + (dt / eta) * (D_instant_trial - damage_v_t);
                }

                current_D_v = std::max(0.0, current_D_v);
                current_D_v = std::min(1.0, current_D_v);

                m_Damage.flat(i) = current_D_v;
                m_Damage_v.flat(i) = current_D_v; 

                // Shall I limit the damage to one? What if damage is averaged for a cell?
                if (element_erosion && m_Damage.flat(i) >= 1.0) {
                    m_failed.flat(i) = true; 
                    }                    
                
                // Step 4: Calculate traction vector
                T_local(0) = (1 - current_D_v) * Kn * delta_n;
                T_local(1) = (1 - current_D_v) * Kt * delta_t;    
                
                // Transform local traction vector to global coordinates
                GT::A2_dot_B1(P_matrix.data(), T_local.data(), T.data());
                
                // --- Tangent Calculation ---
                if (!compute_tangent) {
                    return;
                }
                
                // Step 4: Calculate local tangent stiffness C_local
                double dD_d_delta_eff = 0.0;
                if (delta_eff > delta0 && delta_eff < deltafrac && std::abs(G) > 1e-12) {
                    // Derivative of D_trial = (deltafrac * (delta_eff - delta0)) / (delta_eff * G)
                    // w.r.t delta_eff
                    // d/dx (a * (x-b) / (x * c)) = a/c * d/dx ( (x-b)/x ) = a/c * d/dx (1 - b/x)
                    // = a/c * (b/x^2)
                    // So, dD_d_delta_eff = (deltafrac / G) * (delta0 / (delta_eff * delta_eff));
                    dD_d_delta_eff = (deltafrac * delta0) / (G * delta_eff * delta_eff);
                }

                // Derivatives of delta_eff w.r.t delta_n and delta_t
                double d_delta_eff_d_delta_n = (delta_eff > 1e-12) ? delta_n / delta_eff : 0.0;
                double d_delta_eff_d_delta_t = (delta_eff > 1e-12) ? beta * delta_t / delta_eff : 0.0;

                // Derivative of viscous damage w.r.t. instantaneous damage.
                // d(current_D_v)/d(D_trial_instantaneous) = dt / eta if eta > 0
                double d_current_D_v_d_D_instant_trial = (eta > 1e-12) ? (dt / eta) : 1.0;

                // Effective stiffness without damage, but scaled by current_D_v (often called secant modulus for damage)
                double K_n_eff_secant = (1.0 - current_D_v) * Kn;
                double K_t_eff_secant = (1.0 - current_D_v) * Kt;

                if (delta_eff <= delta0){
                    C_local(0, 0) = Kn;
                    C_local(0, 1) = 0.0;
                    C_local(1, 0) = 0.0;
                    for (size_t k = 0; k < m_ndim - 1; ++k) {
                        C_local(k + 1, k + 1) = Kt; 
                    }
                }
                else if (current_D_v >= 1.0 - 1e-6){ // Use current_D_v for failure check
                    C_local.fill(0.0);
                }
                else {
                    // dT_n/d_delta_n = (1-d_v) * Kn - Kn * delta_n * d(d_v)/d(delta_n)
                    // d(d_v)/d(delta_n) = d(d_v)/d(D_trial_instantaneous) * d(D_trial_instantaneous)/d(delta_eff) * d(delta_eff)/d(delta_n)

                    double d_dv_d_delta_n = d_current_D_v_d_D_instant_trial * dD_d_delta_eff * d_delta_eff_d_delta_n;
                    double d_dv_d_delta_t = d_current_D_v_d_D_instant_trial * dD_d_delta_eff * d_delta_eff_d_delta_t;

                    // d_Tn/d_delta_n = (1-d_v) * Kn + (-Kn * delta_n) * d(d_v)/d(delta_n)
                    C_local(0, 0) = K_n_eff_secant - Kn * delta_n * d_dv_d_delta_n; // Added derivative term for d_v
                    // d_Tn/d_delta_t = (-Kn * delta_n) * d(d_v)/d(delta_t)
                    C_local(0, 1) = -Kn * delta_n * d_dv_d_delta_t; // Added derivative term for d_v

                    // d_Tt/d_delta_n = (-Kt * delta_t) * d(d_v)/d(delta_n)
                    C_local(1, 0) = -Kt * delta_t * d_dv_d_delta_n; // Added derivative term for d_v
                    // d_Tt/d_delta_t = (1-d_v) * Kt + (-Kt * delta_t) * d(d_v)/d(delta_t)
                    C_local(1, 1) = K_t_eff_secant - Kt * delta_t * d_dv_d_delta_t; // Added derivative term for d_v
                }

                // ------ REMOVE --------
                // if (delta_eff <= delta0){
                //     C_local(0, 0) = Kn; 
                //     for (size_t k = 0; k < m_ndim - 1; ++k) {
                //         C_local(k + 1, k + 1) = Kt; 
                //     }
                //     C_local(0,1) = 0.0;
                //     C_local(1,0) = 0.0;
                // }
                // else if (delta_eff >= deltafrac){
                //     C_local.fill(0.0);
                // }
                // else {
                //     C_local(0,0) = (1-m_Damage.flat(i))*Kn - (Kn * delta_n * delta_n)/(G * delta_eff);
                //     C_local(0,1) = -(Kn * beta * delta_n * delta_t)/(G * delta_eff);
                //     C_local(1,0) = -(Kt * delta_n * delta_t)/(G * delta_eff);
                //     C_local(1,1) = (1-m_Damage.flat(i)) * Kt - (Kt * beta * delta_t * delta_t)/(G * delta_eff);
                // }   
                // ------ REMOVE --------                
                    
                xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> P_T;
                P_T(0,0) = P_matrix(0,0); P_T(0,1) = P_matrix(1,0); 
                P_T(1,0) = P_matrix(0,1); P_T(1,1) = P_matrix(1,1); 

                // Transform local stiffness matrix to global coordinates
                xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> Temp;
                GT::A2_dot_B2(C_local.data(), P_T.data(), Temp.data());
                GT::A2_dot_B2(Temp.data(), P_matrix.data(), C.data());
            }
        }
    }

    void increment()
    {
        std::copy(m_Damage.cbegin(), m_Damage.cend(), m_Damage_t.begin());
        std::copy(m_Damage_v.cbegin(), m_Damage_v.cend(), m_Damage_v_t.begin());
    }
};

// --- CohesiveExponentialGc class ---

template <size_t N>
class CohesiveExponential : public GMatTensor::Cartesian2d::Array<N> {
protected:
    array_type::tensor<double, N> m_beta; ///< weighting for tangential separation per item.
    array_type::tensor<double, N> m_Kn; ///< initial normal separation stiffness per item.
    array_type::tensor<double, N> m_Kt; ///< initial tangential separation stiffness per item.
    array_type::tensor<double, N> m_delta0; ///< Effective displacement at peak traction (initiation).
    array_type::tensor<double, N> m_Gc; ///< Critical energy release rate per item.

    // Derived parameter from Gc and delta0
    array_type::tensor<double, N> m_delta_exp_char; ///< Characteristic decay length for exponential.

    array_type::tensor<double, N + 1> m_delta; ///< displacement jump per item.
    array_type::tensor<double, N + 1> m_T_local; ///< Traction per item in local coordinate system.
    array_type::tensor<double, N + 2> m_C_local; ///< Tangent per item in local coordinate system.
    array_type::tensor<double, N + 1> m_T; ///< Traction per item.
    array_type::tensor<double, N + 2> m_C; ///< Tangent per item.
    array_type::tensor<double, N + 2> m_P_matrix; ///< Transformation matrix (local to global).
    array_type::tensor<bool, N> m_failed;
    array_type::tensor<double, N> m_delta_eff;

    array_type::tensor<double, N> m_Damage; ///< Accumulated damage variable per item.
    array_type::tensor<double, N> m_Damage_t; ///< Accumulated damage variable at previous increment per item.

    using GMatTensor::Cartesian2d::Array<N>::m_ndim;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor1;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_stride_tensor4;
    using GMatTensor::Cartesian2d::Array<N>::m_size;
    using GMatTensor::Cartesian2d::Array<N>::m_shape;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor1;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor2;
    using GMatTensor::Cartesian2d::Array<N>::m_shape_tensor4;

public:
    using GMatTensor::Cartesian2d::Array<N>::rank;

    CohesiveExponential() = default;

    /**
    Construct system for exponential cohesive law based on Gc.
    \param Kn Normal separation stiffness per item.
    \param Kt Tangential separation stiffness per item.
    \param delta0 Effective separation at peak traction.
    \param Gc Critical energy release rate per item.
    \param beta Weighting for tangential separation per item.
    */
    template <class T>
    CohesiveExponential(
        const T& Kn,
        const T& Kt,
        const T& delta0,
        const T& Gc,
        const T& beta
    )
    {
        GOOSEFEM_ASSERT(Kn.dimension() == N);
        GOOSEFEM_ASSERT(Kt.dimension() == N);
        GOOSEFEM_ASSERT(delta0.dimension() == N);
        GOOSEFEM_ASSERT(Gc.dimension() == N);
        GOOSEFEM_ASSERT(beta.dimension() == N);
        GOOSEFEM_ASSERT(xt::has_shape(Kn, Kt.shape()));
        std::copy(Kn.shape().cbegin(), Kn.shape().cend(), m_shape.begin());
        this->init(m_shape);

        m_Kn = Kn;
        m_Kt = Kt;
        m_delta0 = delta0;
        m_Gc = Gc;
        m_beta = beta;

        m_delta_exp_char = xt::empty<double>(m_shape);

        // Pre-calculate characteristic exponential length for each element
        // Peak effective traction: sigma_peak_eff = sqrt( (Kn * delta0)^2 + beta * (Kt * delta0)^2 )
        // Gc = 0.5 * sigma_peak_eff * delta0 + sigma_peak_eff * delta_exp_char
        // Rearranging for delta_exp_char:
        // delta_exp_char = Gc / sigma_peak_eff - 0.5 * delta0

        for (size_t i = 0; i < m_size; ++i) {
            double dp = m_delta0.flat(i);
            double kn_val = m_Kn.flat(i);
            double kt_val = m_Kt.flat(i);
            double beta_val = m_beta.flat(i);
            double gc_val = m_Gc.flat(i);

            // Effective peak stress, assuming max normal/tangential stress happens at delta0
            // This is an approximation for mixed-mode loading.
            // A common simplification is to take max normal traction for peak_eff.
            // For general effective traction, we'd need to consider the peak *effective* traction.
            // Let's use the normal peak traction for simplicity in the effective stress calculation.
            // A more rigorous approach for mixed-mode might involve defining Gc for pure modes and an interaction law.
            // For now, let's consider the maximum of normal and effective tangential peak tractions.
            // A more standard approach takes the max normal traction (Kn*delta0) as the peak stress for Gc calc.
            
            // Let's use the normal peak traction for simplicity in the effective stress calculation.
            // sigma_peak_n = kn_val * dp;
            // sigma_peak_t = kt_val * dp;
            // double sigma_peak_eff = std::sqrt(sigma_peak_n * sigma_peak_n + beta_val * sigma_peak_t * sigma_peak_t);
            // This is problematic. Let's define the peak stress at delta0 to be the elastic stress.
            // sigma_peak_eff = std::sqrt(std::pow(kn_val * dp, 2.0) + beta_val * std::pow(kt_val * dp, 2.0));
            // Let's simplify and use the critical normal stress:
            
            // Re-evaluating the effective peak stress for calculation of delta_exp_char:
            // The formula Gc = 0.5 * T_0 * delta_0 + T_0 * delta_exp (where T_0 is peak effective stress)
            // implies that T_0 is the peak effective stress at delta0.
            // If delta_eff = delta0, then T_n = Kn * delta0, T_t = Kt * delta0.
            // So, effective peak stress (for energy calculation) is:
            double T_peak_n = kn_val * dp;
            double T_peak_t = kt_val * dp;
            double sigma_peak_eff = std::sqrt(T_peak_n * T_peak_n + beta_val * T_peak_t * T_peak_t);

            GOOSEFEM_ASSERT(sigma_peak_eff > 1e-12); // Ensure non-zero peak stress
            
            m_delta_exp_char.flat(i) = gc_val / sigma_peak_eff - 0.5 * dp;
            GOOSEFEM_ASSERT(m_delta_exp_char.flat(i) > 0); // Ensure positive exponential decay length
        }


        m_delta = xt::zeros<double>(m_shape_tensor1);
        m_delta_eff = xt::zeros<double>(m_shape);
        m_T_local = xt::empty<double>(m_shape_tensor1);
        m_C_local = xt::empty<double>(m_shape_tensor2);
        m_Damage = xt::zeros<double>(m_shape);
        m_Damage_t = m_Damage;
        m_failed = xt::zeros<bool>(m_shape);

        m_P_matrix = xt::empty<double>(m_shape_tensor2);

        m_T = xt::empty<double>(m_shape_tensor1);
        m_C = xt::empty<double>(m_shape_tensor2);

        this->refresh();
    }

    // Accessors (similar to Cohesive, omitted for brevity but they should be there)
    const array_type::tensor<double, N>& Kn() const { return m_Kn; }
    const array_type::tensor<double, N>& Kt() const { return m_Kt; }
    const array_type::tensor<double, N>& delta0() const { return m_delta0; }
    const array_type::tensor<double, N>& Gc() const { return m_Gc; }
    const array_type::tensor<double, N>& beta() const { return m_beta; }
    const array_type::tensor<double, N>& delta_exp_char() const { return m_delta_exp_char; }

    const array_type::tensor<double, N+2>& C() const { return m_C; }
    const array_type::tensor<double, N+2>& C_local() const { return m_C_local; }
    const array_type::tensor<double, N+1>& T() const { return m_T; }
    const array_type::tensor<double, N+1>& T_local() const { return m_T_local; }
    const array_type::tensor<double, N+2>& ori() const { return m_P_matrix; }
    array_type::tensor<double, N+2>& ori() { return m_P_matrix; }
    const array_type::tensor<double, N+1>& delta() const { return m_delta; }
    array_type::tensor<double, N+1>& delta() { return m_delta; }
    const array_type::tensor<double, N>& Damage() const { return m_Damage; }
    const array_type::tensor<bool, N>& failed() const { return m_failed; }
    const array_type::tensor<double, N>& delta_eff() const { return m_delta_eff; }

    template <class T>
    void set_delta(const T& arg)
    {
        GOOSEFEM_ASSERT(xt::has_shape(arg, m_shape_tensor1));
        std::copy(arg.cbegin(), arg.cend(), m_delta.begin());
        this->refresh();
    }

    template <class T>
    void set_ori(const T& arg)
    {
        GOOSEFEM_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_P_matrix.begin());
        this->refresh();
    }

    /*
    Refresh function for Exponential Cohesive Zone Model based on Gc.
    */
    void refresh(bool compute_tangent = true, bool element_erosion = true)
    {
        namespace GT = GMatTensor::Cartesian2d::pointer;

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
                double beta = m_beta.flat(i);
                double delta_exp_char = m_delta_exp_char.flat(i);

                double damage_t = m_Damage_t.flat(i);
                bool failed_prev = m_failed.flat(i);

                auto delta = xt::view(m_delta, i, xt::all());

                T_local.reset_buffer(&m_T_local.flat(i * m_ndim), m_ndim);
                T.reset_buffer(&m_T.flat(i * m_ndim), m_ndim);
                C_local.reset_buffer(&m_C_local.flat(i * m_ndim * m_ndim), m_ndim * m_ndim);  
                C.reset_buffer(&m_C.flat(i * m_ndim * m_ndim), m_ndim * m_ndim);
                P_matrix.reset_buffer(&m_P_matrix.flat(i * m_ndim * m_ndim), m_ndim * m_ndim);

                if (element_erosion && failed_prev) {
                    T.fill(0.0);
                    C.fill(0.0);
                    m_Damage.flat(i) = 1.0; 
                    continue;
                }      

                // Step 1: decompose delta into delta_n and delta_t
                double delta_n = std::max(0.0, delta[0]);
                double delta_t = delta[1];

                // Step 2: calculate effective displacement jump delta_eff = sqrt{ pow(max(delta_n), 2) + beta * pow(max(delta_t), 2) }
                m_delta_eff.flat(i) = std::sqrt(delta_n * delta_n + beta * delta_t * delta_t);

                double delta_eff = m_delta_eff.flat(i);

                // --- Exponential Damage Calculation based on Gc ---
                double D_trial;
                if (delta_eff <= delta0) {
                    D_trial = 0.0;
                } else {
                    if (delta_exp_char < 1e-12) { // Extremely sharp decay, effectively instant failure after delta0
                        D_trial = 1.0;
                    } else {
                        D_trial = 1.0 - (delta0 / delta_eff) * std::exp(-(delta_eff - delta0) / delta_exp_char);
                    }
                }

                double current_D = std::max(D_trial, damage_t);
                current_D = std::min(current_D, 1.0);
                m_Damage.flat(i) = current_D;

                // Mark as failed if damage is very close to 1.0
                if (element_erosion && m_Damage.flat(i) >= 1.0 - 1e-9) {
                    m_failed.flat(i) = true;
                }

                // Calculate traction vector
                T_local(0) = (1.0 - m_Damage.flat(i)) * Kn * delta_n;
                T_local(1) = (1.0 - m_Damage.flat(i)) * Kt * delta_t;
                
                // Transform local traction vector to global coordinates
                GT::A2_dot_B1(P_matrix.data(), T_local.data(), T.data());

                if (!compute_tangent) {
                    // for(size_t j=0; j<m_ndim; ++j) m_T.flat(i * m_ndim + j) = T(j);
                    continue;
                }

                // --- Tangent Calculation for Exponential Gc Model ---
                if (delta_eff <= delta0) {
                    C_local.fill(0.0);
                    C_local(0, 0) = Kn;
                    C_local(1, 1) = Kt;
                }
                else {
                    // Derivatives for D = 1 - (delta0 / delta_eff) * exp(-(delta_eff - delta0) / delta_exp_char)
                    double dD_ddelta_eff;
                    if (delta_exp_char < 1e-12) {
                        dD_ddelta_eff = 0.0; // Effectively fully damaged, tangent is zero.
                    } else {
                        double term_exp = std::exp(-(delta_eff - delta0) / delta_exp_char);
                        dD_ddelta_eff = delta0 * term_exp * (1.0/delta_eff/delta_eff + 1.0/(delta_eff * delta_exp_char));
                    }

                    double term_D = (1.0 - m_Damage.flat(i));
                    double d_delta_eff_ddelta_n = (delta_eff > 1e-12) ? delta_n / delta_eff : 0.0;
                    double d_delta_eff_ddelta_t = (delta_eff > 1e-12) ? beta * delta_t / delta_eff : 0.0;

                    // d(Tn)/d(delta_n) = (1-D)Kn - Kn*delta_n * dD/ddelta_eff * ddelta_eff/ddelta_n
                    C_local(0, 0) = term_D * Kn - Kn * delta_n * dD_ddelta_eff * d_delta_eff_ddelta_n;

                    // d(Tn)/d(delta_t) = - Kn*delta_n * dD/ddelta_eff * ddelta_eff/ddelta_t
                    C_local(0, 1) = -Kn * delta_n * dD_ddelta_eff * d_delta_eff_ddelta_t;

                    // d(Tt)/d(delta_n) = - Kt*delta_t * dD/ddelta_eff * ddelta_eff/ddelta_n
                    C_local(1, 0) = -Kt * delta_t * dD_ddelta_eff * d_delta_eff_ddelta_n;

                    // d(Tt)/d(delta_t) = (1-D)Kt - Kt*delta_t * dD/ddelta_eff * ddelta_eff/ddelta_t
                    C_local(1, 1) = term_D * Kt - Kt * delta_t * dD_ddelta_eff * d_delta_eff_ddelta_t;
                }

                xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> P_T;
                P_T(0,0) = P_matrix(0,0); P_T(0,1) = P_matrix(1,0);
                P_T(1,0) = P_matrix(0,1); P_T(1,1) = P_matrix(1,1);

                xt::xtensor_fixed<double, xt::xshape<m_ndim, m_ndim>> Temp;
                GT::A2_dot_B2(C_local.data(), P_T.data(), Temp.data());
                GT::A2_dot_B2(Temp.data(), P_matrix.data(), C.data());
            }
        }
    }

    void increment()
    {
        std::copy(m_Damage.cbegin(), m_Damage.cend(), m_Damage_t.begin());
    }
};

}
} // end namespace ConstitutiveModels
} // end namespace Cartesian2d

#endif