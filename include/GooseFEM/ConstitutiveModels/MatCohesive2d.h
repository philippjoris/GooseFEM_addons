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
class Cohesive : public GMatTensor::Cartesian2d::Array<N> {
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
    // array_type::tensor<double, N> m_Gc; ///< Fracture energy per item.        

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

    Cohesive() = default;

    /**
    Construct system.
    \param Kn Normal separation stiffness per item.
    \param Kt Tangential separation stiffness per item.
    \param delta0 Initial displacement per item.    
    \param beta Initial displacement per item.    
    */
    template <class T>
    Cohesive(
        const T& Kn,
        const T& Kt,
        const T& delta0,
        const T& deltafrac,
        const T& beta
    )
    {
        GOOSEFEM_ASSERT(Kn.dimension() == N);
        GOOSEFEM_ASSERT(Kt.dimension() == N);        
        GOOSEFEM_ASSERT(delta0.dimension() == N);                
        GOOSEFEM_ASSERT(beta.dimension() == N);                        
        GOOSEFEM_ASSERT(xt::has_shape(Kn, Kt.shape()));
        std::copy(Kn.shape().cbegin(), Kn.shape().cend(), m_shape.begin());
        // std::copy(Kt.shape().cbegin(), Kt.shape().cend(), m_shape.begin());        
        this->init(m_shape);

        m_Kn = Kn;
        m_Kt= Kt;
        m_delta0 = delta0;
        m_deltafrac = deltafrac;
        m_beta = beta;        
        
        m_delta = xt::zeros<double>(m_shape_tensor1); 
        m_delta_eff = xt::zeros<double>(m_shape);
        m_T_local = xt::empty<double>(m_shape_tensor1);
        m_C_local = xt::empty<double>(m_shape_tensor2);
        m_Damage = xt::zeros<double>(m_shape);
        m_Damage_t = m_Damage; 
        m_failed = xt::zeros<bool>(m_shape);

        // rotation matrix between local and global coordinates
        m_P_matrix = xt::empty<double>(m_shape_tensor2);;

        m_T = xt::empty<double>(m_shape_tensor1); // Global tractions (results of refresh)
        m_C = xt::empty<double>(m_shape_tensor2); // Global tangent stiffness        

        this->refresh();
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

    template <class T>
    void set_delta(const T& arg)
    {
        GOOSEFEM_ASSERT(xt::has_shape(arg, m_shape_tensor2));
        std::copy(arg.cbegin(), arg.cend(), m_delta.begin());
        this->refresh();
    }

    template <class T>
    void set_ori(const T& arg)
    {
        GOOSEFEM_ASSERT(xt::has_shape(arg, m_shape_tensor4));
        std::copy(arg.cbegin(), arg.cend(), m_P_matrix.begin());
        this->refresh();
    }


    /*
    This refresh function returns the traction vector based on the displacement jump.

    If compute_tangent flag, it computes the cohesive tangent stifness per element. For bi-linar
    traction-separation law, the tangent stiffness is the initial stiffness multiplied by a damage factor.

    Compute delta_eff = sqrt{ pow(max(delta_n), 2) + beta * pow(max(delta_t), 2) }
    Compute Damage D: {\frac{(delta_eff - delta0)}{(deltaf - delta0)}}

    Note though that you can call this function as often as you like, you will only loose time.
    */
    void refresh(bool compute_tangent = true, bool element_erosion = true)
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
                double G = m_deltafrac.flat(i) - m_delta0.flat(i); GOOSEFEM_ASSERT(G > 1e-12);
                                
                double damage_t = m_Damage_t.flat(i);
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
                    continue;
                }                
                
                // Step 1: decompose delta into delta_n and delta_t
                double delta_n = std::max(0.0, delta[0]);
                double delta_t = delta[1];

                // Step 2: calculate effective displacement jump delta_eff = sqrt{ pow(max(delta_n), 2) + beta * pow(max(delta_t), 2) }
                m_delta_eff.flat(i) = std::sqrt(delta_n * delta_n + beta * delta_t * delta_t);

                double delta_eff = m_delta_eff.flat(i);
                // Step 3: calculate Damage value
                double D_trial;
                if (delta_eff <= delta0) {
                    D_trial = 0.0;
                } else if (delta_eff >= deltafrac) {
                    D_trial = 1.0;
                } else {
                    if (std::abs(G) < 1e-12) { 
                        D_trial = 1.0;
                    } else {
                        D_trial = (deltafrac * (delta_eff - delta0)) / (delta_eff * G);
                    }
                }

                double current_D = std::max(D_trial, damage_t);
                current_D = std::min(current_D, 1.0);
                m_Damage.flat(i) = current_D;

                // Shall I limit the damage to one? What if damage is averaged for a cell?
                if (element_erosion && m_Damage.flat(i) >= 1.0) {
                    m_failed.flat(i) = true; 
                    }                    
                
                // Step 4: Calculate traction vector
                T_local(0) = (1 - m_Damage.flat(i)) * Kn * delta_n;
                T_local(1) = (1 - m_Damage.flat(i)) * Kt * delta_t;    
                
                // Transform local traction vector to global coordinates
                GT::A2_dot_B1(P_matrix.data(), T_local.data(), T.data());
                
                // --- Tangent Calculation ---
                if (!compute_tangent) {
                    return;
                }
                
                // Step 4: Calculate local tangent stiffness C_local
                if (delta_eff <= delta0){
                    C_local(0, 0) = Kn; 
                    for (size_t k = 0; k < m_ndim - 1; ++k) {
                        C_local(k + 1, k + 1) = Kt; 
                    }
                    C_local(0,1) = 0.0;
                    C_local(1,0) = 0.0;
                }
                else if (delta_eff >= deltafrac){
                    C_local.fill(0.0);
                }
                else {
                    C_local(0,0) = (1-m_Damage.flat(i))*Kn - (Kn * delta_n * delta_n)/(G * delta_eff);
                    C_local(0,1) = -(Kn * beta * delta_n * delta_t)/(G * delta_eff);
                    C_local(1,0) = -(Kt * delta_n * delta_t)/(G * delta_eff);
                    C_local(1,1) = (1-m_Damage.flat(i)) * Kt - (Kt * beta * delta_t * delta_t)/(G * delta_eff);
                }            
                    
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
    }
};
}
} // end namespace ConstitutiveModels
} // end namespace Cartesian2d

#endif