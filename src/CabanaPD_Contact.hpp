/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#ifndef CONTACT_H
#define CONTACT_H

#include <cmath>

#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{
/******************************************************************************
  Contact model
******************************************************************************/
struct ContactModel
{
    double delta;
    double Rc;

    ContactModel(){};
    ContactModel( const double _delta, const double _Rc, const PosType& x, const PosType& u )
        : delta( _delta ){}, // PD horizon
        : Rc( _Rc ){};       // Contact radius

        // Create contact neighbor list
        double mesh_min[3] = { particles->ghost_mesh_lo[0],
                               particles->ghost_mesh_lo[1],
                               particles->ghost_mesh_lo[2] };
        double mesh_max[3] = { particles->ghost_mesh_hi[0],
                               particles->ghost_mesh_hi[1],
                               particles->ghost_mesh_hi[2] };
        const auto x = particles->slice_x();
        auto u = particles->slice_u();
        auto y = x + u;
        contact_neighbors = std::make_shared<neighbor_type>( y, 0, particles->n_local,
                                                     contact_model.Rc, 1.0,
                                                     mesh_min, mesh_max );
};

/* Normal repulsion */

struct NormalRepulsionModel : public ContactModel
{
    using ContactModel::delta;
    using ContactModel::Rc;

    double c;
    double K;

    NormalRepulsionModel(){};
    NormalRepulsionModel( const double delta, const double Rc, const double K )
        : ContactModel( delta ),
        : ContactModel( Rc )
    {
        set_param( delta, Rc, K );
    }

    void set_param( const double _delta, const double _Rc, const double _K )
    {
        delta = _delta;
        Rc = _Rc;
        K = _K;
        c = 18.0 * K / ( 3.141592653589793 * delta * delta * delta * delta );
    }
};

/******************************************************************************
  Contact helper functions (similar to CabanaPD_Force.hpp but without computing stretch)
******************************************************************************/
template <class PosType>
KOKKOS_INLINE_FUNCTION void
getDistanceComponents( const PosType& x, const PosType& u, const int i,
                       const int j, double& xi, double& r, // double& s,
                       double& rx, double& ry, double& rz )
{
    // Get the reference positions and displacements.
    const double xi_x = x( j, 0 ) - x( i, 0 );
    const double eta_u = u( j, 0 ) - u( i, 0 );
    const double xi_y = x( j, 1 ) - x( i, 1 );
    const double eta_v = u( j, 1 ) - u( i, 1 );
    const double xi_z = x( j, 2 ) - x( i, 2 );
    const double eta_w = u( j, 2 ) - u( i, 2 );
    rx = xi_x + eta_u;
    ry = xi_y + eta_v;
    rz = xi_z + eta_w;
    r = sqrt( rx * rx + ry * ry + rz * rz );
    xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
 //   s = ( r - xi ) / xi;
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void getDistance( const PosType& x, const PosType& u,
                                         const int i, const int j, double& xi,
                                         double& r) //, double& s )
{
    double rx, ry, rz;
    //getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );
    getDistanceComponents( x, u, i, j, xi, r, rx, ry, rz );
}

/******************************************************************************
  Normal repulsion computation
******************************************************************************/
template <class ExecutionSpace>
class Contact<ExecutionSpace, NormalRepulsionModel>
{
  protected:
    bool _half_neigh;
    NormalRepulsionModel _model;

  public:
    using exec_space = ExecutionSpace;

    Contact( const bool half_neigh, const NormalRepulsionModel model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ContactType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void compute_contact_full( ContactType& fc, const PosType& x, const PosType& u,
                             const ParticleType& particles,
                             const NeighListType& neigh_list, const int n_local,
                             ParallelType& neigh_op_tag ) const
    {
        auto delta = _model.delta;
        auto Rc = _model.Rc;
        auto c = _model.c;
        const auto vol = particles.slice_vol();

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            // getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );
            getDistanceComponents( x, u, i, j, xi, r, rx, ry, rz );
            // Normal repulsion uses a 15 factor compared to the PMB force

            // Contact "stretch"
            const double sc = (r - Rc)/delta;

            const double coeff = 15 * c * sc * vol( j );
            fx_i = coeff * rx / r;
            fy_i = coeff * ry / r;
            fz_i = coeff * rz / r;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, contact_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForcePMB::compute_full" );
    }
};

template <class ExecutionSpace>
class Force<ExecutionSpace, PMBDamageModel>
    : public Force<ExecutionSpace, PMBModel>
{
  protected:
    using base_type = Force<ExecutionSpace, PMBModel>;
    using base_type::_half_neigh;
    PMBDamageModel _model;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const PMBDamageModel model )
        : base_type( half_neigh, model )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    void compute_force_full( ForceType& f, const PosType& x, const PosType& u,
                             const ParticleType& particles,
                             const NeighListType& neigh_list, MuView& mu,
                             const int n_local, ParallelType& ) const
    {
        auto c = _model.c;
        auto break_coeff = _model.bond_break_coeff;
        const auto vol = particles.slice_vol();

        auto force_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                if ( mu( i, n ) > 0 )
                {
                    double fx_i = 0.0;
                    double fy_i = 0.0;
                    double fz_i = 0.0;

                    std::size_t j =
                        Cabana::NeighborList<NeighListType>::getNeighbor(
                            neigh_list, i, n );

                    // Get the reference positions and displacements.
                    double xi, r, s;
                    double rx, ry, rz;
                    getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

                    if ( r * r >= break_coeff * xi * xi )
                        mu( i, n ) = 0;
                    if ( mu( i, n ) > 0 )
                    {
                        const double coeff = c * s * vol( j );
                        double muij = mu( i, n );
                        fx_i = muij * coeff * rx / r;
                        fy_i = muij * coeff * ry / r;
                        fz_i = muij * coeff * rz / r;

                        f( i, 0 ) += fx_i;
                        f( i, 1 ) += fy_i;
                        f( i, 2 ) += fz_i;
                    }
                }
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::ForcePMBDamage::compute_full", policy,
                              force_full );
    }
};

template <class ForceType, class ParticleType, class NeighListType,
          class ParallelType>
void compute_force( const ForceType& force, ParticleType& particles,
                    const NeighListType& neigh_list,
                    const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto x = particles.slice_x();
    auto u = particles.slice_u();
    auto f = particles.slice_f();
    auto f_a = particles.slice_f_a();

    // Reset force.
    Cabana::deep_copy( f, 0.0 );

    // if ( half_neigh )
    // Forces must be atomic for half list
    // compute_force_half( f_a, x, u, neigh_list, n_local,
    //                    neigh_op_tag );

    // Forces only atomic if using team threading
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        force.compute_force_full( f_a, x, u, particles, neigh_list, n_local,
                                  neigh_op_tag );
    else
        force.compute_force_full( f, x, u, particles, neigh_list, n_local,
                                  neigh_op_tag );
    Kokkos::fence();
}

} // namespace CabanaPD

#endif