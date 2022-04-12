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

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_Comm.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_config.hpp>

namespace Test
{

//---------------------------------------------------------------------------//
void testHalo()
{
    using exec_space = TEST_EXECSPACE;
    using mem_space = TEST_MEMSPACE;
    using device_type = TEST_DEVICE;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    double delta = 0.20000001;
    int halo_width = 2;
    // FIXME: This is for m = 1; should be calculated from m
    int expected_n = 6;
    CabanaPD::Particles<mem_space> particles( exec_space(), box_min, box_max,
                                              num_cells, halo_width );
    // Set ID equal to MPI rank.
    int rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    auto x = particles.slice_x();
    auto id = particles.slice_id();
    auto init_functor = KOKKOS_LAMBDA( const int pid ) { id( pid ) = rank; };
    particles.update_particles( exec_space{}, init_functor );

    int init_num_particles = particles.n_local;
    using HostAoSoA =
        Cabana::AoSoA<Cabana::MemberTypes<double[3], int>, Kokkos::HostSpace>;
    HostAoSoA aosoa_init_host( "host_aosoa", init_num_particles );
    auto x_init_host = Cabana::slice<0>( aosoa_init_host );
    auto id_init_host = Cabana::slice<1>( aosoa_init_host );
    Cabana::deep_copy( x_init_host, x );
    Cabana::deep_copy( id_init_host, id );

    CabanaPD::Comm<device_type> comm( particles );

    HostAoSoA aosoa_host( "host_aosoa", particles.size );
    x = particles.slice_x();
    id = particles.slice_id();
    auto x_host = Cabana::slice<0>( aosoa_host );
    auto id_host = Cabana::slice<1>( aosoa_host );
    Cabana::deep_copy( x_host, x );
    Cabana::deep_copy( id_host, id );

    EXPECT_EQ( particles.n_local, init_num_particles );

    // Check all local particles unchanged.
    for ( int p = 0; p < particles.n_local; ++p )
    {
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_EQ( x_host( p, d ), x_init_host( p, d ) );
        }
        EXPECT_EQ( id_host( p ), id_init_host( p ) );
    }

    // Check all ghost particles in the halo region.
    for ( int p = particles.n_local; p < particles.size; ++p )
    {
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_GE( x_host( p, d ), particles.ghost_mesh_lo[d] );
            EXPECT_LE( x_host( p, d ), particles.ghost_mesh_hi[d] );
        }
        EXPECT_NE( id_host( p ), rank );
    }

    double mesh_min[3] = { particles.ghost_mesh_lo[0],
                           particles.ghost_mesh_lo[1],
                           particles.ghost_mesh_lo[2] };
    double mesh_max[3] = { particles.ghost_mesh_hi[0],
                           particles.ghost_mesh_hi[1],
                           particles.ghost_mesh_hi[2] };
    using NeighListType =
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    NeighListType nlist( x, 0, particles.n_local, delta, 1.0, mesh_min,
                         mesh_max );

    // Check that all local particles (away from global boundaries) have a full
    // set of neighbors.
    // FIXME: Expected neighbors per particle could also be calculated at the
    // boundaries (less than internal particles).
    for ( int p = 0; p < particles.n_local; ++p )
    {
        if ( x_host( p, 0 ) > box_min[0] + delta * 1.01 &&
             x_host( p, 0 ) < box_max[0] - delta * 1.01 &&
             x_host( p, 1 ) > box_min[1] + delta * 1.01 &&
             x_host( p, 1 ) < box_max[1] - delta * 1.01 &&
             x_host( p, 2 ) > box_min[2] + delta * 1.01 &&
             x_host( p, 2 ) < box_max[2] - delta * 1.01 )
        {
            auto n =
                Cabana::NeighborList<NeighListType>::numNeighbor( nlist, p );
            EXPECT_EQ( n, expected_n );
        }
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_particle_halo ) { testHalo(); }

//---------------------------------------------------------------------------//

} // end namespace Test