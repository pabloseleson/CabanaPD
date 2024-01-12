/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        // FIXME: change backend at compile time for now.
        using exec_space = Kokkos::DefaultExecutionSpace;
        using memory_space = typename exec_space::memory_space;

        std::array<int, 3> num_cell = { 101, 31, 3 };
        std::array<double, 3> low_corner = { -0.5, 0.0, -0.015 };
        std::array<double, 3> high_corner = { 0.5, 0.3, 0.015 };
        double t_final = 0.0093;
        double dt = 7.5E-7;
        int output_frequency = 200;
        bool output_reference = true;

        double rho0 = 3980;                    // [kg/m^3]
        double E = 370e+9;                     // [Pa]
        double nu = 0.25;                      // unitless
        double K = E / ( 3 * ( 1 - 2 * nu ) ); // [Pa]
        double alpha = 7.5e-6;                 // [1/oC]
        double delta = 0.03;

        // Reference temperature
        double temp0 = 0.0;

        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cell[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Elastic>;
        // model_type force_model( delta, K );
        model_type force_model( delta, K, alpha );
        // using model_type =
        //     CabanaPD::ForceModel<CabanaPD::LinearLPS, CabanaPD::Elastic>;
        // model_type force_model( delta, K, G );

        CabanaPD::Inputs<3> inputs( num_cell, low_corner, high_corner, t_final,
                                    dt, output_frequency, output_reference );
        inputs.read_args( argc, argv );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<
            CabanaPD::Particles<memory_space, typename model_type::base_model>>(
            exec_space(), inputs.low_corner, inputs.high_corner,
            inputs.num_cells, halo_width );

        // Define particle initialization.
        auto x = particles->sliceReferencePosition();
        auto u = particles->sliceDisplacement();
        auto f = particles->sliceForce();
        auto v = particles->sliceVelocity();
        auto rho = particles->sliceDensity();
        // auto temp = particles->sliceTemperature();

        // Domain to apply b.c.
        CabanaPD::RegionBoundary domain1( low_corner[0], high_corner[0],
                                          low_corner[1], high_corner[1],
                                          low_corner[2], high_corner[2] );
        std::vector<CabanaPD::RegionBoundary> domain = { domain1 };

        double b0 = 0.0;
        auto bc =
            createBoundaryCondition( CabanaPD::ForceCrackBranchBCTag{},
                                     exec_space{}, *particles, domain, b0 );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            // temp( pid ) = 5000 * x( pid, 1 ) * t_final;
            rho( pid ) = rho0;
        };
        particles->updateParticles( exec_space{}, init_functor );

        auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
            inputs, particles, force_model, bc );
        cabana_pd->init_force();
        cabana_pd->run();

        // ============================
        // Output displacement profiles
        // ============================
        x = particles->sliceReferencePosition();
        u = particles->sliceDisplacement();
        double num_cell_x = inputs.num_cells[0];
        auto profile = Kokkos::View<double* [2], memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
            num_cell_x );
        int mpi_rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        Kokkos::View<int*, memory_space> count( "c", 1 );
        double dx = particles->dx[0];
        double midx = 0.5 * ( low_corner[0] + high_corner[0] );
        double midy = 0.5 * ( low_corner[1] + high_corner[1] );
        double midz = 0.5 * ( low_corner[2] + high_corner[2] );
        auto measure_profile = KOKKOS_LAMBDA( const int pid )
        {
            // if ( x( pid, 1 ) < midy + dx / 2.0 && x( pid, 1 ) > midy - dx
            // / 2.0 &&
            if ( x( pid, 0 ) < midx + dx / 2.0 &&
                 x( pid, 0 ) > midx - dx / 2.0 &&
                 x( pid, 2 ) < midz + dx / 2.0 &&
                 x( pid, 2 ) > midz - dx / 2.0 )
            {
                auto c = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
                // profile( c, 0 ) = x( pid, 0 );
                profile( c, 0 ) = x( pid, 1 );
                profile( c, 1 ) = u( pid, 1 ); // y displacement
                profile( c, 2 ) = std::sqrt(
                    u( pid, 0 ) * u( pid, 0 ) + u( pid, 1 ) * u( pid, 1 ) +
                    u( pid, 2 ) * u( pid, 2 ) ); // displacement magnitude
            }
        };
        Kokkos::RangePolicy<exec_space> policy( 0, x.size() );
        Kokkos::parallel_for( "displacement_profile", policy, measure_profile );
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, count );
        auto profile_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profile );
        std::fstream fout;
        // std::string file_name = "displacement_profile_x_direction.txt";
        std::string file_name = "displacement_profile_y_direction.txt";
        fout.open( file_name, std::ios::app );
        for ( int p = 0; p < count_host( 0 ); p++ )
        {
            fout << mpi_rank << " " << profile_host( p, 0 ) << " "
                 << profile_host( p, 1 ) << " " << profile_host( p, 2 )
                 << std::endl;
        }
    }

    MPI_Finalize();
}
