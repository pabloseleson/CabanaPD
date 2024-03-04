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
        // ====================================================
        //                      ???
        // ====================================================
        Kokkos::ScopeGuard scope_guard( argc, argv );

        // FIXME: change backend at compile time for now.
        using exec_space = Kokkos::DefaultExecutionSpace;
        using memory_space = typename exec_space::memory_space;

        // ====================================================
        //                   Read inputs
        // ====================================================
        CabanaPD::Inputs inputs( argv[1] );

        // ====================================================
        //                Material parameters
        // ====================================================
        double rho0 = inputs["density"];
        double E = inputs["elastic_modulus"];
        double nu = 0.25;                      
        double K = E / ( 3 * ( 1 - 2 * nu ) ); 
        double delta = inputs["horizon"];
        double alpha = inputs["thermal_coeff"];
        // Reference temperature
        // double temp0 = 0.0;

        // ====================================================
        //                  Discretization
        // ====================================================
        std::array<double, 3> low_corner = inputs["low_corner"];
        std::array<double, 3> high_corner = inputs["high_corner"];
        std::array<int, 3> num_cells = inputs["num_cells"];
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // ====================================================
        //                    Force model
        // ====================================================
        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Elastic>;
        // model_type force_model( delta, K );
        model_type force_model( delta, K, alpha );
        // using model_type =
        //     CabanaPD::ForceModel<CabanaPD::LinearLPS, CabanaPD::Elastic>;
        // model_type force_model( delta, K, G );

        // ====================================================
        //                  Grid generation
        // ====================================================
        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<
            CabanaPD::Particles<memory_space, typename model_type::base_model>>(
            exec_space(), low_corner, high_corner, num_cells, halo_width );     
        particles->createParticles( exec_space());

        // ====================================================
        //             Particle fields initialization
        // ====================================================
        // Define particle initialization.
        auto rho = particles->sliceDensity();
        auto x = particles->sliceReferencePosition();
        auto u = particles->sliceDisplacement();
        auto v = particles->sliceVelocity();
        auto f = particles->sliceForce();
        // auto temp = particles->sliceTemperature();

        // ====================================================
        //                Boundary conditions
        // ====================================================
        // Domain to apply b.c.
        CabanaPD::RegionBoundary domain1( low_corner[0], high_corner[0],
                                          low_corner[1], high_corner[1],
                                          low_corner[2], high_corner[2] );
        std::vector<CabanaPD::RegionBoundary> domain = { domain1 };

        auto bc = createBoundaryCondition( CabanaPD::TempBCTag{}, 5000.0,
                                           exec_space{}, *particles, domain );

        // ====================================================
        //               Particle fields setting
        // ====================================================
        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
        };
        particles->updateParticles( exec_space{}, init_functor );

        // ====================================================
        //                   Simulation run
        // ====================================================
        auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
            inputs, particles, force_model, bc );
        cabana_pd->init_force();
        cabana_pd->run();

        // ====================================================
        //                      Outputs
        // ====================================================
        
        // Displacement profiles in x-direction

        double num_cell_x = num_cells[0];
        auto profile = Kokkos::View<double* [3], memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
            num_cell_x );
        int mpi_rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        Kokkos::View<int*, memory_space> count( "c", 1 );

        double dx = particles->dx[0];

        auto measure_profile = KOKKOS_LAMBDA( const int pid )
        {
            if ( x( pid, 1 ) < dx / 2.0 && x( pid, 1 ) > -dx / 2.0 &&
                 x( pid, 2 ) < dx / 2.0 && x( pid, 2 ) > -dx / 2.0 )
            {
                auto c = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
                profile( c, 0 ) = x( pid, 0 );
                profile( c, 1 ) = u( pid, 1 );
                profile( c, 2 ) = std::sqrt(u( pid, 0 )*u( pid, 0 ) + u( pid, 1 )*u( pid, 1 ) + u( pid, 2 )*u( pid, 2 ));
            }
        };
        Kokkos::RangePolicy<exec_space> policy( 0, x.size() );
        Kokkos::parallel_for( "displacement_profile", policy, measure_profile );
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, count );
        auto profile_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profile );
        std::fstream fout;

        std::string file_name = "displacement_profile_x_direction.txt";
        fout.open( file_name, std::ios::app );
        for ( int p = 0; p < count_host( 0 ); p++ )
        {
            fout << mpi_rank << " " << profile_host( p, 0 ) << " "
                 << profile_host( p, 1 ) << " "
                 << profile_host( p, 2 ) << std::endl;
        }

        // Displacement profiles in y-direction

        // double num_cell_x = num_cells[0];
        auto profile_y = Kokkos::View<double* [3], memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
            num_cell_x );
        // int mpi_rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        Kokkos::View<int*, memory_space> count_y( "c", 1 );

        // double dx = particles->dx[0];

        auto measure_profile_y = KOKKOS_LAMBDA( const int pid )
        {
            if ( x( pid, 0 ) < dx / 2.0 && x( pid, 0 ) > -dx / 2.0 &&
                 x( pid, 2 ) < dx / 2.0 && x( pid, 2 ) > -dx / 2.0 )
            {
                auto c = Kokkos::atomic_fetch_add( &count_y( 0 ), 1 );
                profile_y( c, 0 ) = x( pid, 1 );
                profile_y( c, 1 ) = u( pid, 1 );
                profile_y( c, 2 ) = std::sqrt(u( pid, 0 )*u( pid, 0 ) + u( pid, 1 )*u( pid, 1 ) + u( pid, 2 )*u( pid, 2 ));
            }
        };
        Kokkos::RangePolicy<exec_space> policy_y( 0, x.size() );
        Kokkos::parallel_for( "displacement_profile", policy_y, measure_profile_y );
        auto count_host_y =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, count_y );
        auto profile_host_y =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profile_y );
        std::fstream fout_y;

        std::string file_name_y = "displacement_profile_y_direction.txt";
        fout_y.open( file_name_y, std::ios::app );
        for ( int p = 0; p < count_host_y( 0 ); p++ )
        {
            fout_y << mpi_rank << " " << profile_host_y( p, 0 ) << " "
                 << profile_host_y( p, 1 ) << " "
                 << profile_host_y( p, 2 ) << std::endl;
        }

    }

    MPI_Finalize();
}
