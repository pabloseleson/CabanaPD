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
#include <random>

#include <math.h>

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

        // Plate dimensions (m)
        double height = 0.1;
        double width = 0.04;
        double thickness = 0.002;

        // Domain
        std::array<int, 3> num_cell = { 400, 160, 8 }; // 400 x 160 x 8
        double low_x = -0.5 * height;
        double low_y = -0.5 * width;
        double low_z = -0.5 * thickness;
        double high_x = 0.5 * height;
        double high_y = 0.5 * width;
        double high_z = 0.5 * thickness;
        std::array<double, 3> low_corner = { low_x, low_y, low_z };
        std::array<double, 3> high_corner = { high_x, high_y, high_z };

        // Time
        double t_final = 43e-6;
        double dt = 5e-8;
        double output_frequency = 5;
        bool output_reference = true;

        // Material constants
        double E = 72e+9;                      // [Pa]
        double nu = 0.25;                      // unitless
        double K = E / ( 3 * ( 1 - 2 * nu ) ); // [Pa]
        double rho0 = 2440;                    // [kg/m^3]
        double G0 = 3.8;                       // [J/m^2]

        // PD horizon
        double delta = 0.001 + 1e-10;

        // FIXME: set halo width based on delta
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cell[0] ) );
        int halo_width = m + 1;

        // Multiple random pre-notches

        // Number of pre-notches
        constexpr int Npn = 200;

        // Minimum and maximum pre-notch length
        double minl = 0.001;
        double maxl = 0.0025;

        // Initialize pre-notch arrays
        Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_positions;
        Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_v1;
        Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_v2;

        // Reference for random number generator:
        // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
        std::random_device rd;
        std::mt19937 gen( rd() );
        std::uniform_real_distribution<> dis( 0.0, 1.0 );

        // Loop over pre-notches
        for ( int n = 0; n < Npn; n++ )
        {
            // Random numbers for pre-notch position
            double random_number_x = dis( gen );
            double random_number_y = dis( gen );

            // Coordinates of one endpoint of the pre-notch (random)
            // Note: the addition and subtraction of "maxl" ensures the prenotch
            // does not extend outside the domain
            double Xc1 =
                ( low_x + maxl ) +
                ( ( high_x - maxl ) - ( low_x + maxl ) ) * random_number_x;
            double Yc1 =
                ( low_y + maxl ) +
                ( ( high_y - maxl ) - ( low_y + maxl ) ) * random_number_y;
            Kokkos::Array<double, 3> p0 = { Xc1, Yc1, low_z };

            // Assign pre-notch position
            notch_positions[n] = p0;

            //  Random pre-notch length on XY-plane
            double random_number_l = dis( gen );
            double l = minl + ( maxl - minl ) * random_number_l;

            // Random pre-notch orientation on XY-plane
            double random_number_theta = dis( gen );
            double theta = M_PI * random_number_theta;

            // Pre-notch v1 vector
            Kokkos::Array<double, 3> v1 = { l * cos( theta ), l * sin( theta ),
                                            0 };
            notch_v1[n] = v1;

            // Pre-notch v2 vector

            // Random number for y-component of v2: the angle of v2 in the
            // YZ-plane is between 0 and 45 deg.
            double random_number_v2_y = dis( gen );
            Kokkos::Array<double, 3> v2 = { 0, random_number_v2_y * thickness,
                                            thickness };
            notch_v2[n] = v2;
        }

        CabanaPD::Prenotch<Npn> prenotch( notch_v1, notch_v2, notch_positions );

        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
        model_type force_model( delta, K, G0 );
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
        auto v = particles->sliceVelocity();
        auto f = particles->sliceForce();
        auto rho = particles->sliceDensity();
        auto nofail = particles->sliceNoFail();

        // Relying on uniform grid here.
        double dy = particles->dx[1];
        double b0 = 2e6 / dy; // Pa/m

        CabanaPD::RegionBoundary plane1( low_x, high_x, low_y - dy, low_y + dy,
                                         low_z, high_z );
        CabanaPD::RegionBoundary plane2( low_x, high_x, high_y - dy,
                                         high_y + dy, low_z, high_z );
        std::vector<CabanaPD::RegionBoundary> planes = { plane1, plane2 };
        auto bc =
            createBoundaryCondition( CabanaPD::ForceCrackBranchBCTag{},
                                     exec_space{}, *particles, planes, b0 );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
            // Set the no-fail zone.
            if ( x( pid, 1 ) <= plane1.low_y + delta + 1e-10 ||
                 x( pid, 1 ) >= plane2.high_y - delta - 1e-10 )
                nofail( pid ) = 1;
        };
        particles->updateParticles( exec_space{}, init_functor );

        // FIXME: use createSolver to switch backend at runtime.
        auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
            inputs, particles, force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}