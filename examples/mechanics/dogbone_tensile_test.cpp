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

#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

// Simulate ASTM D638 type I dogbone tensile test.
void compactTensionTestExample( const std::string filename )
{
    // ====================================================
    //               Choose Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                Material parameters
    // ====================================================
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25; // Use bond-based model
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double sigma_y = inputs["yield_stress"];
    double delta = inputs["horizon"];
    delta += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                Force model type
    // ====================================================
    using model_type = CabanaPD::PMB;
    using thermal_type = CabanaPD::TemperatureIndependent;
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double LO = inputs["system_size"][0]; // Length overall
    double WO = inputs["system_size"][1]; // Width overall
    double T = inputs["system_size"][2];  // Thickness
    double G = inputs["gage_length"];
    double W = inputs["width_narrow_section"];
    double R = inputs["fillet_radius"];
    double D = inputs["distance_between_grips"];

    // x- and y-coordinates of center of domain
    double midx = 0.5 * ( low_corner[0] + high_corner[0] );
    double midy = 0.5 * ( low_corner[1] + high_corner[1] );

    // Do not create particles outside dogbone tensile test specimen region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Filler radius squared
        double Rsq = R * R;

        // Bottom-left fillet circle center
        double xc_bl = midx - 0.5 * G;
        double yc_bl = midy - 0.5 * W - R;

        // Bottom-right fillet circle center
        double xc_br = midx + 0.5 * G;
        double yc_br = midy - 0.5 * W - R;

        // Top-left fillet circle center
        double xc_tl = midx - 0.5 * G;
        double yc_tl = midy + 0.5 * W + R;

        // Top-right fillet circle center
        double xc_tr = midx + 0.5 * G;
        double yc_tr = midy + 0.5 * W + R;

        // Gauge section
        if ( Kokkos::abs( x[0] - midx ) < 0.5 * G &&
             Kokkos::abs( x[1] - midy ) > 0.5 * W )
        {
            return false;
        }
        // Bottom-left fillet
        else if ( Kokkos::abs( x[0] - xc_bl ) * Kokkos::abs( x[0] - xc_bl ) +
                      Kokkos::abs( x[1] - yc_bl ) *
                          Kokkos::abs( x[1] - yc_bl ) <
                  Rsq )
        {
            return false;
        }
        // Bottom-right fillet
        else if ( Kokkos::abs( x[0] - xc_br ) * Kokkos::abs( x[0] - xc_br ) +
                      Kokkos::abs( x[1] - yc_br ) *
                          Kokkos::abs( x[1] - yc_br ) <
                  Rsq )
        {
            return false;
        }
        // Top-left fillet
        else if ( Kokkos::abs( x[0] - xc_tl ) * Kokkos::abs( x[0] - xc_tl ) +
                      Kokkos::abs( x[1] - yc_tl ) *
                          Kokkos::abs( x[1] - yc_tl ) <
                  Rsq )
        {
            return false;
        }
        // Top-right fillet
        else if ( Kokkos::abs( x[0] - xc_tr ) * Kokkos::abs( x[0] - xc_tr ) +
                      Kokkos::abs( x[1] - yc_tr ) *
                          Kokkos::abs( x[1] - yc_tr ) <
                  Rsq )
        {
            return false;
        }
        else
        {
            return true;
        }
    };

    auto particles =
        CabanaPD::createParticles<memory_space, model_type, thermal_type>(
            exec_space(), low_corner, high_corner, num_cells, halo_width,
            init_op );

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto v = particles->sliceVelocity();
    auto nofail = particles->sliceNoFail();

    auto dx = particles->dx;
    double factor = inputs["grid_perturbation_factor"];

    // Grips' velocity magnitude
    double v0 = inputs["grip_velocity"];

    // Create region for each grip.
    CabanaPD::Region<CabanaPD::RectangularPrism> left_grip(
        low_corner[0], low_corner[0] + 0.025, low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> right_grip(
        high_corner[0] - 0.025, high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

    using pool_type = Kokkos::Random_XorShift64_Pool<exec_space>;
    using random_type = Kokkos::Random_XorShift64<exec_space>;
    pool_type pool;
    int seed = 456854;
    pool.init( seed, particles->localOffset() );

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // Perturb particle positions in gauge region
        auto gen = pool.get_state();
        for ( std::size_t d = 0; d < 3; d++ )
        {
            // if ( x( pid, 0 ) >= midx - 0.5*G &&  x( pid, 0 ) <= midx + 0.5*G
            // )
            if ( x( pid, 0 ) >= midx - 0.1 * G &&
                 x( pid, 0 ) <= midx + 0.1 * G )
            {
                auto rand =
                    Kokkos::rand<random_type, double>::draw( gen, 0.0, 1.0 );
                x( pid, d ) += ( 2.0 * rand - 1.0 ) * factor * dx[d];
            }
        }
        pool.free_state( gen );

        // grips' x-velocity
        if ( left_grip.inside( x, pid ) )
            v( pid, 0 ) = -v0;
        else if ( right_grip.inside( x, pid ) )
            v( pid, 0 ) = v0;

        // No-fail zone
        if ( x( pid, 0 ) >= low_corner[0] && x( pid, 0 ) <= high_corner[1] )
            nofail( pid ) = 1;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto force_model = CabanaPD::createForceModel(
        model_type{}, mechanics_type{}, *particles, delta, K, G0, sigma_y );

    // PMB only
    // CabanaPD::ForceModel<model_type> force_model( delta, K, G0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd =
        CabanaPD::createSolver<memory_space>( inputs, particles, force_model );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Reset forces on both grips.
    std::vector<CabanaPD::Region<CabanaPD::RectangularPrism>> grips = {
        left_grip, right_grip };
    auto bc = createBoundaryCondition( CabanaPD::ForceValueBCTag{}, 0.0,
                                       exec_space{}, *particles, grips, true );

    // ====================================================
    //                      Outputs
    // ====================================================

    // Create region for gauge.
    CabanaPD::Region<CabanaPD::RectangularPrism> half_gage_region(
        midx, midx + 0.5 * G, low_corner[1], high_corner[1], low_corner[2],
        high_corner[2] );

    auto output = CabanaPD::createOutputTimeSeries(
        CabanaPD::ForceDisplacementTag{}, "gage_force_displacement.txt", inputs,
        exec_space{}, *particles, half_gage_region );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init();
    // cabana_pd->run( bc );
    cabana_pd->run( bc, output );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    compactTensionTestExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
