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

// Simulate ASTM D5379/D5379M V-notched beam test.
void vnotchedBeamTestExample( const std::string filename )
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
    using mechanics_type = CabanaPD::Elastic;

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    // Domain parameters
    double d1 = inputs["system_size"][1];
    double r = inputs["notch_radius"];
    double W = inputs["distance_between_notches"];

    // Auxiliary variables
    double alpha = 45 * CabanaPD::pi / 180;
    double d2 = ( d1 - W ) / 2;
    double d3 = r * ( 1 - sin( alpha ) );
    double d4 = d2 - d3;

    // x- and y-coordinates of center of domain
    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );

    // Do not create particles outside V-notched beam test specimen region.
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Initialize flag
        double flag_create = 1;

        // -----------------------
        //  Top half of specimen
        // -----------------------

        // Top circle: x- and y-coordinates of center
        double xc_top = x_center;
        double yc_top = y_center + 0.5 * W + r;

        // Top circle
        if ( Kokkos::abs( x[0] - xc_top ) * Kokkos::abs( x[0] - xc_top ) +
                 Kokkos::abs( x[1] - yc_top ) * Kokkos::abs( x[1] - yc_top ) <
             r * r )
            flag_create = 0;

        // y-position of line to remove points above it
        double y_line_top = y_center + 0.5 * W + d3;
        // x-distance from center to circle's intersections with specimen
        double xdist_min = r * std::sin( alpha );
        // x-distance from center to openings on top of specimen
        double xdist_max = xdist_min + d4 * std::tan( alpha );

        if ( x[1] > y_line_top )
        {
            // Top half within x-distance r*sin(45o) from center
            if ( Kokkos::abs( x[0] - x_center ) < xdist_min )
            {
                flag_create = 0;
            }
            // Top half within side triangles
            else if ( Kokkos::abs( x[0] - x_center ) < xdist_max )
            {
                if ( x[1] - y_line_top >
                     Kokkos::abs( x[0] - x_center ) - xdist_min )
                    flag_create = 0;
            };
        };

        // -----------------------
        // Bottom half of specimen
        // -----------------------

        // Bottom circle: x- and y-coordinates of center
        double xc_bot = x_center;
        double yc_bot = y_center - 0.5 * W - r;

        // Top circle
        if ( Kokkos::abs( x[0] - xc_bot ) * Kokkos::abs( x[0] - xc_bot ) +
                 Kokkos::abs( x[1] - yc_bot ) * Kokkos::abs( x[1] - yc_bot ) <
             r * r )
            flag_create = 0;

        // y-position of line to remove points below it
        double y_line_bot = y_center - 0.5 * W - d3;

        if ( x[1] < y_line_bot )
        {
            // Bottom half within x-distance r*sin(45o) from center
            if ( Kokkos::abs( x[0] - x_center ) < xdist_min )
            {
                flag_create = 0;
            }
            // Bottom half within side triangles
            else if ( Kokkos::abs( x[0] - x_center ) < xdist_max )
            {
                if ( y_line_bot - x[1] >
                     Kokkos::abs( x[0] - x_center ) - xdist_min )
                    flag_create = 0;
            };
        };

        // Determine if to create particle
        if ( flag_create == 1 )
        {
            return true;
        }
        else
        {
            return false;
        };
    };

    auto particles =
        CabanaPD::createParticles<memory_space, model_type, thermal_type>(
            exec_space(), low_corner, high_corner, num_cells, halo_width,
            init_op );

    auto rho = particles->sliceDensity();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto force_model = CabanaPD::createForceModel(
        model_type{}, mechanics_type{}, *particles, delta, K, G0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd =
        CabanaPD::createSolver<memory_space>( inputs, particles, force_model );

    // ====================================================
    //                  Impose field
    // ====================================================
    // Grip's velocity
    double v0 = inputs["grip_velocity"];

    // Create region for each grip.
    double l_grip_max = inputs["grip_max_length"];
    double l_grip_min = inputs["grip_min_length"];
    double m_slop = d1 / ( l_grip_max - l_grip_min );

    // Create BC last to ensure ghost particles are included.
    auto x = particles->sliceReferencePosition();
    auto u = particles->sliceDisplacement();
    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        // Right grip
        if ( high_corner[0] - x( pid, 0 ) < l_grip_min )
            u( pid, 1 ) = -v0 * t;
        else if ( high_corner[0] - x( pid, 0 ) < l_grip_max &&
                  x( pid, 1 ) - low_corner[1] >
                      m_slop *
                          ( ( high_corner[0] - x( pid, 0 ) ) - l_grip_min ) )
            u( pid, 1 ) = -v0 * t;

        // Left grip
        if ( x( pid, 0 ) - low_corner[0] < l_grip_min )
            u( pid, 1 ) = 0;
        else if ( x( pid, 0 ) - low_corner[0] < l_grip_max &&
                  high_corner[1] - x( pid, 1 ) >
                      m_slop *
                          ( ( x( pid, 0 ) - low_corner[0] ) - l_grip_min ) )
            u( pid, 1 ) = 0;
    };

    auto body_term =
        CabanaPD::createBodyTerm( disp_func, particles->size(), false );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( body_term );
    cabana_pd->run( body_term );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    vnotchedBeamTestExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
