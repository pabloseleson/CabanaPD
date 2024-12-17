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

// Simulate a cylinder under hot isostatic pressing (HIP)
void hipCylinderExample( const std::string filename )
{
    // ====================================================
    //             Use default Kokkos spaces
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
    double delta = inputs["horizon"];
    delta += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    // FIXME: set halo width based on delta
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Elastic>;
    model_type force_model( delta, K );

    // ====================================================
    //                 Particle generation
    // ====================================================
    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );
    double Rout = inputs["cylinder_outer_radius"];
    double Rin = inputs["cylinder_inner_radius"];

    // Do not create particles outside given cylindrical region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = ( x[0] - x_center ) * ( x[0] - x_center ) +
                     ( x[1] - y_center ) * ( x[1] - y_center );
        if ( rsq < Rin * Rin || rsq > Rout * Rout )
            return false;
        return true;
    };
    particles->createParticles( exec_space(), init_op );

    auto rho = particles->sliceDensity();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
        inputs, particles, force_model );

    // ====================================================
    //                   Imposed field
    // ====================================================
    auto x = particles->sliceReferencePosition();
    auto f = particles->sliceForce();
    double dx = particles->dx[0];
    double dz = particles->dx[2];
    double Pmax = inputs["maximum_pressure"];
    double ramp_t0 = inputs["ramp_times"][0];
    double ramp_t1 = inputs["ramp_times"][1];
    double ramp_t2 = inputs["ramp_times"][2];
    double ramp_t3 = inputs["ramp_times"][3];
    double top = high_corner[2];
    double bottom = low_corner[2];

    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto force_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        double sigma0 = 0;
        if ( t >= ramp_t0 )
        {
            if ( t <= ramp_t1 )
            {
                sigma0 = Pmax * ( t - ramp_t0 ) / ( ramp_t1 - ramp_t0 );
            }
            else if ( t <= ramp_t2 )
            {
                sigma0 = Pmax;
            }
            else if ( t <= ramp_t3 )
            {
                sigma0 = Pmax - Pmax * ( t - ramp_t2 ) / ( ramp_t3 - ramp_t2 );
            }
            else
            {
                sigma0 = 0;
            }
        }
        double bz = sigma0 / dz;
        double br = sigma0 / dx; // Here we assume dx = dy

        // Pressure on top surface
        if ( x( pid, 2 ) > top - dz )
        {
            f( pid, 2 ) += -bz;
        }
        // Pressure on bottom surface
        else if ( x( pid, 2 ) < bottom + dz )
        {
            f( pid, 2 ) += bz;
        }

        double rx = x( pid, 0 ) - x_center;
        double ry = x( pid, 1 ) - y_center;
        double rsq = rx * rx + ry * ry;

        // Pressure on outer cylinder: assumes dx = dy
        if ( rsq > ( Rout - dx ) * ( Rout - dx ) )
        {
            double r = std::sqrt( rsq );
            f( pid, 0 ) += -br * rx / r;
            f( pid, 1 ) += -br * ry / r;
        }
        // Pressure on inner cylinder: assumes dx = dy
        else if ( rsq < ( Rin + dx ) * ( Rin + dx ) )
        {
            double r = std::sqrt( rsq );
            f( pid, 0 ) += br * rx / r;
            f( pid, 1 ) += br * ry / r;
        }
    };
    auto body_term = CabanaPD::createBodyTerm( force_func, true );

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

    hipCylinderExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
