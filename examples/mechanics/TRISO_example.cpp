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

#include <Kokkos_Random.hpp>

// Tristructural-isotropic (TRISO) particle.
void TRISOParticleExample( const std::string filename )
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
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel force_model( model_type{}, delta, K, G0 );

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double kernel_Rout = inputs["kernel_outer_radius"];
    double buffer_Rout = inputs["buffer_outer_radius"];
    double IPyC_Rout = inputs["IPyC_outer_radius"];
    double SiC_Rout = inputs["SiC_outer_radius"];
    double OPyC_Rout = inputs["OPyC_outer_radius"];

    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );
    double z_center = 0.5 * ( low_corner[2] + high_corner[2] );

    // std::size_t seed = 44758454;
    //  Random number generator
    // std::mt19937 gen( seed );
    // std::uniform_real_distribution<> dis( 0.0, 1.0 );

    std::size_t seed = 44758454;                        // Random seed
    Kokkos::Random_XorShift64_Pool<> rand_pool( seed ); // Create a random pool

    // Do not create particles outside TRISO fuel region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Create a random generator specific to this thread
        auto rand_gen = rand_pool.get_state();

        // Generate a random perturbation in the range [0.0, 1.0]
        double perturbation = 0.1 * rand_gen.drand( 0.0, 1.0 );

        // Random numbers for pre-notch position
        // double perturbation = dis( gen );

        double rsq = ( x[0] - x_center ) * ( x[0] - x_center ) +
                     ( x[1] - y_center ) * ( x[1] - y_center ) +
                     ( x[2] - z_center ) * ( x[2] - z_center );
        // if ( rsq > OPyC_Rout * OPyC_Rout )
        if ( rsq > ( OPyC_Rout * ( 1.0 - perturbation ) ) *
                       ( OPyC_Rout * ( 1.0 - perturbation ) ) )
            return false;
        return true;
    };

    // ====================================================
    //                 Particle generation
    // ====================================================
    CabanaPD::Particles particles(
        memory_space{}, model_type{}, low_corner, high_corner, num_cells,
        halo_width, Cabana::InitRandom{}, init_op, exec_space{} );

    /*
    CabanaPD::Particles particles(
        memory_space{}, model_type{}, low_corner, high_corner, num_cells,
        halo_width, Cabana::InitUniform{}, init_op, exec_space{} );
    */
    // CabanaPD::Particles particles( memory_space{}, model_type{}, inputs,
    //                               exec_space{} );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto type = particles.sliceType();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        double rsq = ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center ) +
                     ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center ) +
                     ( x( pid, 2 ) - z_center ) * ( x( pid, 2 ) - z_center );

        if ( rsq < kernel_Rout * kernel_Rout )
        {
            type( pid ) = 0;
        }
        else if ( rsq < buffer_Rout * buffer_Rout )
        {
            type( pid ) = 1;
        }
        else if ( rsq < IPyC_Rout * IPyC_Rout )
        {
            type( pid ) = 2;
        }
        else if ( rsq < SiC_Rout * SiC_Rout )
        {
            type( pid ) = 3;
        }
        else
        {
            type( pid ) = 2;
        }
    };
    particles.updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init();
    solver.run();
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    TRISOParticleExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
