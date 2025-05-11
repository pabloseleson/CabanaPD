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

#include <cmath>
#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

// std::vector<double> readCoefficients(const std::string& filename);
// double evaluatePolynomial(const std::vector<double>& coeffs, double t);

// Simulate a tungsten block exposed to a heat pulse within a central square
// region on its top surface.
void wirtzCaseExample( const std::string filename )
{
    std::vector<double> readCoefficients( const std::string& filename );
    double evaluatePolynomial( const std::vector<double>& coeffs, double t );

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
    //            Material and problem parameters
    // ====================================================
    // Material parameters
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25;
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    double alpha = inputs["thermal_expansion_coeff"];

    // Problem parameters
    double temp0 = inputs["reference_temperature"];
    double t_ramp = inputs["surface_temperature_ramp_time"];

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
    using thermal_type = CabanaPD::TemperatureDependent;

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    CabanaPD::Particles particles( memory_space{}, model_type{}, thermal_type{},
                                   low_corner, high_corner, num_cells,
                                   halo_width, exec_space{} );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto init_functor = KOKKOS_LAMBDA( const int pid ) { rho( pid ) = rho0; };
    particles.updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto temp = particles.sliceTemperature();
    CabanaPD::ForceModel force_model( model_type{}, delta, K, G0, temp, alpha,
                                      temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                   Impose field
    // ====================================================
    auto x = solver.particles.sliceReferencePosition();
    temp = solver.particles.sliceTemperature();
    double dz = solver.particles.dx[2];

    // std::vector<double> coeffs;

    std::vector<double> coeffs_left =
        readCoefficients( "poly_coeffs_left.txt" );
    std::vector<double> coeffs_right =
        readCoefficients( "poly_coeffs_right.txt" );

    // ==========================
    /*
    // Test function
    std::vector<double> test_points = {0.0, 1e-7, 5e-7, 1e-6, 5e-5, 1e-5, 5e-5,
    1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1};

    for (double t : test_points) {
        const std::vector<double>& coeffs = (t <= t_ramp) ? coeffs_left :
    coeffs_right; double t_log = log10(t + 1e-16); double result =
    evaluatePolynomial(coeffs, t_log); result = pow(10.0, result); if (result <
    22) result = 22; std::cout << "Polynomial evaluated at t = " << t << ": " <<
    result << std::endl;
    }
        */
    // ==========================

    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        /*
        if ( t <= t_ramp )
        {
           // std::vector<double> coeffs =
        readCoefficients("poly_coeffs_left.txt");
        }
        else
        {
           // std::vector<double> coeffs =
        readCoefficients("poly_coeffs_right.txt");
        }
           */

        const std::vector<double>& coeffs =
            ( t <= t_ramp ) ? coeffs_left : coeffs_right;

        // double temp_bc;

        // if( t > 0 )
        // {
        double t_log = log10( t + 1e-16 );
        double temp_bc = evaluatePolynomial( coeffs, t_log );
        temp_bc = pow( 10.0, temp_bc );
        // FIXME: use input for 22
        if ( temp_bc < 22 )
            temp_bc = 22;

        // }
        // else
        // {
        //     temp_bc = 100;
        // }

        /*
        for (double t : test_points) {
         const std::vector<double>& coeffs = (t <= t_ramp) ? coeffs_left :
     coeffs_right; double t_log = log10(t + 1e-16); double result =
     evaluatePolynomial(coeffs, t_log); result = pow(10.0, result); if (result <
     22) result = 22; std::cout << "Polynomial evaluated at t = " << t << ": "
     << result << std::endl;
     }
         */

        // Assign particle temperature
        if ( x( pid, 0 ) > -0.002 && x( pid, 0 ) < 0.002 &&
             x( pid, 1 ) > -0.002 && x( pid, 1 ) < 0.002 &&
             x( pid, 2 ) > high_corner[2] - dz )
        { // temp( pid ) = 1000*t; //temp_bc;
            temp( pid ) = temp_bc;
            //    std::cout << "pid: " << pid << ", t: " << t << ", temp_bc: "
            //    << temp( pid ) << std::endl;
        }
    };
    CabanaPD::BodyTerm body_term( temp_func, solver.particles.size(), false );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( body_term );
    solver.run( body_term );
}

// Read polynomial coefficients
std::vector<double> readCoefficients( const std::string& filename )
{
    std::ifstream infile( filename );
    std::vector<double> coeffs;
    std::string line;
    while ( std::getline( infile, line ) )
    {
        coeffs.push_back( std::stod( line ) );
    }
    return coeffs;
}

// Evaluate polynomial using coefficients:
// the cefficients are ordered beginning with the one corresponding to the
// largest power.
double evaluatePolynomial( const std::vector<double>& coeffs, double t )
{
    double poly = 0.0;
    int Np = coeffs.size();
    for ( int i = 0; i < Np; ++i )
    {
        poly += coeffs[i] * std::pow( t, Np - 1 - i );
    }
    return poly;
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    wirtzCaseExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
