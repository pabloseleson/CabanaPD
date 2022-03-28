#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD_Output.hpp>
#include <CabanaPD_Solver.hpp>

int main( int argc, char* argv[] )
{

    MPI_Init( &argc, &argv );

    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        // FIXME: change backend at compile time for now.
        using memory_space = Kokkos::HostSpace;
        using exec_space = Kokkos::Serial;

        int num_cell = 41;
        double low_corner = -0.5;
        double high_corner = 0.5;
        double t_final = 0.6;
        double dt = 0.01;
        double K = 1.0;
        double delta = 0.05;
        // FIXME: set halo width based on delta
        int halo_width = 2;

        CabanaPD::Inputs inputs( num_cell, low_corner, high_corner, K, delta,
                                 t_final, dt );
        inputs.read_args( argc, argv );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<CabanaPD::Particles<memory_space>>(
            exec_space(), inputs.low_corner, inputs.high_corner,
            inputs.num_cells, halo_width );

        // Define particle initialization.
        auto x = particles->slice_x();
        auto u = particles->slice_u();
        auto v = particles->slice_v();
        auto rho = particles->slice_rho();

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            double a = 0.001;
            double r0 = 0.25;
            double l = 0.07;
            double norm = std::sqrt( x( pid, 0 ) * x( pid, 0 ) +
                                     x( pid, 1 ) * x( pid, 1 ) +
                                     x( pid, 2 ) * x( pid, 2 ) );
            double diff = norm - r0;
            double arg = diff * diff / l / l;
            for ( int d = 0; d < 3; d++ )
            {
                double comp = 0.0;
                if ( norm > 0.0 )
                    comp = x( pid, d ) / norm;
                u( pid, d ) = a * std::exp( -arg ) * comp;
                v( pid, d ) = 0.0;
            }
            rho( pid ) = 100;
        };
        particles->update_particles( exec_space{}, init_functor );

        // FIXME: use createSolver to switch backend at runtime.
        auto cabana_pd = std::make_shared<
            CabanaPD::Solver<Kokkos::Device<exec_space, memory_space>>>(
            inputs, particles );
        cabana_pd->run();

        x = particles->slice_x();
        u = particles->slice_u();
        auto profile = Kokkos::View<double* [2], memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
            num_cell );
        double length = ( high_corner - low_corner );
        int mpi_rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        int count = 0;
        auto measure_profile = KOKKOS_LAMBDA( const int pid, int& c )
        {
            double dx = length / num_cell;
            if ( x( pid, 1 ) < dx / 2.0 && x( pid, 1 ) > -dx / 2.0 &&
                 x( pid, 2 ) < dx / 2.0 && x( pid, 2 ) > -dx / 2.0 )
            {
                profile( c, 0 ) = x( pid, 0 );
                profile( c, 1 ) = u( pid, 0 );
                c++;
            }
        };
        Kokkos::RangePolicy<exec_space> policy( 0, x.size() );
        Kokkos::parallel_reduce( "displacement_profile", policy,
                                 measure_profile, count );

        auto profile_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profile );
        std::fstream fout;
        std::string file_name = "displacement_profile.txt";
        fout.open( file_name, std::ios::app );
        for ( int p = 0; p < count; p++ )
        {
            fout << mpi_rank << " " << profile_host( p, 0 ) << " "
                 << profile_host( p, 1 ) << std::endl;
        }
    }

    MPI_Finalize();
}
