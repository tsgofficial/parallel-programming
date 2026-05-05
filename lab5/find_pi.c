#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long total_points = (argc > 1) ? atoll(argv[1]) : 10000000LL;
    long long local_points = total_points / size;
    if (rank == size - 1)
    {
        local_points += total_points % size;
    }

    unsigned int seed = (unsigned int)time(NULL) + rank;

    long long local_inside = 0;
    for (long long i = 0; i < local_points; i++)
    {
        double x = (double)rand_r(&seed) / (double)RAND_MAX;
        double y = (double)rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0)
            local_inside++;
    }

    double t0 = MPI_Wtime();
    long long global_inside = 0;
    if (rank == 0)
    {
        global_inside = local_inside;
        for (int src = 1; src < size; src++)
        {
            long long recv_inside = 0;
            MPI_Recv(&recv_inside, 1, MPI_LONG_LONG, src, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global_inside += recv_inside;
        }
    }
    else
    {
        MPI_Send(&local_inside, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    double t1 = MPI_Wtime();

    printf("[rank %d/%d] points=%lld inside=%lld\n",
           rank, size, local_points, local_inside);

    if (rank == 0)
    {
        double pi = 4.0 * (double)global_inside / (double)total_points;
        printf("\n──────────────────────────────────────────────\n");
        printf(" Total points : %lld\n", total_points);
        printf(" Inside       : %lld\n", global_inside);
        printf(" π estimate   : %.10f\n", pi);
        printf(" Reduce time  : %.6f s\n", t1 - t0);
        printf("──────────────────────────────────────────────\n");
    }

    MPI_Finalize();
    return 0;
}
