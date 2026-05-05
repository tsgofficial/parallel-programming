/*
 * 1D heat diffusion on a metal rod, parallelised with MPI.
 *
 * Build : mpicc -O2 -o heat_diffusion heat_diffusion.c -lm
 * Run   : mpirun -n 4 ./heat_diffusion 1000
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N_TOTAL 4096
#define C_COEFF 0.25
#define T_LEFT 100.0
#define T_RIGHT 0.0

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);               // MPI_Init: start the MPI runtime
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // MPI_Comm_rank: this process's ID
    MPI_Comm_size(MPI_COMM_WORLD, &size); // MPI_Comm_size: total processes

    int time_steps = (argc > 1) ? atoi(argv[1]) : 1000;

    if (rank == 0)
    {
        printf("1D Heat Diffusion — MPI\n");
        printf("  total points : %d\n", N_TOTAL);
        printf("  processes    : %d\n", size);
        printf("  points / proc: %d\n", N_TOTAL / size);
        printf("  time steps   : %d\n", time_steps);
        printf("  C coefficient: %.2f\n", C_COEFF);
        printf("  left  BC     : %.1f\n", T_LEFT);
        printf("  right BC     : %.1f\n\n", T_RIGHT);
    }

    if (N_TOTAL % size != 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "N (%d) is not divisible by size (%d)\n", N_TOTAL, size);
        }
        MPI_Finalize();
        return 1;
    }

    int local_n = N_TOTAL / size;
    int buf_len = local_n + 2; // +2 for ghost cells at index 0 and local_n+1

    double *current = calloc(buf_len, sizeof(double));
    double *next = calloc(buf_len, sizeof(double));

    if (!current || !next)
    {
        fprintf(stderr, "[rank %d] allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1); // MPI_Abort: kill every process at once
    }

    if (rank == 0)
    {
        current[1] = T_LEFT;
    }

    int left = rank - 1;
    int right = rank + 1;
    int has_left = (left >= 0);
    int has_right = (right < size);

    double t_start = MPI_Wtime(); // MPI_Wtime: wall-clock seconds

    for (int t = 0; t < time_steps; t++)
    {
        MPI_Status status;

        // MPI_Sendrecv: send one message and receive another in a single call.
        // Doing it this way avoids the deadlock you can hit if every process
        // calls MPI_Send before MPI_Recv.
        //
        // Exchange with the right neighbour: send our last real cell, receive
        // its first real cell into our right ghost.
        if (has_right)
        {
            MPI_Sendrecv(
                &current[local_n], 1, MPI_DOUBLE, right, 0,
                &current[local_n + 1], 1, MPI_DOUBLE, right, 1,
                MPI_COMM_WORLD, &status);
        }
        else
        {
            current[local_n + 1] = T_RIGHT;
        }

        // Exchange with the left neighbour (mirror of the above).
        if (has_left)
        {
            MPI_Sendrecv(
                &current[1], 1, MPI_DOUBLE, left, 1,
                &current[0], 1, MPI_DOUBLE, left, 0,
                MPI_COMM_WORLD, &status);
        }
        else
        {
            current[0] = T_LEFT;
        }

        // Finite-difference update: next = cur + C * (left - 2*cur + right).
        // Rank 0's first cell is the fixed left BC, so it starts at index 2.
        int i_start = (rank == 0) ? 2 : 1;
        for (int i = i_start; i <= local_n; i++)
        {
            next[i] = current[i] + C_COEFF *
                                       (current[i - 1] - 2.0 * current[i] + current[i + 1]);
        }
        if (rank == 0)
            next[1] = T_LEFT;

        double *tmp = current;
        current = next;
        next = tmp;

        if (rank == 0 && (t + 1) % 100 == 0)
        {
            printf("  [t=%5d/%d]  temp at pos 10: %7.4f   pos 50: %7.4f\n",
                   t + 1, time_steps, current[11], current[51]);
        }
    }

    double t_end = MPI_Wtime();

    double *gathered = NULL;
    if (rank == 0)
    {
        gathered = malloc(N_TOTAL * sizeof(double));
        if (!gathered)
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // MPI_Gather: every process sends its chunk to the root (rank 0), which
    // concatenates them in rank order into one array of N_TOTAL doubles.
    MPI_Gather(
        &current[1], local_n, MPI_DOUBLE,
        gathered, local_n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        char filename[64];
        snprintf(filename, sizeof(filename), "output_%d_steps.csv", time_steps);

        FILE *fp = fopen(filename, "w");
        if (!fp)
        {
            perror("fopen");
            free(gathered);
            MPI_Finalize();
            return 1;
        }

        fprintf(fp, "position,temperature\n");
        for (int i = 0; i < N_TOTAL; i++)
        {
            fprintf(fp, "%d,%.6f\n", i, gathered[i]);
        }
        fclose(fp);

        int front = 0;
        for (int i = N_TOTAL - 1; i >= 0; i--)
        {
            if (gathered[i] > 0.001)
            {
                front = i;
                break;
            }
        }

        printf("\nDone.\n");
        printf("  output file       : %s\n", filename);
        printf("  elapsed time      : %.4f s\n", t_end - t_start);
        printf("  first point (BC)  : %.4f\n", gathered[0]);
        printf("  heat front pos    : %d  (last cell with temp > 0.001)\n", front);
        printf("  temp at pos 10    : %.4f\n", gathered[10]);
        printf("  temp at pos 50    : %.4f\n", gathered[50]);
        printf("  middle point      : %.4f\n", gathered[N_TOTAL / 2]);
        printf("  last point  (BC)  : %.4f\n", gathered[N_TOTAL - 1]);
        printf("\n  plot with: python3 plot_results.py %s\n\n", filename);

        free(gathered);
    }

    free(current);
    free(next);

    MPI_Finalize(); // MPI_Finalize: shut MPI down (last MPI call)
    return 0;
}
