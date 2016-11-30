/**
 * Finalization step - write results and other computational vectors to files
 *
 * @date 22-Oct-2012
 * @author V. Petkov
 */

#include <stdio.h>
#include <stdlib.h>
#include "util_write_files.h"
#include "mpi.h"

void finalization(char* file_in, int nprocs, int myrank, int total_iters, double residual_ratio,
                  int nintci, int nintcf, double* var, int* local_global_index) {

    char file_out[100];

    MPI_Status status;
    int i;
    int nintcf_tot; // global number of cells

    // reduce global number of cells to all processes
    MPI_Allreduce(&nintcf, &nintcf_tot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	if (myrank == 0) // process 0 gathers data from all processes and prints them
	{
		int nintcf_other; // number of cell in examined process
		nintcf_tot += nprocs - 2; // correct value of total cells
		// allocate arrays to hold global var values
	    double *var_all = (double *) malloc((nintcf_tot + 2) * sizeof(double));
	    double *var_temp = (double *) malloc((nintcf_tot + 2) * sizeof(double));

	    for (i = 0; i < nintcf + 1; ++i)
	    	var_all[local_global_index[i]] = var[i]; // copy own local values to the global var

	    for (i = 1; i < nprocs; ++i) // traverse the other processes
	    {
	    	MPI_Recv(&nintcf_other, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status); // receive number of cells

	    	// allocate local_global_index for examined process
	    	int *l_g_i_other = (int *) malloc((nintcf_other + 1) * sizeof(int));

	    	MPI_Recv(var_temp, nintcf_other + 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status); // receive var

	    	MPI_Recv(l_g_i_other, nintcf_other + 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status); // receive l_g_index

	    	int j;

	    	for (j = 0; j < nintcf_other + 1; ++j)
	    		var_all[l_g_i_other[j]] = var_temp[j]; // copy received var values to global var

	    	free (l_g_i_other); // free local_global_index
	    }

	    sprintf(file_out, "%s_summary.out", file_in);

	    // print global values with global indexing
	    int status = store_simulation_stats(file_in, file_out, nintci, nintcf_tot, var_all, total_iters, residual_ratio);

	    if ( status != 0 ) fprintf(stderr, "Error when trying to write to file %s\n", file_out);

	    // free temporary arrays
	    free(var_all);
	    free(var_temp);
	}

	else
	{
		MPI_Send(&nintcf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(var, nintcf + 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
		MPI_Send(local_global_index, nintcf + 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
	}

	sprintf(file_out, "%s_rank%d_summary.out", file_in, myrank);

	// every process also prints its own values with local indexing
    int status_store = store_simulation_stats(file_in, file_out, nintci, nintcf, var, total_iters, residual_ratio);

    if ( status_store != 0 ) fprintf(stderr, "Error when trying to write to file %s\n", file_out);
}