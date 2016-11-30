/**
 * Initialization step - parse the input file, compute data distribution, initialize LOCAL computational arrays
 *
 * @date 22-Oct-2012, 03-Nov-2014
 * @author V. Petkov, A. Berariu
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mpi.h"
#include "metis.h"
#include "papi.h"
#include "util_read_files.h"
#include "initialization.h"
#include "test_functions.h"
#include "helper.h"

int initialization(char* file_in, char* part_type, char* read_type, int nprocs, int myrank,
                   int* nintci, int* nintcf, int* nextci,
                   int* nextcf, int*** lcc, double** bs, double** be, double** bn, double** bw,
                   double** bl, double** bh, double** bp, double** su, int* points_count,
                   int*** points, int** elems, double** var, double** cgup, double** oc,
                   double** cnorm, int** local_global_index, int** global_local_index,
                   int *nghb_cnt, int** nghb_to_rank, int** send_cnt, int*** send_lst, 
                   int **recv_cnt, int*** recv_lst) {
    /********** START INITIALIZATION **********/
    int i = 0;
    // allocate variables to hold global values of data in input file
    int **lcc_global;
    double *bs_global, *be_global, *bn_global, *bw_global, *bl_global, *bh_global, *bp_global, *su_global;
    int nintci_global, nintcf_global, nextci_global, nextcf_global;

    int *elems_procs; // number of internal elements per process
    idx_t *epart; // array assigning each element to a process (e.g. epart[1]=3 means process number 3 has element 1)
    int num_elm; // global internal elements - helper variable
    int ext_cells; // number of external cells for each process
    int ghost_cells; // number of ghost cells for each process (globally internal, belonging to another process)

    // ********* START INITIALIZATION ********//
    // Kill program if asked to run METIS with less than 2 processes
    if ( nprocs < 2 && strcmp( part_type, "classic") ) 
    {
        printf("Too few processes for METIS\n");
        return -1;
    }

    long long start_usec, end_usec;

    start_usec = PAPI_get_real_usec(); // start measuring time

    // ALL READ

    if ( !strcmp( read_type, "allread") )
    {
        // all processes read the input file
        int r_status = read_partition(file_in, &nintci_global, &nintcf_global, &nextci_global, &nextcf_global, &lcc_global, &bs_global,
                                       &be_global, &bn_global, &bw_global, &bl_global, &bh_global, &bp_global, &su_global,
                                       &*points_count, &*points, &*elems, nprocs, &elems_procs, &epart, part_type);

        if ( r_status != 0 ) return r_status;

        num_elm = nintcf_global - nintci_global + 1;
    }

    // ONE READ

    else
    {
        // allocate batch of data to be broadcasted
        int *firstbatch = (int *) malloc((5 + nprocs) * sizeof(int));

        if (myrank == 0)
        {
            int i;
            // one process reads the input file
            int r_status = read_partition(file_in, &nintci_global, &nintcf_global, &nextci_global, &nextcf_global, &lcc_global, &bs_global,
                                       &be_global, &bn_global, &bw_global, &bl_global, &bh_global, &bp_global, &su_global,
                                       &*points_count, &*points, &*elems, nprocs, &elems_procs, &epart, part_type);

            if ( r_status != 0 ) return r_status;

            firstbatch[0] = nintci_global;
            firstbatch[1] = nintcf_global;
            firstbatch[2] = nextci_global;
            firstbatch[3] = nextcf_global;
            firstbatch[4] = *points_count;
            
            for ( i = 0; i < nprocs; ++i)
                firstbatch[i + 5] = elems_procs[i];
        }

        // broadcast global data to be used for allocation of arrays
        MPI_Bcast(firstbatch, nprocs + 5, MPI_INT, 0, MPI_COMM_WORLD);

        if (myrank != 0)
        {
            allocate_elems_procs(&elems_procs, nprocs);
         
            nintci_global = firstbatch[0];
            nintcf_global = firstbatch[1];
            nextci_global = firstbatch[2];
            nextcf_global = firstbatch[3];
            *points_count = firstbatch[4];
            
            for ( i = 0; i < nprocs; ++i)
                elems_procs[i] = firstbatch[i + 5];
        
            // allocate epart
            if ( ((epart) = (idx_t *) malloc((nintcf_global - nintci_global + 1) * sizeof(idx_t))) == NULL) {
                printf("malloc(epart) failed\n");
                return -1;
            }

            // allocate LCC
            if ( (lcc_global = (int**) malloc((nintcf_global + 1) * sizeof(int*))) == NULL ) {
                fprintf(stderr, "malloc failed to allocate first dimension of LCC");
                return -1;
            }

            for ( i = 0; i < nintcf_global + 1; i++ ) {
                if ( (lcc_global[i] = (int *) malloc(6 * sizeof(int))) == NULL ) {
                    fprintf(stderr, "malloc failed to allocate second dimension of LCC\n");
                    return -1;
                }
            }

            // allocate other arrays
            int a_status = allocate_2(&bs_global, &be_global, &bn_global, &bw_global, &bl_global, &bh_global, &bp_global, &su_global, &nextcf_global);

            if ( a_status != 0 ) return a_status;

            // allocate points matrix and elems vector to avoid BAD TERMINATION
            if ( (*points = (int **) malloc(*points_count * sizeof(int*))) == NULL ) {
                fprintf(stderr, "malloc() POINTS 1st dim. failed\n");
                return -1;
            }

            for ( i = 0; i < *points_count; i++ ) {
                if ( ((*points)[i] = (int *) calloc(3, sizeof(int))) == NULL ) {
                    fprintf(stderr, "malloc() POINTS 2nd dim. failed\n");
                    return -1;
                }
            }

            if ( (*elems = (int *) malloc(1 * sizeof(int))) == NULL ) {
                fprintf(stderr, "malloc(elems) failed\n");
                return -1;
            }
        }

        free(firstbatch);

        num_elm = nintcf_global - nintci_global + 1;

        // allocate batch of data to be broadcasted - secondbatch contains lcc matrix, thirdbatch contains boundary vectors
        int *secondbatch = (int *) malloc((6 * num_elm) * sizeof(int));
        double *thirdbatch = (double *) malloc((8 * (nextcf_global + 1)) * sizeof(double));

        if (myrank == 0)
        {
            int i;

            for (i = 0; i < num_elm * 6; ++i)
                secondbatch[i] = lcc_global[i / 6][i % 6];

            // copy boundary vectors to thirdbatch (to be broadcasted) sequentially
            memcpy(thirdbatch, bs_global, (nextcf_global + 1) * sizeof(double));
            memcpy(thirdbatch + (nextcf_global + 1), be_global, (nextcf_global + 1) * sizeof(double));
            memcpy(thirdbatch + 2 * (nextcf_global + 1), bn_global, (nextcf_global + 1) * sizeof(double));
            memcpy(thirdbatch + 3 * (nextcf_global + 1), bw_global, (nextcf_global + 1) * sizeof(double));
            memcpy(thirdbatch + 4 * (nextcf_global + 1), bl_global, (nextcf_global + 1) * sizeof(double));
            memcpy(thirdbatch + 5 * (nextcf_global + 1), bh_global, (nextcf_global + 1) * sizeof(double));
            memcpy(thirdbatch + 6 * (nextcf_global + 1), bp_global, (nextcf_global + 1) * sizeof(double));
            memcpy(thirdbatch + 7 * (nextcf_global + 1), su_global, (nextcf_global + 1) * sizeof(double));
        }

        // broadcast epart, global lcc and boundary values
        MPI_Bcast(&*epart, num_elm, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(secondbatch, num_elm * 6, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(thirdbatch, (nextcf_global + 1) * 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (myrank != 0)
        {
            int i;

            for (i = 0; i < num_elm * 6; ++i)
                lcc_global[i / 6][i % 6] = secondbatch[i];

            // copy (received) thirdbatch to boundary vectors sequentially
            memcpy(bs_global, thirdbatch, (nextcf_global + 1) * sizeof(double));
            memcpy(be_global, thirdbatch + (nextcf_global + 1), (nextcf_global + 1) * sizeof(double));
            memcpy(bn_global, thirdbatch + 2 * (nextcf_global + 1), (nextcf_global + 1) * sizeof(double));
            memcpy(bw_global, thirdbatch + 3 * (nextcf_global + 1), (nextcf_global + 1) * sizeof(double));
            memcpy(bl_global, thirdbatch + 4 * (nextcf_global + 1), (nextcf_global + 1) * sizeof(double));
            memcpy(bh_global, thirdbatch + 5 * (nextcf_global + 1), (nextcf_global + 1) * sizeof(double));
            memcpy(bp_global, thirdbatch + 6 * (nextcf_global + 1), (nextcf_global + 1) * sizeof(double));
            memcpy(su_global, thirdbatch + 7 * (nextcf_global + 1), (nextcf_global + 1) * sizeof(double));
        }

        free(secondbatch);
        free(thirdbatch);
    }

    end_usec = PAPI_get_real_usec(); // end measuring time

    printf("Rank %d: Read and partition time: %lld\n", myrank, end_usec - start_usec); // print time interval

    // allocate space for local lcc and copy global values to it; also count number of external cells for each process
    int a_status = allocate_1(&elems_procs, myrank, local_global_index, global_local_index, &*lcc, &*nintci, &*nintcf, &*nextci, &nintcf_global);

    if ( a_status != 0 ) return a_status;

    int t_status = transform_1(&*lcc, &lcc_global, myrank, epart, local_global_index, global_local_index, num_elm, &nextci_global, &nextcf_global, &ext_cells);

    if ( t_status != 0 ) return t_status;

    /* count number of ghost cells for each process and change values of the local lcc matrix to correspond to the local
     indexing; local indexing follows the pattern: internal cells - external cells - ghost cells, in an ascending order.
     also count the number of neighbours and construct send and receive lists of cells whose data values need to be
     exchanged during the computation phase */
    int l_status = localise_lcc(elems_procs, &*lcc, &*nintcf, &nextci_global, &nextcf_global, &**local_global_index, &**global_local_index, myrank, &ext_cells, &ghost_cells,
                 &*nghb_cnt, &*nghb_to_rank, &*send_cnt, &*send_lst, &*recv_cnt, &*recv_lst, epart, nprocs);

    if ( l_status != 0 ) return l_status;

    // assign the correct value to the total number of cells for each process
    *nextcf = *nintcf + ext_cells + ghost_cells;

    // allocate space for local boundary values and copy global values to them
    a_status = allocate_2(&*bs, &*be, &*bn, &*bw, &*bl, &*bh, &*bp, &*su, &*nextcf);

    if ( a_status != 0 ) return a_status;

    num_elm = nintcf_global - nintci_global + 1;

    t_status = transform_2(&*bs, &*be, &*bn, &*bw, &*bl, &*bh, &*bp, &*su, &bs_global, &be_global, &bn_global, &bw_global, &bl_global, &bh_global,
                           &bp_global, &su_global, myrank, epart, num_elm, *global_local_index);

    if ( t_status != 0 ) return t_status;

    // allocate space for the var, cgup and cnorm arrays
    *var = (double*) calloc(sizeof(double), (*nextcf + 1));
    *cgup = (double*) calloc(sizeof(double), (*nextcf + 1));
    *cnorm = (double*) calloc(sizeof(double), (*nintcf + 1));

    // initialize the arrays
    for ( i = 0; i <= 10; i++ ) {
        (*cnorm)[i] = 1.0;
    }

    for ( i = (*nintci); i <= (*nintcf); i++ ) {
        (*var)[i] = 0.0;
    }

    for ( i = (*nintci); i <= (*nintcf); i++ ) {
        (*cgup)[i] = 1.0 / ((*bp)[i]);
    }

    for ( i = (*nextci); i <= (*nextcf); i++ ) {
        (*var)[i] = 0.0;
        (*cgup)[i] = 0.0;
        (*bs)[i] = 0.0;
        (*be)[i] = 0.0;
        (*bn)[i] = 0.0;
        (*bw)[i] = 0.0;
        (*bh)[i] = 0.0;
        (*bl)[i] = 0.0;
    }

    // free not needed arrays
    for ( i = 0; i < nintcf_global + 1; i++ ) {
        free(lcc_global[i]);
    }
    free(lcc_global);
    free(bs_global);
    free(be_global);
    free(bn_global);
    free(bw_global);
    free(bl_global);
    free(bh_global);
    free(bp_global);
    free(su_global);
    free(epart);
    free(elems_procs);

    return 0;
}
