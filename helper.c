/* Helper functions for initialisation */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "metis.h"
#include "helper.h"
#include "util_read_files.h"
#include "mpi.h"

int read_partition(char *file_name, int *NINTCI, int *NINTCF, int *NEXTCI, int *NEXTCF, int ***LCC,
                    double **BS, double **BE, double **BN, double **BW, double **BL, double **BH,
                    double **BP, double **SU, int* points_count, int*** points, int** elems, int nprocs,
                    int **elems_procs, idx_t **epart, char *part_type)
{
    int i;
    int f_status = read_binary_geo(file_name, NINTCI, NINTCF, NEXTCI, NEXTCF, LCC, BS,
                                   BE, BN, BW, BL, BH, BP, SU,
                                   &*points_count, &*points, &*elems);

    if ( f_status != 0 ) return f_status;

    // METIS variables
    idx_t ne = *NINTCF - *NINTCI + 1; // total number of elements
    idx_t nn = *points_count; // total number of points
    idx_t *eptr, *eind; // arrays that define the domain - nodes of each element
    idx_t ncommon = 4; // number of common nodes between neighbouring elements - one rectangular side
    idx_t nparts = nprocs; // number of processes
    idx_t objval = 0; // metis output variable
    idx_t *npart; // metis output variable

    // element and nodal mappings allocations
    if ( ((*epart) = (idx_t *) malloc((ne) * sizeof(idx_t))) == NULL) {
        printf("malloc(epart) failed\n");
        return -1;
    }
    if ( (npart = (idx_t *) malloc((nn) * sizeof(idx_t))) == NULL) {
        printf("malloc(npart) failed\n");
        return -1;
    }

    if ( (eptr = (idx_t *) malloc((ne + 1) * sizeof(idx_t))) == NULL) {
        printf("malloc(eptr) failed\n");
        return -1;
    }

    if ( (eind = (idx_t *) malloc((ne * 8) * sizeof(idx_t))) == NULL) {
        printf("malloc(eind) failed\n");
        return -1;
    }

    // distribute elements to processes
    if ( (distribution(ne, nn, eind, eptr, ncommon, nparts, objval, &*epart, &npart, part_type, elems)) != 0) {
        printf("Distribution failed\n");
        return -1;
    }

    // allocate elems_procs array
    int e_status = allocate_elems_procs(&*elems_procs, nprocs);

    if ( e_status != 0 ) return e_status;

    // fill elems_procs array
    for (i = 0; i < ne; ++i)
        ((*elems_procs)[(*epart)[i]])++;

    free(npart);
    free(eptr);
    free(eind);

    return 0;
}

int distribution(idx_t ne, idx_t nn, idx_t* eind, idx_t* eptr, idx_t ncommon, idx_t nparts, idx_t objval, idx_t** epart, idx_t** npart, char* part_type, int** elems)
{
    int i = 0;

    // element and nodal mappings for metis/classic

    if ( !strcmp( part_type, "classic" ) )
        for (i = 0; i < ne; ++i)
            (*epart)[i] = (i * nparts / ne); // distribute cells to processes evenly and sequentially

    if ( !strcmp( part_type, "dual" ) )
    {
        for (i = 0; i <= ne; ++i)
            eptr[i] = 8 * i; // points to every 8 values - number of nodes on a cube element
        
        for (i = 0; i < ne * 8; ++i)
            eind[i] = (*elems)[i]; // contains index of the 8 nodes of each element

        // 'dual' metis distribution
        if ( (METIS_PartMeshDual(&ne, &nn, eptr, eind, NULL, NULL, &ncommon, &nparts, NULL, NULL, &objval, *epart, *npart)) != METIS_OK) {
            printf("Metis(dual) failed\n");
            return -1;
        }
    }

    if ( !strcmp( part_type, "nodal" ) )
    {
        for (i = 0; i <= ne; ++i)
            eptr[i] = 8 * i; // points to every 8 values - number of nodes on a cube element

        for (i = 0; i < ne * 8; ++i)
            eind[i] = (*elems)[i]; // contains index of the 8 nodes of each element

        // 'nodal' metis distribution
        if ( (METIS_PartMeshNodal(&ne, &nn, eptr, eind, NULL, NULL, &nparts, NULL, NULL, &objval, *epart, *npart)) != METIS_OK) {
            printf("Metis(nodal) failed\n");
            return -1;
        }
    }
    return 0;
}

int allocate_elems_procs(int **elems_procs, int nprocs)
{
    if ( ((*elems_procs) = (int *) calloc((nprocs), sizeof(int))) == NULL) {
        printf("malloc(elems_procs) failed\n");
        return -1;
    }
    return 0;
}

int allocate_1(int **elems_procs, int myrank, int **local_global_index, int **global_local_index, int ***LCC, int *nintci, int *nintcf, int *nextci, int *nintcf_global)
{
    int i;
    *nintci = 0;
    *nintcf = (*elems_procs)[myrank] - 1;
    *nextci = (*elems_procs)[myrank];

    // allocating local_global_index
    if ( (*local_global_index = (int *) malloc((*elems_procs)[myrank] * sizeof(int))) == NULL ) {
        fprintf(stderr, "malloc(local_global_index) failed\n");
        return -1;
    }

    // allocating local_global_index - contains all elements of the domain
    if ( (*global_local_index = (int *) malloc((*nintcf_global + 1) * sizeof(int))) == NULL ) {
        fprintf(stderr, "malloc(global_local_index) failed\n");
        return -1;
    }

    // every element of global_local_index is assigned to -1
    // (this will change for internal, external and ghost cells of process at hand later)
    for (i = 0; i < *nintcf_global + 1; ++i)
        (*global_local_index)[i] = -1;

    // allocate local lcc array
    if ( (*LCC = (int**) malloc((*nintcf + 1) * sizeof(int*))) == NULL ) {
        fprintf(stderr, "malloc failed to allocate first dimension of lcc_local");
        return -1;
    }

    for ( i = 0; i < *nintcf + 1; i++ ) {
        if ( ((*LCC)[i] = (int *) malloc(6 * sizeof(int))) == NULL ) {
            fprintf(stderr, "malloc failed to allocate second dimension of lcc_local\n");
            return -1;
        }
    }

    return 0;
}

int transform_1(int ***lcc, int ***lcc_global, int myrank, idx_t *epart, int **local_global_index, int **global_local_index, int ne, int *nextci, int *nextcf, int *num_ext_cells)
{
    // calculating local_global_index, global_local_index (internal cells) and local lcc arrays
    int i, j;
    int counter = 0;
    int *hash_table; // hash table to indicate whether external cell has been counted before

    if ( (hash_table = (int *) calloc((*nextcf) + 1, sizeof(int))) == NULL) {
        printf("malloc(hash_table) failed\n");
        return -1;
    }

    *num_ext_cells = 0;

    for ( i = 0; i < ne; ++i )
    {
        if ( myrank == epart[i] ) // cell belongs to process
        {
            for (j = 0; j < 6; ++j) // look at its 6 neighbours
            {
                int val = (*lcc_global)[i][j];
                (*lcc)[counter][j] = val; // copy indices of the 6 neighbours to corresponding position in local lcc

                if (val >= *nextci && !hash_table[val]) // if one neighbour is external and it hasn't been
                {                                       // counted before, increment counter of external cells
                    (*num_ext_cells)++;
                    hash_table[val] = *num_ext_cells;
                }
            }

            (*local_global_index)[counter] = i; // set local_global and global_local indices accordingly
            (*global_local_index)[i] = counter;
            counter++; // increment internal cell counter
        }
    }

    free(hash_table);

    return 0;
}

int cmpfunc (const void * a, const void * b) // function used by qsort, to sort integer elements
{
   return ( *(int*)a - *(int*)b );
}

int localise_lcc(int *elems_procs, int ***lcc, int *nintcf_local, int *nextci, int *nextcf, int *local_global_index,
                 int *global_local_index, int myrank, int *num_ext_cells, int *num_ghost_cells,
                 int *nghb_cnt, int** nghb_to_rank, int** send_cnt, int*** send_lst, 
                 int **recv_cnt, int*** recv_lst, idx_t *epart, int nprocs)
{
    int i, j;
    *nghb_cnt = 0;
    int *hash_table; // used to determine whether ghost or external cell has been counted before and to assign an index to it
    int *rank_to_nghb; // reverse of nghb_to_rank, gives rank of neighbour
    int *send_check; // array to check whether cell examined has been added to send list or not
    int **send_lst_temp, **recv_lst_temp; // temporary (bigger) arrays for send and receive counts and lists
    int *send_cnt_temp, *recv_cnt_temp;
    int *nghb_to_rank_temp; // temporary (bigger) array for neighbour to rank

    // allocate (temporary) arrays
    if ( (hash_table = (int *) calloc((*nextcf) + 1, sizeof(int))) == NULL) {
        printf("malloc(hash_table) failed\n");
        return -1;
    }

    if ( (nghb_to_rank_temp = (int*) calloc(nprocs - 1, sizeof(int))) == NULL ) {
        fprintf(stderr, "malloc() nghb_to_rank_temp failed\n");
        return -1;
    }

    if ( (rank_to_nghb = (int*) calloc(sizeof(int), nprocs)) == NULL ) {
        fprintf(stderr, "malloc() rank_to_nghb failed\n");
        return -1;
    }

    if ( (recv_cnt_temp = (int*) calloc(sizeof(int), nprocs - 1)) == NULL ) {
        fprintf(stderr, "malloc() recv_cnt_temp failed\n");
        return -1;
    }

    if ( (send_cnt_temp = (int*) calloc(sizeof(int), nprocs - 1)) == NULL ) {
        fprintf(stderr, "malloc() send_cnt_temp failed\n");
        return -1;
    }

    if ( (send_check = (int*) calloc(sizeof(int), nprocs - 1)) == NULL ) {
        fprintf(stderr, "malloc() send_check failed\n");
        return -1;
    }

    if ( (send_lst_temp = (int**) calloc(sizeof(int*), nprocs - 1)) == NULL ) {
        fprintf(stderr, "malloc() send_lst_temp 1st dimension failed\n");
        return -1;
    }

    if ( (recv_lst_temp = (int**) calloc(sizeof(int*), nprocs - 1)) == NULL ) {
        fprintf(stderr, "malloc() recv_lst_temp failed\n");
        return -1;
    }

    for (i = 0; i < nprocs - 1; ++i)
    {
        if ( (send_lst_temp[i] = (int*) calloc(sizeof(int), *nextci)) == NULL ) {
            fprintf(stderr, "malloc() send_lst_temp 2nd dimension failed\n");
            return -1;
        }

        if ( (recv_lst_temp[i] = (int*) calloc(sizeof(int), *nextci)) == NULL ) {
            fprintf(stderr, "malloc() recv_lst_temp 2nd dimension failed\n");
            return -1;
        }
    }

    *num_ghost_cells = 0; // counter for ghost cells - will contain total number in the end
    int counter = 0; // counter for external cells

    for (i = 0; i < elems_procs[myrank] * 6; ++i) // traverse through the whole (local) lcc array
    {
        if (i % 6 == 0) // each element's neighbours are 6 values of lcc array - when starting to examine next element's
                        // neighbours, reset send_check counter
            for (j = 0; j < *nghb_cnt; ++j)
                send_check[j] = 0; // check to see if element has already been added to send list to a specific neighbour

        int val = (*lcc)[i / 6][i % 6]; // store examined value (global index of neighbour)

        if (val >= *nextci) // if external globally
        {
            if (!hash_table[val]) // not counted before
            {
                counter++; // increment counter
                hash_table[val] = counter; // assign counter to hash table value for that cell
            }
            // change index to local (right after internal cells)
            (*lcc)[i / 6][i % 6] = *nintcf_local + hash_table[val];
        }

        else // internal or ghost
        {
            if (epart[val] == myrank) // check if it is internal
                (*lcc)[i / 6][i % 6] = global_local_index[val]; // and change index to local if so

            else // otherwise it's a ghost cell
            {
                // check which neighbour this cell belongs to and increment numver of neighbours if it's a new one
                count_nghb(val, &*nghb_cnt, myrank, &epart, nghb_to_rank_temp, rank_to_nghb);

                int index = rank_to_nghb[ epart[val] ]; // index of neighbour that this cell belongs to
                
                if (!hash_table[val]) // ghost cell has not been encountered before
                {
                    (*num_ghost_cells)++; // increment number of ghost cells
                    hash_table[val] = *num_ghost_cells; // assign that number to hash table value for that cell

                    // change index to local (right after external cells)
                    (*lcc)[i / 6][i % 6] = *nintcf_local + *num_ext_cells + hash_table[val];

                    global_local_index[val] = (*lcc)[i / 6][i % 6]; // update global_local_index accordingly
                    
                    recv_lst_temp[index] [recv_cnt_temp[ index ]] = val; // put cell on the receive list with global indexing
                    (recv_cnt_temp[index])++; // increment counter of receive list for that neighbour
                }

                else // ghost cell has not been encountered before
                    (*lcc)[i / 6][i % 6] = *nintcf_local + *num_ext_cells + hash_table[val]; // change index to local (right after external cells)

                if (!send_check[index]) // if element has already been added to send list for that neighbour, don't do anything
                {
                    send_lst_temp[index] [send_cnt_temp[ index ]] = i / 6; // put cell on the send list for that neighbour with local indexing
                    (send_cnt_temp[index])++; // increment counter of send list for that neighbour
                    send_check[index] = 1; // confirm the addition of the element - no need to add it again for the same neighbour
                }
            }
        }
    }

    // allocate (appropriate) space for send and receive counts and lists, neighbour to rank array
    if ( (*recv_cnt = (int*) calloc(sizeof(int), *nghb_cnt)) == NULL ) {
        fprintf(stderr, "malloc() recv_cnt failed\n");
        return -1;
    }

    if ( (*send_cnt = (int*) calloc(sizeof(int), *nghb_cnt)) == NULL ) {
        fprintf(stderr, "malloc() send_cnt failed\n");
        return -1;
    }
    
    if ( (*nghb_to_rank = (int*) calloc(sizeof(int), *nghb_cnt)) == NULL ) {
        fprintf(stderr, "malloc() recv_cnt failed\n");
        return -1;
    }

    // copy values from temporary arrays to permanent ones
    for (i = 0; i < *nghb_cnt; ++i)
    {
        (*send_cnt)[i] = send_cnt_temp[i];
        (*recv_cnt)[i] = recv_cnt_temp[i];
        (*nghb_to_rank)[i] = nghb_to_rank_temp[i];
    }

    if ( (*send_lst = (int**) calloc(sizeof(int*), *nghb_cnt)) == NULL ) {
        fprintf(stderr, "malloc() send_lst 1st dimension failed\n");
        return -1;
    }

    if ( (*recv_lst = (int**) calloc(sizeof(int*), *nghb_cnt)) == NULL ) {
        fprintf(stderr, "malloc() recv_lst failed\n");
        return -1;
    }

    for (i = 0; i < *nghb_cnt; ++i)
    {
        if ( ((*send_lst)[i] = (int*) calloc(sizeof(int), (*send_cnt)[i])) == NULL ) {
            fprintf(stderr, "malloc() send_lst 2nd dimension failed\n");
            return -1;
        }

        // copy values from temporary arrays to permanent ones
        for (j = 0; j < (*send_cnt)[i]; ++j)
            (*send_lst)[i][j] = send_lst_temp[i][j];

        if ( ((*recv_lst)[i] = (int*) calloc(sizeof(int), (*recv_cnt)[i])) == NULL ) {
            fprintf(stderr, "malloc() recv_lst 2nd dimension failed\n");
            return -1;
        }

        // copy values from temporary arrays to permanent ones
        for (j = 0; j < (*recv_cnt)[i]; ++j)
            (*recv_lst)[i][j] = recv_lst_temp[i][j];
    }

    // free temporary arrays
    for (i = 0; i < nprocs - 1; ++i)
    {        
        free(send_lst_temp[i]);
        free(recv_lst_temp[i]);
    }

    free(send_lst_temp);
    free(recv_lst_temp);
    free(send_check);
    free(hash_table);
    free(rank_to_nghb);

    // sort receive list to an ascending order of global indices - the same as the corresponding
    // send list of the neighbouring process; the send list is already ordered by construction
    for (i = 0; i < *nghb_cnt; ++i)
    {
        qsort((*recv_lst)[i], (*recv_cnt)[i], sizeof(int), cmpfunc);
        
        // finally change global indexing of receive list to local one
        for (j = 0; j < (*recv_cnt)[i]; ++j)
            (*recv_lst)[i][j] = global_local_index[ (*recv_lst)[i][j] ];
    }

    return 0;
}

int count_nghb (int val, int *nghb_cnt, int myrank, idx_t **epart, int *nghb_to_rank, int *rank_to_nghb) // this is to check if the neighbour is counted or not
{
    int i;

    for (i = 0; i < *nghb_cnt; i++)
        if ((*epart)[val] == nghb_to_rank[i]) return 0; // if neighbour has been counted before, return

    // else assign it to the next slot, update rank_to_nghb and increment neighbour count
    nghb_to_rank[*nghb_cnt] = (*epart)[val];
    rank_to_nghb[(*epart)[val]] = *nghb_cnt;
    (*nghb_cnt)++;

    return 0;
}

int allocate_2(double **BS, double **BE, double **BN, double **BW, double **BL, double **BH, double **BP, double **SU, int *nextcf)
{
    // allocate other arrays (boundary)
    if ( (*BS = (double *) malloc((*nextcf + 1) * sizeof(double))) == NULL ) {
        fprintf(stderr, "malloc(bs_local) failed\n");
        return -1;
    }

    if ( (*BE = (double *) malloc((*nextcf + 1) * sizeof(double))) == NULL ) {
        fprintf(stderr, "malloc(be_local) failed\n");
        return -1;
    }

    if ( (*BN = (double *) malloc((*nextcf + 1) * sizeof(double))) == NULL ) {
        fprintf(stderr, "malloc(bn_local) failed\n");
        return -1;
    }

    if ( (*BW = (double *) malloc((*nextcf + 1) * sizeof(double))) == NULL ) {
        fprintf(stderr, "malloc(bw_local) failed\n");
        return -1;
    }

    if ( (*BL = (double *) malloc((*nextcf + 1) * sizeof(double))) == NULL ) {
        fprintf(stderr, "malloc(bl_local) failed\n");
        return -1;
    }

    if ( (*BH = (double *) malloc((*nextcf + 1) * sizeof(double))) == NULL ) {
        fprintf(stderr, "malloc(bh_local) failed\n");
        return -1;
    }

    if ( (*BP = (double *) malloc((*nextcf + 1) * sizeof(double))) == NULL ) {
        fprintf(stderr, "malloc(bp_local) failed\n");
        return -1;
    }

    if ( (*SU = (double *) malloc((*nextcf + 1) * sizeof(double))) == NULL ) {
        fprintf(stderr, "malloc(su_local) failed\n");
        return -1;
    }

    return 0;
}

int transform_2(double **bs, double **be, double **bn, double **bw, double **bl, double **bh, double **bp, double **su, double **bs_global, double **be_global, double **bn_global, double **bw_global,
              double **bl_global, double **bh_global, double **bp_global, double **su_global, int myrank, idx_t *epart, int num_elm, int *global_local_index)
{
    int i;

    for ( i = 0; i < num_elm; ++i ) // for all elements in a process (internal, external and ghost)
    {
        if ( global_local_index[i] >= 0 ) // if not -1 then not external
        {
            int index = global_local_index[i];
            (*bs)[index] = (*bs_global)[i]; // copy boundary values
            (*be)[index] = (*be_global)[i];
            (*bn)[index] = (*bn_global)[i];
            (*bw)[index] = (*bw_global)[i];
            (*bl)[index] = (*bl_global)[i];
            (*bh)[index] = (*bh_global)[i];
            (*bp)[index] = (*bp_global)[i];
            (*su)[index] = (*su_global)[i];
        }
    }

    return 0;
}