#ifndef HELPER_H__
#define HELPER_H__

int allocate_elems_procs(int **elems_procs, int nprocs);

int read_partition(char *file_name, int *NINTCI, int *NINTCF, int *NEXTCI, int *NEXTCF, int ***LCC,
                    double **BS, double **BE, double **BN, double **BW, double **BL, double **BH,
                    double **BP, double **SU, int* points_count, int*** points, int** elems, int nprocs,
                    int **elems_procs, idx_t **epart, char *part_type);

int allocate_1(int **elems_procs, int myrank, int **local_global_index, int **global_local_index, int ***LCC, int *nintci, int *nintcf, int *nextci, int *nintcf_global);

int allocate_2(double **BS, double **BE, double **BN, double **BW, double **BL, double **BH, double **BP, double **SU, int *nextcf);

int transform_1(int ***lcc, int ***lcc_global, int myrank, idx_t *epart, int **local_global_index, int **global_local_index, int ne, int *nextci, int *nextcf, int *num_ext_cells);

int transform_2(double **bs, double **be, double **bn, double **bw, double **bl, double **bh, double **bp, double **su, double **bs_global, double **be_global, double **bn_global, double **bw_global,
              double **bl_global, double **bh_global, double **bp_global, double **su_global, int myrank, idx_t *epart, int num_elm, int *global_local_index);

int distribution(idx_t ne, idx_t nn, idx_t* eind, idx_t* eptr, idx_t ncommon, idx_t nparts, idx_t objval, idx_t** epart, idx_t** npart, char* part_type, int** elems);

int localise_lcc(int *elems_procs, int ***lcc, int *nintcf_local, int *nextci, int *nextcf, int *local_global_index, int *global_local_index, int myrank, int *counter, int *ghost_counter,
                        int *nghb_cnt, int** nghb_to_rank, int** send_cnt, int*** send_lst, 
                   int **recv_cnt, int*** recv_lst, idx_t *epart, int nprocs);
int allocate_3(int *nghb_cnt, int** nghb_to_rank, int** send_cnt, int*** send_lst, int **recv_cnt, int*** recv_lst);
int count_nghb (int val, int *nghb_cnt, int myrank, idx_t **epart, int *nghb_to_rank,  int *rank_to_nghb);

#endif
