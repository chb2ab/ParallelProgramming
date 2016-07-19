// Heated plate.c, chb2ab, Crispin Bernier
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
// Define the immutable boundary conditions and the inital cell value
#define TOP_BOUNDARY_VALUE 0.0
#define BOTTOM_BOUNDARY_VALUE 100.0
#define LEFT_BOUNDARY_VALUE 0.0
#define RIGHT_BOUNDARY_VALUE 100.0
#define INITIAL_CELL_VALUE 50.0
#define hotSpotRow 4500
#define hotSptCol 6500
#define hotSpotTemp 1000;

float **allocate_cells(int n_x, int n_y); // Used only by process 0 to initialize a matrix to write to the ppm
void create_snapshot(float **cells, int n_x, int n_y, int id);
void die(const char *error);

/**************************   MAIN    **************************/
int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	double etime0, etime1;
	etime0 = MPI_Wtime();
	// Read input parameters
	int num_cols = atoi(argv[1]);
	int num_rows =  atoi(argv[2]);
	int iterations = atoi(argv[3]);
	int y_dim = atoi(argv[4]);
	int iters_per_cell = atoi(argv[5]);
	int iterations_per_snapshot = atoi(argv[6]);
	int boundary_thickness = atoi(argv[7]);
	int bs = atoi(argv[8]);
	int num_cpus; // get the rank and total number of cpu's available
	int my_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &num_cpus);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	// number of partitions is always equal to the number of cpus
	y_dim = num_cpus;
	int rows_per_process = num_rows/num_cpus;
	int row_length;
	int bot_boundary_start;
	int top_boundary_start;
	int bot_end;
	int top_end;
	int row_start;
	// iterators
	int x, y, g, i, outer_iter, bx, by, x2, y2, ipc, rowstart, rowend;
	int cur_index = 0, nex_index = 1;
	int contains_hotspot;
	int hot_col;
	int hot_row;
	// calculate important boundary locations in each cell
	if (my_rank == 0) {
		row_length = 1+rows_per_process+boundary_thickness;
		bot_boundary_start = 1+rows_per_process-boundary_thickness;
		top_boundary_start = -1;
		bot_end = 1+rows_per_process;
		top_end = 1;
		row_start = 1;
	} else if (my_rank == num_cpus-1) {
		row_length = boundary_thickness+rows_per_process+1;
		bot_boundary_start = -1;
		top_boundary_start = boundary_thickness;
		bot_end = boundary_thickness+rows_per_process;
		top_end = 0;
		row_start = boundary_thickness;
	} else {
		row_length = boundary_thickness+rows_per_process+boundary_thickness;
		bot_boundary_start = rows_per_process;
		top_boundary_start = boundary_thickness;
		bot_end = boundary_thickness+rows_per_process;
		top_end = 0;
		row_start = boundary_thickness;
	}
	// determine whether or not this process has the hotspot
	if (num_cols >= 6500 && rows_per_process*my_rank-boundary_thickness <= 4499 && rows_per_process*(my_rank+1)+boundary_thickness > 4499) {
		contains_hotspot = 1;
		hot_row = 4499-(rows_per_process*my_rank)+row_start;
		hot_col = 6500;
	}
	else contains_hotspot = 0;
	int outer_iterations = iterations/boundary_thickness;
	int iter = 0;
	int cell_size = (num_cols+2)*row_length;
	// initialize 1 dimensional heated plate array
	float *cells1d = malloc(2*sizeof(float)*cell_size);
	for (y = 0; y < row_length; y++) {
		for (x = 1; x <= num_cols; x++) {
			cells1d[0*(cell_size)+y*(num_cols+2)+x] = INITIAL_CELL_VALUE;
		}
	}
	if (my_rank == 0) for (x = 1; x <= num_cols; x++) cells1d[x] = cells1d[cell_size+x] = TOP_BOUNDARY_VALUE;
	else if (my_rank == num_cpus-1) for (x = 1; x <= num_cols; x++) cells1d[(row_length-1)*(num_cols+2)+x] = cells1d[cell_size+(row_length-1)*(num_cols+2)+x] = BOTTOM_BOUNDARY_VALUE;
	for (y = 0; y < row_length; y++) cells1d[y*(num_cols+2)] = cells1d[cell_size+y*(num_cols+2)] = LEFT_BOUNDARY_VALUE;
	for (y = 0; y < row_length; y++) cells1d[y*(num_cols+2)+num_cols+1] = cells1d[cell_size+y*(num_cols+2)+num_cols+1] = RIGHT_BOUNDARY_VALUE;
	// process 0 needs a matrix for writing data to ppm
	float *recBlockF;
	float **final_cells;
	if (my_rank == 0) {
		recBlockF = malloc(sizeof(float)*rows_per_process*(num_cols+2));
		final_cells = allocate_cells(num_cols+2, num_rows+2);
	}
    MPI_Request send_top, send_bot, rec_top, rec_bot;
    /*** Perform each iteration ***/
	for (outer_iter = 0; outer_iter < outer_iterations; outer_iter++) {
		// communications
		if (my_rank != 0) {
			// send up
			MPI_Isend(&cells1d[cur_index*cell_size+top_boundary_start*(num_cols+2)], boundary_thickness*(num_cols+2), MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD, &send_top);
			// listen up
			MPI_Irecv(&cells1d[cur_index*cell_size+top_end*(num_cols+2)], boundary_thickness*(num_cols+2), MPI_FLOAT, my_rank-1, 2, MPI_COMM_WORLD, &rec_top);
		}
		if (my_rank != num_cpus-1) {
			// send down
			MPI_Isend(&cells1d[cur_index*cell_size+bot_boundary_start*(num_cols+2)], boundary_thickness*(num_cols+2), MPI_FLOAT, my_rank+1, 2, MPI_COMM_WORLD, &send_bot);
			// listen down
			MPI_Irecv(&cells1d[cur_index*cell_size+bot_end*(num_cols+2)], boundary_thickness*(num_cols+2), MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD, &rec_bot);
		}
		/** Do first inner loop iteration as message is being sent **/
		for (ipc = 0; ipc < iters_per_cell; ipc++) { // repeated calculations for iters_per_cell
			y = 1;
			x = 1;
			bx = num_cols;
			for (by = row_start+1; by+bs < bot_end-1; by += bs) {
				for (bx = 1; bx+bs <= num_cols; bx += bs) {
					for (y = by; y < by+bs; y++) {
						for (x = bx; x < bx+bs; x++) {
							cells1d[nex_index*cell_size+y*(num_cols+2)+x] = (cells1d[cur_index*cell_size+y*(num_cols+2)+x-1] +
								cells1d[cur_index*cell_size+y*(num_cols+2)+x+1] +
								cells1d[cur_index*cell_size+(y-1)*(num_cols+2)+x] +
								cells1d[cur_index*cell_size+(y+1)*(num_cols+2)+x]) * 0.25;
			}	}	}	}
			// complete extra cells from loop blocking
			if (bx != num_cols) {
				for (y2 = 1; y2 < by; y2++) {
					for (x2 = bx; x2 <= num_cols; x2++) {
						cells1d[nex_index*cell_size+y2*(num_cols+2)+x2] = (cells1d[cur_index*cell_size+y2*(num_cols+2)+x2-1] +
							cells1d[cur_index*cell_size+y2*(num_cols+2)+x2+1] +
							cells1d[cur_index*cell_size+(y2-1)*(num_cols+2)+x2] +
							cells1d[cur_index*cell_size+(y2+1)*(num_cols+2)+x2]) * 0.25;
			}	}	}
			while (y < bot_end-1) {
				for (x = 1; x <= num_cols; x++) {
					cells1d[nex_index*cell_size+y*(num_cols+2)+x] = (cells1d[cur_index*cell_size+y*(num_cols+2)+x-1] +
						cells1d[cur_index*cell_size+y*(num_cols+2)+x+1] +
						cells1d[cur_index*cell_size+(y-1)*(num_cols+2)+x] +
						cells1d[cur_index*cell_size+(y+1)*(num_cols+2)+x]) * 0.25;
				}
				y++;
		}	}
		/** wait for message to arrive before finishing the iteration **/
		if (my_rank != 0) {
			MPI_Wait(&rec_top, MPI_STATUS_IGNORE);
		}
		for (y = 1; y <= row_start; y++) {
			for (x = 1; x <= num_cols; x++) {
				cells1d[nex_index*cell_size+y*(num_cols+2)+x] = (cells1d[cur_index*cell_size+y*(num_cols+2)+x-1] +
					cells1d[cur_index*cell_size+y*(num_cols+2)+x+1] +
					cells1d[cur_index*cell_size+(y-1)*(num_cols+2)+x] +
					cells1d[cur_index*cell_size+(y+1)*(num_cols+2)+x]) * 0.25;
		}	}
		if (my_rank != num_cpus-1) {
			MPI_Wait(&rec_bot, MPI_STATUS_IGNORE);
		}
		for (y = bot_end-1; y < row_length-1; y++) {
			for (x = 1; x <= num_cols; x++) {
				cells1d[nex_index*cell_size+y*(num_cols+2)+x] = (cells1d[cur_index*cell_size+y*(num_cols+2)+x-1] +
					cells1d[cur_index*cell_size+y*(num_cols+2)+x+1] +
					cells1d[cur_index*cell_size+(y-1)*(num_cols+2)+x] +
					cells1d[cur_index*cell_size+(y+1)*(num_cols+2)+x]) * 0.25;
		}	}
		cur_index = nex_index;
		nex_index = !nex_index;
		iter++;
		if (contains_hotspot) cells1d[cur_index*cell_size+hot_row*(num_cols+2)+hot_col] = hotSpotTemp;
		/** Complete the remaining inner loop iterations **/
		for (g = 1; g < boundary_thickness; g++) {
			if (my_rank == 0 || my_rank == num_cpus-1) {
				rowstart = 1;
				rowend = row_length-1;
			} else {
				rowstart = 2;
				rowend = row_length-2;
			}
			for (ipc = 0; ipc < iters_per_cell; ipc++) { // repeated calculations for iters_per_cell
				y = rowstart;
				x = 1;
				bx = num_cols;
				for (by = rowstart; by+bs < rowend; by += bs) {
					for (bx = 1; bx+bs <= num_cols; bx += bs) {
						for (y = by; y < by+bs; y++) {
							for (x = bx; x < bx+bs; x++) {
								cells1d[nex_index*cell_size+y*(num_cols+2)+x] = (cells1d[cur_index*cell_size+y*(num_cols+2)+x-1] +
									cells1d[cur_index*cell_size+y*(num_cols+2)+x+1] +
									cells1d[cur_index*cell_size+(y-1)*(num_cols+2)+x] +
									cells1d[cur_index*cell_size+(y+1)*(num_cols+2)+x]) * 0.25;
				}	}	}	}
				// complete extra cells from loop blocking
				if (bx != num_cols) {
					for (y2 = 1; y2 < by; y2++) {
						for (x2 = bx; x2 <= num_cols; x2++) {

							cells1d[nex_index*cell_size+y2*(num_cols+2)+x2] = (cells1d[cur_index*cell_size+y2*(num_cols+2)+x2-1] +
								cells1d[cur_index*cell_size+y2*(num_cols+2)+x2+1] +
								cells1d[cur_index*cell_size+(y2-1)*(num_cols+2)+x2] +
								cells1d[cur_index*cell_size+(y2+1)*(num_cols+2)+x2]) * 0.25;
				}	}	}
				while (y < rowend) {
					for (x = 1; x <= num_cols; x++) {
						cells1d[nex_index*cell_size+y*(num_cols+2)+x] = (cells1d[cur_index*cell_size+y*(num_cols+2)+x-1] +
							cells1d[cur_index*cell_size+y*(num_cols+2)+x+1] +
							cells1d[cur_index*cell_size+(y-1)*(num_cols+2)+x] +
							cells1d[cur_index*cell_size+(y+1)*(num_cols+2)+x]) * 0.25;
					}
					y++;
			}	}
			cur_index = nex_index;
			nex_index = !nex_index;
			iter++;
			if (contains_hotspot) cells1d[cur_index*cell_size+hot_row*(num_cols+2)+hot_col] = hotSpotTemp;
			/** Take snapshot every iterations_per_snapshot **/
			if (iter%iterations_per_snapshot == 0) {
				if (my_rank == 0) {
					MPI_Barrier(MPI_COMM_WORLD);
					for (y = 1; y < row_length-boundary_thickness; y++) {
						for (x = 1; x <= num_cols; x++) {
							final_cells[y][x] = cells1d[cur_index*cell_size+y*(num_cols+2)+x];
						}
					}
					for (i = 1; i < num_cpus; i++) {
						MPI_Recv(recBlockF, rows_per_process*(num_cols+2), MPI_FLOAT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						for (y = 0; y < rows_per_process; y++) {
							for (x = 1; x <= num_cols; x++) {
								final_cells[1+(i*rows_per_process)+y][x] = recBlockF[y*(num_cols+2)+x];
							}
						}
					}
					create_snapshot(final_cells, num_cols, num_rows, iter);
					etime1 = MPI_Wtime();
					printf("\nExecution time: %14.4f seconds\n", etime1-etime0);
				} else {
					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Send(&cells1d[cur_index*cell_size+row_start*(num_cols+2)], rows_per_process*(num_cols+2), MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
				}
			}
		}
		// Wait for the sent messages for this round to complete
		if (my_rank != 0) {
			MPI_Wait(&send_top, MPI_STATUS_IGNORE);
		}
		if (my_rank != num_cpus-1) {
			MPI_Wait(&send_bot, MPI_STATUS_IGNORE);
		}
		// Wait for all processes to complete the round before continuing so as to not overwrite data with a new message
		MPI_Barrier(MPI_COMM_WORLD);
	}
	// Print final execution time
	if (my_rank == 0) {
		etime1 = MPI_Wtime();
		printf("\nExecution time: %14.4f seconds\n", etime1-etime0);
	}
	MPI_Finalize();
	return 0;
}
/*******************
Function Definitions
*******************/
// Create new matrix
float **allocate_cells(int num_cols, int num_rows) {
	float **array = (float **) malloc(num_rows * sizeof(float *));
	if (array == NULL) die("Error allocating array!\n");
	array[0] = (float *) malloc(num_rows * num_cols * sizeof(float));
	if (array[0] == NULL) die("Error allocating array!\n");
	int i;
	for (i = 1; i < num_rows; i++) {
		array[i] = array[0] + (i * num_cols);
	}
	return array;
}
// Create a ppm snapshot
void create_snapshot(float **cells, int num_cols, int num_rows, int id) {
	int scale_x, scale_y;
	scale_x = scale_y = 1;
	if (num_cols > 1000) {
		if ((num_cols % 1000) == 0) scale_x = num_cols / 1000;
		else {
			die("Cannot create snapshot for x-dimensions >1,000 that are not multiples of 1,000!\n");
			return;
	}	}
	if (num_rows > 1000) {
		if ((num_rows % 1000) == 0) scale_y = num_rows / 1000;
		else {
			printf("Cannot create snapshot for y-dimensions >1,000 that are not multiples of 1,000!\n");
			return;
	}	}
	char text[255];
	sprintf(text, "snapshot.%d.ppm", id);
	FILE *out = fopen(text, "w");
	if (out == NULL) {
		printf("Error creating snapshot file!\n");
		return;
	}
	fprintf(out, "P3 %d %d 100\n", num_cols / scale_x, num_rows / scale_y);
	float inverse_cells_per_pixel = 1.0 / ((float) scale_x * scale_y);
	int x, y, i, j;
	for (y = 1; y <= num_rows; y += scale_y) {
		for (x = 1; x <= num_cols; x += scale_x) {
			float sum = 0.0;
			for (j = y; j < y + scale_y; j++) {
				for (i = x; i < x + scale_x; i++) {
					sum += cells[j][i];
			}	}
			int average = (int) (sum * inverse_cells_per_pixel);
			if (average > 100) {
				average = 100;
			}
			fprintf(out, "%d 0 %d\t", average, 100 - average);
		}
		fwrite("\n", sizeof(char), 1, out);
	}
	fclose(out);
}
void die(const char *error) {
	printf("%s", error);
	exit(1);
}