// Heated plate.c, chb2ab, Crispin Bernier
#define _GNU_SOURCE
#include <pthread.h> 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
// Define the immutable boundary conditions and the inital cell value
#define TOP_BOUNDARY_VALUE 0.0
#define BOTTOM_BOUNDARY_VALUE 100.0
#define LEFT_BOUNDARY_VALUE 0.0
#define RIGHT_BOUNDARY_VALUE 100.0
#define INITIAL_CELL_VALUE 50.0
#define hotSpotRow 4500
#define hotSptCol 6500
#define hotSpotTemp 1000;
// Each thread has a start (inclusive) and end row (non inclusive) and a thread number
typedef struct thread_vars {
	int start;
	int end;
	int thread_number;
} thread_vars;
// Dep will enforce synchronization between threads
typedef struct dep {
	int not_ready;
	pthread_mutex_t count_mutex;
	pthread_cond_t count_threshold_cv;
} dep;
// Global variables for a thread to access
float **cells[2];// global pointer to the plate
dep **deps;		// dependency array that enforces synchronization
int bs;			// block size
int num_cols;	// total number of columns
int num_rows;	// total number of rows
int iterations;	// number of iterations to perform
int num_threads;// total number of threads
float **allocate_cells(int n_x, int n_y);
/********************************
This is the thread function. This is where each thread does all of its work.
********************************/
void *process(void *param) {
	// Variables local to the thread
	thread_vars *thr = (thread_vars *)(param);	// data passed in thread_vars struct
	int start = thr->start;
	int end = thr->end;
	int thread_number = thr->thread_number;
	int length = end-start;						// length of the this threads rows
	int cur_cells_index = 0, next_cells_index = 1;
	int iteration = 0;							// number of iterations performed
	int upper = thread_number == 0;				// true if this is the first thread
	int lower = thread_number == num_threads-1;	// true if this is the last thread
	int x = 1, y = start, by, bx;
// true if cell contains the hotspot, including if it is in it's ghost range
	int hotspot = start <= hotSpotRow && end > hotSpotRow && num_cols >= hotSptCol;
// extra columns to handle after blocking
	int leftoverx = num_cols-(num_cols/bs)*bs;
/*
	On each iteration, a thread will check it's neighboring threads before performing the
	heated plate analysis. When it is done it will write the results back to the global
	array and broadcast that it has completed an iteration.
*/
/*** Handle the first and last threads. This also includes if there is only 1 thread. ***/
	if (upper || lower) {
		while (iteration < iterations) {
// Check neighboring threads
			if (lower && !upper) {
				pthread_mutex_lock(&deps[iteration][thread_number-1].count_mutex);
				if (deps[iteration][thread_number-1].not_ready == 1) {
					pthread_cond_wait(&deps[iteration][thread_number-1].count_threshold_cv, &deps[iteration][thread_number-1].count_mutex);
				}
				pthread_mutex_unlock(&deps[iteration][thread_number-1].count_mutex);
			}
			if (upper && !lower) {
				pthread_mutex_lock(&deps[iteration][thread_number+1].count_mutex);
				if (deps[iteration][thread_number+1].not_ready == 1) {
					pthread_cond_wait(&deps[iteration][thread_number+1].count_threshold_cv, &deps[iteration][thread_number+1].count_mutex);
				}
				pthread_mutex_unlock(&deps[iteration][thread_number+1].count_mutex);
			}
// heated plate function
			y = start;
			for (by = start; by < end-bs; by += bs) {
				for (bx = 1; bx <= num_cols-bs; bx += bs) {
					for (y = by; y < by+bs; y++) {
						for (x = bx; x < bx+bs; x++) {
					// The new value of this cell is the average of the old values of this cell's four neighbors
					cells[next_cells_index][y][x] = (cells[cur_cells_index][y][x - 1]  +
					                                 cells[cur_cells_index][y][x + 1]  +
					                                 cells[cur_cells_index][y - 1][x]  +
					                                 cells[cur_cells_index][y + 1][x]) * 0.25;
				}	}	}
				if (leftoverx > 0) {
					for (y = by; y < by+bs;y++) {
						for(x = bx; x <= num_cols; x++) {
						cells[next_cells_index][y][x] = (cells[cur_cells_index][y][x - 1]  +
							                                 cells[cur_cells_index][y][x + 1]  +
							                                 cells[cur_cells_index][y - 1][x]  +
							                                 cells[cur_cells_index][y + 1][x]) * 0.25;
			}	}	}	}
			while (y < end) {
				for (x = 1; x <= num_cols; x++) {
					cells[next_cells_index][y][x] = (cells[cur_cells_index][y][x - 1]  +
					                                 cells[cur_cells_index][y][x + 1]  +
					                                 cells[cur_cells_index][y - 1][x]  +
					                                 cells[cur_cells_index][y + 1][x]) * 0.25;
				}
				y++;
			}
			if (hotspot) cells[next_cells_index][hotSpotRow][hotSptCol]=hotSpotTemp;
			cur_cells_index = next_cells_index;
			next_cells_index = !cur_cells_index;

			iteration++;
// broadcast
			if (!lower || !upper) {
				pthread_mutex_lock(&deps[iteration][thread_number].count_mutex);
				deps[iteration][thread_number].not_ready--;
    			pthread_cond_broadcast(&deps[iteration][thread_number].count_threshold_cv);
				pthread_mutex_unlock(&deps[iteration][thread_number].count_mutex);
		}	}
/*** Handle the middle threads. ***/
	} else {
		while (iteration < iterations) {
// Check neighboring threads
			pthread_mutex_lock(&deps[iteration][thread_number-1].count_mutex);
			if (deps[iteration][thread_number-1].not_ready == 1) {
				pthread_cond_wait(&deps[iteration][thread_number-1].count_threshold_cv, &deps[iteration][thread_number-1].count_mutex);
			}
			pthread_mutex_unlock(&deps[iteration][thread_number-1].count_mutex);

			pthread_mutex_lock(&deps[iteration][thread_number+1].count_mutex);
			if (deps[iteration][thread_number+1].not_ready == 1) {
				pthread_cond_wait(&deps[iteration][thread_number+1].count_threshold_cv, &deps[iteration][thread_number+1].count_mutex);
			}
			pthread_mutex_unlock(&deps[iteration][thread_number+1].count_mutex);
// heated plate function
			y = start;
			for (by = start; by < end-bs; by += bs) {
				for (bx = 1; bx <= num_cols-bs; bx += bs) {
					for (y = by; y < by+bs; y++) {
						for (x = bx; x < bx+bs; x++) {
					// The new value of this cell is the average of the old values of this cell's four neighbors
					cells[next_cells_index][y][x] = (cells[cur_cells_index][y][x - 1]  +
					                                 cells[cur_cells_index][y][x + 1]  +
					                                 cells[cur_cells_index][y - 1][x]  +
					                                 cells[cur_cells_index][y + 1][x]) * 0.25;
				}	}	}
				if (leftoverx > 0) {
					for (y = by; y < by+bs;y++) {
						for(x = bx; x <= num_cols; x++) {
						cells[next_cells_index][y][x] = (cells[cur_cells_index][y][x - 1]  +
							                                 cells[cur_cells_index][y][x + 1]  +
							                                 cells[cur_cells_index][y - 1][x]  +
							                                 cells[cur_cells_index][y + 1][x]) * 0.25;
			}	}	}	}
			while (y < end) {
				for (x = 1; x <= num_cols; x++) {
					cells[next_cells_index][y][x] = (cells[cur_cells_index][y][x - 1]  +
					                                 cells[cur_cells_index][y][x + 1]  +
					                                 cells[cur_cells_index][y - 1][x]  +
					                                 cells[cur_cells_index][y + 1][x]) * 0.25;
				}
				y++;
			}
			if (hotspot) cells[next_cells_index][hotSpotRow][hotSptCol]=hotSpotTemp;

			cur_cells_index = next_cells_index;
			next_cells_index = !cur_cells_index;

// broadcast
			iteration++;
			pthread_mutex_lock(&deps[iteration][thread_number].count_mutex);
			deps[iteration][thread_number].not_ready--;
    		pthread_cond_broadcast(&deps[iteration][thread_number].count_threshold_cv);
			pthread_mutex_unlock(&deps[iteration][thread_number].count_mutex);
	}	}
	pthread_exit(0);
};
void print_cells(float **cells, int n_x, int n_y);
void initialize_cells(float **cells, int n_x, int n_y);
void create_snapshot(float **cells, int n_x, int n_y, int id);
dep **allocate_cells_dep(int n_x, int n_y); // Allocates 2D array of dep structs
void die(const char *error);

/**************************   MAIN    **************************/
int main(int argc, char **argv) {
	time_t start_time = time(NULL);  // time program and command line arguments
	num_cols = (argc > 1) ? atoi(argv[1]) : 1000;
	num_rows = (argc > 2) ? atoi(argv[2]) : 1000;
	iterations = (argc > 3) ? atoi(argv[3]) : 100;
	num_threads = (argc > 4) ? atoi(argv[4]) : 10;
	bs = atoi(argv[5]);
	int num_cpus = atoi(argv[6]); // indicate how many cpu's are available
	printf("Grid: %dx%d, Iterations: %d\n", num_cols, num_rows, iterations);
// Initialize global arrays for both timesteps, current and future= allocate_cells(num_cols + 2, num_rows + 2);
	cells[0] = allocate_cells(num_cols + 2, num_rows + 2);
	cells[1] = allocate_cells(num_cols + 2, num_rows + 2);
	initialize_cells(cells[0], num_cols, num_rows);
	int x, y, i;
	for (x = 1; x <= num_cols; x++) cells[0][0][x] = cells[1][0][x] = TOP_BOUNDARY_VALUE;
	for (x = 1; x <= num_cols; x++) cells[0][num_rows + 1][x] = cells[1][num_rows + 1][x] = BOTTOM_BOUNDARY_VALUE;
	for (y = 1; y <= num_rows; y++) cells[0][y][0] = cells[1][y][0] = LEFT_BOUNDARY_VALUE;
	for (y = 1; y <= num_rows; y++) cells[0][y][num_cols + 1] = cells[1][y][num_cols + 1] = RIGHT_BOUNDARY_VALUE;

	deps = allocate_cells_dep(num_threads, iterations+1);
// deps holds the synchronization information needed by the threads
	for (x = 0; x < num_threads; x++) {
		deps[0][x].not_ready = 0;
  		pthread_mutex_init(&deps[0][x].count_mutex, NULL);
  		pthread_cond_init (&deps[0][x].count_threshold_cv, NULL);
	}
	for (y = 1; y < iterations+1; y++) {
		for (x = 0; x < num_threads; x++) {
			deps[y][x].not_ready = 1;
  			pthread_mutex_init(&deps[y][x].count_mutex, NULL);
  			pthread_cond_init (&deps[y][x].count_threshold_cv, NULL);
	}	}
// pin each thread to a processor
	pthread_t *tid = malloc(sizeof(pthread_t)*num_threads);
	pthread_attr_t *attr = malloc(sizeof(pthread_attr_t)*num_threads);
	thread_vars *thr;
	cpu_set_t *cpuset = malloc(sizeof(cpu_set_t)*num_threads);
	int iter;
	int divv = num_threads/num_cpus;
	int offset = num_cpus/num_threads;
	offset = offset == 0 ? 1 : offset;
	for (iter = 0; iter < num_threads; iter++) {
		pthread_attr_init(&attr[iter]);
		CPU_ZERO(&cpuset[iter]);
		CPU_SET( (iter*offset)%num_cpus, &cpuset[iter] );
		pthread_attr_setaffinity_np(&attr[iter], sizeof(cpu_set_t), &cpuset[iter]);
	}
// initialize the threads. Starting from row one each thread gets equal number of rows
	int rows_per_thread = num_rows/num_threads;
	int leftover = num_rows - rows_per_thread * num_threads;
	int start_row = 1;
	for (i = 0; i < leftover; i++) {
		thr = malloc(sizeof(thread_vars));
		thr->start = start_row;
		start_row += rows_per_thread + 1;
		thr->end = start_row;
		thr->thread_number = i;
		pthread_create(&tid[i],&attr[i],process,thr);
	}
	while (i < num_threads) {
		thr = malloc(sizeof(thread_vars));
		thr->start = start_row;
		start_row += rows_per_thread;
		thr->end = start_row;
		thr->thread_number = i;
		pthread_create(&tid[i],&attr[i],process,thr);
		i++;
	}
// wait for threads to finish
	for (i = 0; i < num_threads; i++) {
		pthread_join(tid[i],NULL);
	}
// Output snapshot
	int final_cells = (iterations % 2 == 0) ? 0 : 1;
	create_snapshot(cells[final_cells], num_cols, num_rows, iterations);
	time_t end_time = time(NULL);
	printf("\nExecution time: %d seconds\n", (int) difftime(end_time, start_time));
	free(deps);
	return 0;
}
/*******************
Function Definitions
*******************/
// The only function that was added was allocate_cells_dep
// The rest were unchanged from the original source file
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
dep **allocate_cells_dep(int num_cols, int num_rows) {
	dep **array = (dep **) malloc(num_rows * sizeof(dep *));
	if (array == NULL) die("Error allocating array!\n");
	array[0] = (dep *) malloc(num_rows * num_cols * sizeof(dep));
	if (array[0] == NULL) die("Error allocating array!\n");
	int i;
	for (i = 1; i < num_rows; i++) {
		array[i] = array[0] + (i * num_cols);
	}
	return array;
}
void initialize_cells(float **cells, int num_cols, int num_rows) {
	int x, y;
	for (y = 1; y <= num_rows; y++) {
		for (x = 1; x <= num_cols; x++) {
			cells[y][x] = INITIAL_CELL_VALUE;
}	}	}
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