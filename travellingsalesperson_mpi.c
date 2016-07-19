#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#include <limits.h>

// The schedulers circular queue.
typedef struct queue {
	int start; // next prefix to consume
	int end;
	int full; // 1 if full
	int empty; // 1 if empty
	int *elements;
} queue;

int prefix_length, num_cities, num_cpus;
int queue_length;
queue gen_queue;

// Add a prefix to the queue and update the end value.
// If the new end is equal to start the queue is full, otherwise it is not.
void add_prefix(int *prefix) {
	int i;
	int end = gen_queue.end;
	for (i = 0; i < prefix_length; i++) {
		gen_queue.elements[end+i] = prefix[i];
	}
	gen_queue.end = (end+prefix_length+1)%queue_length;
	if (gen_queue.end == gen_queue.start) gen_queue.full = 1;
	gen_queue.empty = 0;
	return;
}

// Remove a prefix from the queue and update the start value.
// If the new start is equal to end the queue is empty otherwise it is not
void consume_prefix() {
	gen_queue.start = (gen_queue.start+prefix_length+1)%queue_length;
	if (gen_queue.start == gen_queue.end) gen_queue.empty = 1;
	gen_queue.full = 0;
	return;
}

MPI_Request *rec_statuses;
int **rec_buffers; // worker results stored here
int cur_min = INT_MAX; // min path and it's length stored globally
int *min_path;
/*** Generate prefixes and place them into the queue while also recieving results from the workers ***/
void generator(int pindex, int *prefix) {
	int i, p, repeat;
	if (pindex == prefix_length) {
	/** Base case where a prefix has finished generating. Before adding the prefix check for worker requests**/
		int testflag, q;
		int do_once = 0;
		MPI_Status status;
		MPI_Request send;
		/** While the queue is full, loop through assigning work until the queue is not full.
		 If the queue is not full do 1 loop through to assign work to any idle processes **/
		while (do_once == 0 || gen_queue.full == 1) {
			for (q = 1; q < num_cpus; q++) {
			/** If the queue is empty don't check for idle processes because there are no prefixes to give out **/
				if (gen_queue.empty != 1) {
					MPI_Test(&(rec_statuses[q]), &testflag, &status);
					/** If a process has finished it's work process the result and send a new prefix **/
					if (testflag != 0) {
						// Process the recieved work, comparing it to the current minimum
						int branches_min = rec_buffers[q][num_cities];
						if (branches_min < cur_min) {
							cur_min = branches_min;
							for (i = 0; i < num_cities; i++) {
								min_path[i] = rec_buffers[q][i];
							}
						}
						// Send a new prefix to the process that finished
						gen_queue.elements[ gen_queue.start + prefix_length ] = cur_min;
						MPI_Isend(&gen_queue.elements[ gen_queue.start ], prefix_length+1, MPI_INT, q, 0, MPI_COMM_WORLD, &send);
						// Wait asynchronously for that process to finish
						MPI_Irecv(rec_buffers[q], 1+num_cities, MPI_INT, q, 0, MPI_COMM_WORLD, &(rec_statuses[q]));
						// remove the sent prefix from the queue
						consume_prefix();
					}
				} else { break;	} // queue is empty so we can't send out more work
			}
			do_once = 1;
		}
		/*** Add the prefix to the queue ***/
		add_prefix(prefix);
		return;
	}
	/** Recursively generate prefixes by incrementing an index counter **/
	for (i = 1; i < num_cities; i++) {
		repeat = 0;
		for (p = 1; p < pindex; p++) {
			if (prefix[p] == i) {
				repeat = 1;
				break;
			}
		}
		if (repeat == 1) continue;
		prefix[pindex] = i;
		generator(pindex+1, prefix);
	}
	return;
}

int **city_matrix; // Holds the graph represented as a matrix
/*** Traverse a subtree recursively, setting min_path to be the
minimum path found and cur_min to be the minimum length found ***/
void traverse(int curindex, int *visited, int cur_dist) {
	int i, p;
	/** Base case for recursive call, compare the path traversed
	to the current minimum and take the smaller value**/
	if (curindex == num_cities) {
		int last_city = visited[num_cities-1];
		cur_dist += city_matrix[last_city][0];
		if (cur_dist < cur_min) {
			cur_min = cur_dist;
			for (i = 0; i < num_cities; i++) {
				min_path[i] = visited[i];
			}
			min_path[num_cities] = cur_min;
		}
		return;
	}
	/** Recursive solution to TSP problem using an index counter **/
	int prev_city = visited[curindex-1];
	int new_dist;
	int repeat;
	// Loop through all possible cities
	for (i = 1; i < num_cities; i++) {
		repeat = 0;
		for (p = 1; p < curindex; p++) {
			if (visited[p] == i) {
				repeat = 1;
				break;
			}
		}
		if (repeat == 1) continue; // Skip branches that would be repetitions
		new_dist = cur_dist+city_matrix[prev_city][i];
		if (new_dist > cur_min) continue; // Skip branches that are greater then the bound
		visited[curindex] = i;
		traverse(curindex+1, visited, new_dist);
	}
	return;
}

/**************************   MAIN    **************************/
int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int i, j;
	int my_rank;
	/** read in the city matrix **/
	FILE *infile;
	char *file = argv[1];
	char *tok, line[500];

	infile = fopen(file, "r");
	while (fgets(line, 500, infile) != NULL) {
		num_cities++;
	}
	city_matrix = malloc(sizeof(int*)*num_cities);
	for (i = 0; i < num_cities; i++) {
		city_matrix[i] = malloc(sizeof(int)*num_cities);
	}

	rewind(infile);
	i = j = 0;
	while (fgets(line, 500, infile) != NULL) {
		tok = strtok(line, " ");
		for (j = 0; j < num_cities; j++) {
			city_matrix[i][j] = atoi(tok);
			tok = strtok(NULL, " ");
		}
		i++;
	}
	fclose(infile);

	prefix_length = atoi(argv[2]); // Prefix length given as a command line parameter
	// An element in the queue consists of a prefix plus the current min
	queue_length = (prefix_length+1)*1000;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (my_rank == 0) {
		/**** Scheduler ****/
		double etime0, etime1;
		etime0 = MPI_Wtime(); // start timing
		MPI_Comm_size(MPI_COMM_WORLD, &num_cpus);
		min_path = malloc(sizeof(int)*num_cities);
		// Initialize the queue
		gen_queue.start = 0;
		gen_queue.end = 0;
		gen_queue.full = 0;
		gen_queue.empty = 1;
		gen_queue.elements = malloc(sizeof(int)*queue_length);
		/** Greedy approximation of the minimum path length **/
		min_path[0] = 0;
		int initial_guess = 0;
		int k, next_city, cur_city, repeat, next_dist;
		for (i = 1; i < num_cities; i++) {
			cur_city = min_path[i-1];
			next_dist = INT_MAX;
			for (j = 1; j < num_cities; j++) {
				repeat = 0;
				for (k = 0; k < i; k++) {
					if (min_path[k] == j) repeat = 1;
				}
				if (repeat == 1) continue;
				if (city_matrix[cur_city][j] < next_dist) {
					next_city = j;
					next_dist = city_matrix[cur_city][j];
				}
			}
			min_path[i] = next_city;
			initial_guess += city_matrix[cur_city][next_city];
		}
		initial_guess += city_matrix[ min_path[num_cities-1] ][0];
		cur_min = initial_guess;
		// Initialize recieve requests and recieve buffers
		rec_statuses = malloc(sizeof(MPI_Request)*(num_cpus));
		rec_buffers = malloc(sizeof(int*)*(num_cpus));

		for (i = 1; i < num_cpus; i++) {
			rec_buffers[i] = malloc((sizeof(int)+sizeof(int)*num_cities));
			// Each process will signal to 0 when it is ready for work
			MPI_Irecv(rec_buffers[i], 1+num_cities, MPI_INT, i, 0, MPI_COMM_WORLD, &(rec_statuses[i]));
		}
		int prefix[prefix_length];
		// Start city is fixed to city 0
		prefix[0] = 0;
		// Generate prefixes, this function will send prefixes to processes as they are being generated
		generator(1, prefix);
		// While the queue isn't empty, send out the remaining prefixes
		int q = 0;
		int testflag;
		MPI_Status status;
		MPI_Request send;
		while (gen_queue.empty != 1) {
			// q iterates repeatedly from 1 to num_cpus-1
			q += 1;
			MPI_Test(&rec_statuses[q], &testflag, &status);
			/** If a process has finished it's work process the result and send a new prefix **/
			if (testflag != 0) {
				// Process the recieved work, comparing it to the current minimum
				int branches_min = rec_buffers[q][num_cities];
				if (branches_min < cur_min) {
					cur_min = branches_min;
					for (i = 0; i < num_cities; i++) {
						min_path[i] = rec_buffers[q][i];
					}
				}
				// Send a new prefix to the process that finished
				gen_queue.elements[ gen_queue.start + prefix_length ] = cur_min;
				MPI_Isend(&gen_queue.elements[ gen_queue.start ], prefix_length+1, MPI_INT, q, 0, MPI_COMM_WORLD, &send);
				// Wait asynchronously for that process to finish
				MPI_Irecv(rec_buffers[q], 1+num_cities, MPI_INT, q, 0, MPI_COMM_WORLD, &(rec_statuses[q]));
				// remove the sent prefix from the queue
				consume_prefix();
			}
			q = q%(num_cpus-1);
		}
		// Send out a message for each process to kill itself after the work is completed
		int kill_message[prefix_length];
		for (i = 0; i < prefix_length; i++) {
			kill_message[i] = -1;
		}
		for (q = 1; q < num_cpus; q++) {
			MPI_Isend(kill_message, prefix_length, MPI_INT, q, 0, MPI_COMM_WORLD, &send);
		}

		// Print final execution time and path
		etime1 = MPI_Wtime();
		printf("\nExecution time: %14.4f seconds\n", etime1-etime0);
		fflush(stdout);

		printf("The minimum distance was %d\n", cur_min);
		printf("The path was\n");
		fflush(stdout);
		for (i = 0; i < num_cities; i++) {
			printf("%d ",min_path[i]);
		fflush(stdout);
		}
		printf("\n");
		fflush(stdout);

	} else {
		int *visited = malloc(sizeof(int)*(num_cities+1));
		for (i = 0; i < num_cities+1; i++) {
			visited[i] = INT_MAX;
		}
		min_path = malloc(sizeof(int)*(num_cities+1));
		MPI_Request recieve, send;
		MPI_Status status;
		/*** First message will just be to signal that this worker is ready ***/
		MPI_Isend(visited, num_cities+1, MPI_INT, 0, 0, MPI_COMM_WORLD, &send);
		int tautology = 1;
		while (tautology == 1){
			/** Recieve more work from the scheduler **/
			MPI_Irecv(visited, prefix_length+1, MPI_INT, 0, 0, MPI_COMM_WORLD, &recieve);
			MPI_Wait(&recieve, &status);
			// Check for kill message
			if (visited[0] == -1) {
				break;
			}
			// update current minimum bound
			cur_min = visited[prefix_length];
			int start_length = 0;
			int last_city, cur_city;
			// Calculate the length of the prefix
			for (i = 1; i < prefix_length; i++) {
				last_city = visited[i-1];
				cur_city = visited[i];
				start_length += city_matrix[last_city][cur_city];
			}
			/** Traverse the subtree of the prefix **/
			traverse(prefix_length, visited, start_length);
			// append the minimum path length to the message
			min_path[num_cities] = cur_min;
			// send the result to the scheduler
			MPI_Isend(min_path, num_cities+1, MPI_INT, 0, 0, MPI_COMM_WORLD, &send);
		}
	}
	MPI_Finalize();
	return 0;
}
/** for 17 cities **/
/**
Travelling salesman distance was 7460
Min path was
0 8 3 4 11 12 16 10 2 6 1 7 5 13 14 15 9 
**/