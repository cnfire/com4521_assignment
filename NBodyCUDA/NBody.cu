#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "NBody.h"
#include "NBodyVisualiser.h"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#define USER_NAME "Xiaowei Zhu" 

#define THREADS_PER_BLOCK 128 //

int N = 0;	// the number of bodies to simulate
__constant__ int d_N = 0;

int D = 0;	// the integer dimension of the activity grid
__constant__ int d_D = 0;

int I = 0;	// the number of simulation iterations
MODE M;	// operation mode
char* input_file = NULL;	// input file with an initial N bodies of data

nbody* bodies = NULL;
__device__ nbody* d_bodies;
nbody* hd_bodies = nullptr;

float* densities;	// store the density values of the D*D locations (acitvity map)
__device__ float* d_densities;
float* hd_densities = nullptr;

vector* forces;	// force(F) of every body
__device__ vector* d_forces;


// declaration of all functions
void print_help();
void parse_parameter(int argc, char* argv[]);
void init_data_by_random();
void load_data_from_file();
void step(void);
void calc_forces_by_serial();
void calc_forces_by_parallel();

//__global__ void calc_forces_by_cuda(nbody* d_bodies, vector* d_forces);
__global__ void calc_forces_by_cuda();
__global__ void calc_densities_by_cuda();
__global__ void reset_d_densities();
__global__ void update_location_velocity_by_cuda();

void calc_densities();
void calc_densities_by_serial();
void calc_densities_with_critical();
void calc_densities_with_atomic();
void update_location_velocity();
void print_bodies();
void print_densities();
char* get_string_in_range(char string[], int start, int end);
char** split(const char* string, char dim, int size);

void checkCUDAErrors(const char* msg);

int main(int argc, char* argv[]) {
	// Processes the command line arguments
	parse_parameter(argc, argv);
	// Allocate any heap memory
	bodies = (nbody*)malloc(N * sizeof(nbody));
	// initialize all values are zero
	densities = (float*)calloc(D * D, sizeof(float));
	forces = (vector*)malloc(N * sizeof(vector));

	// Depending on program arguments, either read initial data from file or generate random data.
	if (input_file == NULL) {
		printf("\n\nInit n bodies by generating random data...");
		init_data_by_random();
	}
	else {
		printf("\n\nInit n bodies by loading data from file...");
		load_data_from_file();
	}
	//print_bodies();

	// allocate for cuda
	if (M == CUDA) {
		cudaMemcpyToSymbol(d_N, &N, sizeof(int));
		cudaMemcpyToSymbol(d_D, &D, sizeof(int));

		/*cudaMalloc((void**)&d_forces, N * sizeof(vector));
		cudaMalloc((void**)&d_bodies, N * sizeof(nbody));
		cudaMemcpy(d_bodies, bodies, N * sizeof(nbody), cudaMemcpyHostToDevice);
		checkCUDAErrors("cuda malloc");*/

		vector* hd_forces = nullptr;
		cudaMalloc((void**)&hd_forces, N * sizeof(vector));
		checkCUDAErrors("cuda malloc d_forces");
		cudaMemcpyToSymbol(d_forces, &hd_forces, sizeof(hd_forces));

		/*nbody* hd_bodies = nullptr;*/
		cudaMalloc((void**)&hd_bodies, N * sizeof(nbody));
		cudaMemcpyToSymbol(d_bodies, &hd_bodies, sizeof(hd_bodies));
		cudaMemcpy(hd_bodies, bodies, N * sizeof(nbody), cudaMemcpyHostToDevice);
		checkCUDAErrors("cuda malloc d_bodies");

		/*float* hd_densities = nullptr;*/
		cudaMalloc((void**)&hd_densities, D * D * sizeof(float));
		checkCUDAErrors("cuda malloc d_densities");
		cudaMemcpyToSymbol(d_densities, &hd_densities, sizeof(hd_densities));
	}

	// Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	char* mode = M == CPU ? "CPU" : M == OPENMP ? "OPENMP" : "CUDA";
	if (I == 0) {
		printf("\n\nStart simulate by visualisation mode with %s computing...", mode);
		if (M != CUDA) {
			initViewer(N, D, M, step);
			setNBodyPositions(bodies);
			//setActivityMapData(densities);
			setHistogramData(densities);
			startVisualisationLoop();
		}
		else {
			initViewer(N, D, M, step);
			setNBodyPositions(hd_bodies);
			setHistogramData(hd_densities);
			startVisualisationLoop();
		}
	}
	else {
		printf("\n\nStart simulate by console mode with %s computing...", mode);
		// start timer
		double begin_outer = omp_get_wtime();
		for (int i = 0; i < I; i++) {
			double begin = omp_get_wtime();
			step();
			double elapsed = omp_get_wtime() - begin;
			printf("\n\nIteration epoch:%d, Complete in %d seconds %f milliseconds", i, (int)elapsed, 1000 * (elapsed - (int)elapsed));
			//print_bodies();
		}

		//print_densities();
		// stop timer
		double total = omp_get_wtime() - begin_outer;
		printf("\n\nFully Complete in %d seconds %f milliseconds\n", (int)total, 1000 * (total - (int)total));
	}
	return 0;
}

/**
 * Perform the main simulation of the NBody system (Simulation within a single iteration)
 */
void step(void) {
	// compute the force of every body with different way
	if (M == CPU) {
		calc_forces_by_serial();
		// Calculate density for the D*D locations (activity map)
		calc_densities();
		// Update location and velocity of n bodies
		update_location_velocity();
		//print_bodies();
		/*printf("\n");
		for (int i = 0; i < D * D; i++) {
			printf("%f,",densities[i]);
		}
		printf("\n");*/
	}
	else if (M == OPENMP) {
		calc_forces_by_parallel();
		// Calculate density for the D*D locations (activity map)
		calc_densities();
		// Update location and velocity of n bodies
		update_location_velocity();
	}
	else if (M == CUDA) {
		//test_cuda(); exit(0);
		calc_forces_by_cuda << < N / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > ();
		cudaDeviceSynchronize();
		checkCUDAErrors("calc_forces_by_cuda");
		//cudaMemcpyFromSymbol(bodies, d_bodies, N * sizeof(nbody));
		//print_bodies();

		reset_d_densities << <1, 1 >> > ();
		//cudaDeviceSynchronize();
		checkCUDAErrors("reset_d_densities");
		//calc_densities_by_cuda << < N / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > ();
		calc_densities_by_cuda << < 1, 1 >> > ();
		//cudaDeviceSynchronize();
		checkCUDAErrors("calc_densities_by_cuda");

		//update_location_velocity_by_cuda << < N / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > ();
		cudaDeviceSynchronize();
		checkCUDAErrors("update_location_velocity_by_cuda");
		/*for (int i = 0; i < N; i++) {
			printf("\n(%f,%f)", forces[i].x, forces[i].y);
		}*/
	}
	else {
		fprintf(stderr, "\n%d mode is not supported", M);
	}
}


/**
 * Compute fore of every body by serial mode
 */
void calc_forces_by_serial() {
	// compute the force of every body
	for (int j = 0; j < N; j++) {
		nbody* body_j = &bodies[j];
		vector f = { 0, 0 };
		for (int k = 0; k < N; k++) {
			// skip the influence of body on itself
			if (k == j) continue;
			nbody* body_k = &bodies[k];
			vector s1 = { body_k->x - body_j->x, body_k->y - body_j->y };
			vector s2 = { s1.x * body_k->m, s1.y * body_k->m };
			double s3 = pow(pow(s1.x, 2) + pow(s1.y, 2) + pow(SOFTENING, 2), 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		f.x = G * body_j->m * f.x;
		f.y = G * body_j->m * f.y;
		forces[j].x = f.x;
		forces[j].y = f.y;
	}
}

/**
 * Compute fore of every body by paralle mode (using OPENMP)
 */
void calc_forces_by_parallel() {
	// compute the force of every body
	int j;
#pragma omp parallel for default(none) shared(N, bodies, forces) schedule(dynamic)
	for (j = 0; j < N; j++) {
		nbody* body_j = &bodies[j];
		vector f = { 0, 0 };
		for (int k = 0; k < N; k++) {
			// skip the influence of body on itself
			if (k == j) continue;
			nbody* body_k = &bodies[k];
			vector s1 = { body_k->x - body_j->x, body_k->y - body_j->y };
			vector s2 = { s1.x * body_k->m, s1.y * body_k->m };
			double s3 = pow(pow(s1.x, 2) + pow(s1.y, 2) + pow(SOFTENING, 2), 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		f.x = G * body_j->m * f.x;
		f.y = G * body_j->m * f.y;
		forces[j].x = f.x;
		forces[j].y = f.y;
	}
}

void checkCUDAErrors(const char* msg) {
	//cudaError_t err = cudaGetLastError();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "\nCUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		//printf("Error: %s:%d\n,", __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

// implemention of each body
__global__ void calc_forces_by_cuda() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		// compute the force of every body
		nbody* body_i = &d_bodies[i];
		vector f = { 0, 0 };
		for (int k = 0; k < d_N; k++) {
			// skip the influence of body on itself
			if (k == i) continue;
			nbody* body_k = &d_bodies[k];
			vector s1 = { body_k->x - body_i->x, body_k->y - body_i->y };
			vector s2 = { s1.x * body_k->m, s1.y * body_k->m };
			double s3 = powf(s1.x * s1.x + s1.y * s1.y + SOFTENING * SOFTENING, 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		f.x = G * body_i->m * f.x;
		f.y = G * body_i->m * f.y;
		d_forces[i].x = f.x;
		d_forces[i].y = f.y;
		//printf("\n(x:%f,y:%f)", d_forces[i].x, d_forces[i].y);//(x:-0.010679,y:-0.046293)

		// calc the acceleration of body
		vector a = { d_forces[i].x / body_i->m, d_forces[i].y / body_i->m };
		// new velocity
		vector v_new = { body_i->vx + dt * a.x, body_i->vy + dt * a.y };
		// new location
		vector l_new = { body_i->x + dt * v_new.x, body_i->y + dt * v_new.y };
		body_i->x = l_new.x;
		body_i->y = l_new.y;
		body_i->vx = v_new.x;
		body_i->vy = v_new.y;
	}
}



/**
 * Calculate activity map
 */
void calc_densities() {
	// reset the value to zero
	for (int i = 0; i < D * D; i++) {
		densities[i] = 0;
	}
	calc_densities_by_serial();
	//calc_densities_with_critical();
	//calc_densities_with_atomic();
}

/**
 * Calculate density for the D*D locations (activity map) (seri)
 */
void calc_densities_by_serial() {
	for (int i = 0; i < N; i++) {
		nbody* body = &bodies[i];
		double scale = 1.0 / D;
		// x-axis coordinate of D*D locations
		int x = (int)ceil(body->x / scale) - 1;
		// y-axis coordinate of D*D locations
		int y = (int)ceil(body->y / scale) - 1;
		// the index of one dimensional array
		int index = y * D + x;
		densities[index] = densities[index] + 1.0 * D / N;
	}
}


/**
 * Calculate activity map with omp critical
 */
void calc_densities_with_critical() {
	int i;
	double scale = 1.0 / D;
#pragma omp parallel for default(none) shared(densities, scale, N, D, bodies, forces) 
	for (i = 0; i < N; i++) {
		nbody* body = &bodies[i];
		// x-axis coordinate of D*D locations
		int x = (int)ceil(body->x / scale) - 1;
		// y-axis coordinate of D*D locations
		int y = (int)ceil(body->y / scale) - 1;
		// the index of one dimensional array
		int index = y * D + x;
#pragma omp critical 
		densities[index] += 1.0 * D / N;
	}
}

/**
 * Calculate activity map with omp atomic
 */
void calc_densities_with_atomic() {
	int i;
	double scale = 1.0 / D;
#pragma omp parallel for default(none) shared(densities, scale, N, D, bodies, forces) 
	for (i = 0; i < N; i++) {
		nbody* body = &bodies[i];
		// x-axis coordinate of D*D locations
		int x = (int)ceil(body->x / scale) - 1;
		// y-axis coordinate of D*D locations
		int y = (int)ceil(body->y / scale) - 1;
		// the index of one dimensional array
		int index = y * D + x;
#pragma omp atomic
		densities[index] ++;
	}
	for (int i = 0; i < D * D; i++) {
		densities[i] = densities[i] * D / N;
	}
}

__global__ void reset_d_densities() {
	if (blockIdx.x * blockDim.x + threadIdx.x > 0) {
		printf("error: No more than one thread. ");
		return;
	}
	for (int i = 0; i < d_D * d_D; i++) {
		d_densities[i] = 0;
	}
}

__global__ void calc_densities_by_cuda() {
	if (blockIdx.x * blockDim.x + threadIdx.x > 0) {
		printf("error: No more than one thread. ");
		return;
	}
	for (int i = 0; i < d_N; i++) {
		nbody* body = &d_bodies[i];
		double scale = 1.0 / d_D;
		// x-axis coordinate of D*D locations
		int x = (int)ceil(body->x / scale) - 1;
		// y-axis coordinate of D*D locations
		int y = (int)ceil(body->y / scale) - 1;
		// the index of one dimensional array
		int index = y * d_D + x;
		d_densities[index] = d_densities[index] + 1.0 * d_D / d_N;
	}
}


/**
 * Update location and velocity of n bodies
 */
void update_location_velocity() {
	for (int i = 0; i < N; i++) {
		nbody* body = &bodies[i];
		// calc the acceleration of body
		vector a = { forces[i].x / body->m, forces[i].y / body->m };
		// new velocity
		vector v_new = { body->vx + dt * a.x, body->vy + dt * a.y };
		// new location
		vector l_new = { body->x + dt * v_new.x, body->y + dt * v_new.y };
		body->x = l_new.x;
		body->y = l_new.y;
		body->vx = v_new.x;
		body->vy = v_new.y;
	}
}

__global__ void update_location_velocity_by_cuda() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		nbody* body = &d_bodies[i];
		// calc the acceleration of body
		vector a = { d_forces[i].x / body->m, d_forces[i].y / body->m };
		// new velocity
		vector v_new = { body->vx + dt * a.x, body->vy + dt * a.y };
		// new location
		vector l_new = { body->x + dt * v_new.x, body->y + dt * v_new.y };
		body->x = l_new.x;
		body->y = l_new.y;
		body->vx = v_new.x;
		body->vy = v_new.y;
	}
}

void print_help() {
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);
	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation I 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}

/**
 * Convert string to enum
 * @param str target string
 * @return enum data, MODE
 */
MODE str2enum(char* str) {
	if (strcmp(str, "CPU") == 0) return CPU;
	if (strcmp(str, "OPENMP") == 0) return OPENMP;
	if (strcmp(str, "CUDA") == 0) return CUDA;
}

/**
 * Processes the command line arguments
 */
void parse_parameter(int argc, char* argv[]) {
	if (!(argc == 4 || argc == 6 || argc == 8)) {
		fprintf(stderr, "\nInput parameter format is incorrect, please check.\n");
		exit(0);
	}
	N = atoi(argv[1]);
	D = atoi(argv[2]);
	M = str2enum(argv[3]);
	if (argc >= 6) {
		I = atoi(argv[5]);
	}
	if (argc == 8) {
		input_file = argv[7];
	}
	printf("\nParameters{N:%d, D:%d, M:%d, I:%d, f:%s}", N, D, M, I, input_file);
}

/**
 * Generate random float data from 0 ~ 1
 */
float random_float() {
	// Keep two decimal place
	float result = rand() % 100 / (float)100;
	return result;
}

/**
 * Init n bodies by generating random data
 */
void init_data_by_random() {
	for (int i = 0; i < N; i++) {
		bodies[i].x = random_float();
		bodies[i].y = random_float();
		bodies[i].vx = 0.0f;
		bodies[i].vy = 0.0f;
		bodies[i].m = (float)1.0 / N;
	}
}

/**
 * Load all data from local file and init for all bodies
 */
void load_data_from_file() {
	char line[1000];
	FILE* file = NULL;
#pragma warning(suppress : 4996)
	file = fopen(input_file, "rb");
	if (file == NULL) {
		fprintf(stderr, "Error: Could not find file:%s \n", input_file);
		exit(1);
	}
	int i = 0;
	while (!feof(file)) {
		fgets(line, 1000, file);
		// skip the comment line and empty line
		if (line[0] == '#' || isspace(line[0])) {
			continue;
		}
		char** res = split(line, ',', PARAMS_NUM_INPUT);
		bodies[i].x = res[0] == NULL ? random_float() : atof(res[0]);
		bodies[i].y = res[1] == NULL ? random_float() : atof(res[1]);
		bodies[i].vx = res[2] == NULL ? 0 : atof(res[2]);
		bodies[i].vy = res[3] == NULL ? 0 : atof(res[3]);
		bodies[i].m = res[4] == NULL ? 1.0 / N : atof(res[4]);
		i++;
		// free the res memory 
		for (int i = 0; i < PARAMS_NUM_INPUT; i++) {
			// the VS 2019 will display a stupid warrning
			if (res[i] != NULL) {
				free(res[i]);
			}
		}
		free(res);
	}
	fclose(file);
	if (i != N) {
		fprintf(stderr, "\nN is %d and the number of body in csv file is %d, not equal!", N, i);
		exit(0);
	}
}

/**
 * Print all values of n bodies for testing
 */
void print_bodies() {
	for (int i = 0; i < N; i++) {
		printf("\nx:%f, y:%f, vx:%f, vy:%f, m:%f", bodies[i].x, bodies[i].y, bodies[i].vx, bodies[i].vy, bodies[i].m);
	}
	printf("\n");
}

/**
 * Print all values of densities for testing
 */
void print_densities() {
	printf("\nShow densities info:");
	for (int i = 0; i < D * D; i++) {
		printf("\n%d:%f", i, densities[i]);
	}
}

/**
 * Get sub-string from start and end index
 * @param string target string
 * @param start start index
 * @param end end index
 * @return sub-string
 */
char* get_string_in_range(char string[], int start, int end) {
	if (start == end) {
		return NULL;
	}
	char* res = (char*)calloc(end - start, sizeof(char));
	int i = 0, j = 0;
	while (string[i] != '\0') {
		if (i >= start && i < end) {
			// filter space char
			if (!isspace(string[i])) {
				res[j++] = string[i];
			}
		}
		i++;
	}
	return strlen(res) == 0 ? NULL : res;
}

/**
 * Split a string to a sub-string list by delimiter
 * @param string target string
 * @param dim delimiter
 * @param size size of sub-string list
 * @return sub-string list (pointer to pointers)
 */
char** split(const char* string, char dim, int size) {
	int len = strlen(string);
	char* new_string = (char*)calloc(len + 2, sizeof(char));
	strcpy(new_string, string);
	new_string[strlen(new_string)] = dim;
	char** res = (char**)malloc(size * sizeof(char*));
	int i = 0, j = 0, start = 0, end = 0;
	while (new_string[i] != '\0') {
		if (new_string[i] == dim) {
			end = i;
			char* item = get_string_in_range(new_string, start, end);
			start = end + 1;
			res[j] = item;
			j++;
		}
		i++;
	}
	free(new_string);
	return res;
}