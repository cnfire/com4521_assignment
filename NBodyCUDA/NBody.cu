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

#include "cuda_texture_types.h"
#include "texture_fetch_functions.hpp"

#define USER_NAME "Xiaowei Zhu"

#define THREADS_PER_BLOCK 128 

int N = 0;	// the number of bodies to simulate
__constant__ int d_N = 0;

int D = 0;	// the integer dimension of the activity grid
__constant__ int d_D = 0;

int I = 0;	// the number of simulation iterations
MODE M;	// operation mode
char* input_file = NULL;	// input file with an initial N bodies of data

nbody* bodies = NULL;
nbody_soa* bodies_soa = NULL;
__device__ nbody_soa* d_bodies_soa = NULL;
nbody_soa* hd_bodies_soa = NULL;
float* x_soa, * y_soa, * vx_soa, * vy_soa, * m_soa;

float* densities = NULL;	// store the density values of the D*D locations (acitvity map)
__device__ float* d_densities = NULL;
float* hd_densities = NULL;

vector* forces = NULL;	// force(F) of every body
__device__ vector* d_forces = NULL;

texture<float, 1, cudaReadModeElementType>tex_x;
texture<float, 1, cudaReadModeElementType>tex_y;
texture<float, 1, cudaReadModeElementType>tex_vx;
texture<float, 1, cudaReadModeElementType>tex_vy;
texture<float, 1, cudaReadModeElementType>tex_m;


// declaration of all functions
void print_help();
void parse_parameter(int argc, char* argv[]);
void init_data_by_random();
void load_data_from_file();
void step(void);
void update_body_by_serial();
void update_body_by_openmp();
void set_bodies_soa();
void calc_densities();
void calc_densities_by_serial();
void calc_densities_with_critical();
void calc_densities_with_atomic();
void print_bodies();
void print_densities();
char* get_string_in_range(char string[], int start, int end);
char** split(const char* string, char dim, int size);
void checkCUDAErrors(const char* msg);
void cleanup();
void perform_simulation();

__global__ void update_body_by_cuda();
__global__ void update_body_by_cuda_with_texture();
__global__ void calc_densities_by_cuda();
__global__ void reset_d_densities();

int main(int argc, char* argv[]) {
	// Processes the command line arguments
	parse_parameter(argc, argv);
	// Allocate any heap memory
	bodies = (nbody*)malloc(N * sizeof(nbody));

	// Initialize all values are zero
	densities = (float*)calloc(D * D, sizeof(float));
	forces = (vector*)malloc(N * sizeof(vector));

	// Init data for n bodies, read initial data from file or generate random data.
	if (input_file == NULL) {
		printf("\n\nInit n bodies by generating random data...");
		init_data_by_random();
	}
	else {
		printf("\n\nInit n bodies by loading data from file...");
		load_data_from_file();
	}
	//print_bodies();

	// Allocate for cuda
	if (M == CUDA) {
		// Set value for cuda constant variables
		cudaMemcpyToSymbol(d_N, &N, sizeof(int));
		cudaMemcpyToSymbol(d_D, &D, sizeof(int));

		// Set for forces
		vector* hd_forces = NULL;
		cudaMalloc((void**)&hd_forces, N * sizeof(vector));
		checkCUDAErrors("cuda malloc d_forces");
		cudaMemcpyToSymbol(d_forces, &hd_forces, sizeof(hd_forces));

		cudaMalloc((void**)&hd_densities, D * D * sizeof(float));
		checkCUDAErrors("cuda malloc d_densities");
		cudaMemcpyToSymbol(d_densities, &hd_densities, sizeof(hd_densities));

		// Set for bodies_soa
		set_bodies_soa();
	}

	// Start to simulate
	perform_simulation();

	// Reclaiming memory
	cleanup();
	return 0;
}

void perform_simulation() {
	// Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	char* mode = M == CPU ? "CPU" : M == OPENMP ? "OPENMP" : "CUDA";
	if (I == 0) {
		printf("\n\nStart simulate by visualisation mode with %s computing...", mode);
		if (M != CUDA) {
			initViewer(N, D, M, step);
			setNBodyPositions(bodies);
			setHistogramData(densities);
			startVisualisationLoop();
		}
		else {
			initViewer(N, D, M, step);
			setNBodyPositions2f(x_soa, y_soa);
			setHistogramData(hd_densities);
			startVisualisationLoop();
		}
	}
	else {
		printf("\n\nStart simulate by console mode with %s computing...", mode);
		// start timer
		double begin_outer = omp_get_wtime();
		for (int i = 0; i < I; i++) {
			/*printf("\n\nIteration epoch:%d...", i, i);
			step();*/
			double begin = omp_get_wtime();
			step();
			double elapsed = omp_get_wtime() - begin;
			printf("\n\nIteration epoch:%d, Complete in %d seconds %f milliseconds", i, (int)elapsed, 1000 * (elapsed - (int)elapsed));

		}
		// stop timer
		double total = omp_get_wtime() - begin_outer;
		printf("\n\nFully Complete in %d seconds %f milliseconds\n", (int)total, 1000 * (total - (int)total));
	}
}

void set_bodies_soa() {
	bodies_soa = (nbody_soa*)malloc(sizeof(nbody_soa));
	bodies_soa->m = (float*)malloc(N * sizeof(float));
	bodies_soa->x = (float*)malloc(N * sizeof(float)); bodies_soa->y = (float*)malloc(N * sizeof(float));
	bodies_soa->vx = (float*)malloc(N * sizeof(float)); bodies_soa->vy = (float*)malloc(N * sizeof(float));
	for (int i = 0; i < N; i++) {
		bodies_soa->m[i] = bodies[i].m;
		bodies_soa->x[i] = bodies[i].x; bodies_soa->y[i] = bodies[i].y;
		bodies_soa->vx[i] = bodies[i].vx; bodies_soa->vy[i] = bodies[i].vy;
	}

	cudaMalloc((void**)&hd_bodies_soa, sizeof(nbody_soa));
	cudaMemcpyToSymbol(d_bodies_soa, &hd_bodies_soa, sizeof(hd_bodies_soa));
	cudaMemcpy(hd_bodies_soa, bodies_soa, sizeof(bodies_soa), cudaMemcpyHostToDevice);
	int size_b = N * sizeof(float);
	cudaMalloc((void**)&x_soa, size_b); cudaMemcpy(x_soa, bodies_soa->x, size_b, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->x), &x_soa, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&y_soa, size_b); cudaMemcpy(y_soa, bodies_soa->y, size_b, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->y), &y_soa, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&vx_soa, size_b); cudaMemcpy(vx_soa, bodies_soa->vx, size_b, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->vx), &vx_soa, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&vy_soa, size_b); cudaMemcpy(vy_soa, bodies_soa->vy, size_b, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->vy), &vy_soa, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&m_soa, size_b); cudaMemcpy(m_soa, bodies_soa->m, size_b, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->m), &m_soa, sizeof(float*), cudaMemcpyHostToDevice);
	checkCUDAErrors("cuda malloc d_bodies_soa");
}

void cleanup() {
	printf("\nClean memory...\n");
	free(bodies);
	free(densities);
	free(forces);
	cudaFree(x_soa); cudaFree(y_soa); cudaFree(vx_soa); cudaFree(vy_soa); cudaFree(m_soa);
	cudaFree(hd_bodies_soa);
	cudaFree(hd_densities);
	checkCUDAErrors("CUDA cleanup");
}


/**
 * Perform the main simulation of the NBody system (Simulation within a single iteration)
 */
void step(void) {
	// compute the force of every body with different way
	if (M == CPU) {
		update_body_by_serial();
		// Calculate density for the D*D locations (activity map)
		calc_densities();
	}
	else if (M == OPENMP) {
		update_body_by_openmp();
		calc_densities();
	}
	else if (M == CUDA) {
		float time;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);cudaEventCreate(&stop);cudaEventRecord(start, 0);
		//int using_texture = N > 10000000 ? 1 : 0;
		int using_texture = 1;


		if (!using_texture) {
			update_body_by_cuda << < N / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > ();
			checkCUDAErrors("update_body_by_cuda");
		}
		else {
			printf("\nUsing texture to optmize\n");
			int size_f = N * sizeof(float);
			cudaBindTexture(0, tex_x, x_soa, size_f); cudaBindTexture(0, tex_y, y_soa, size_f);
			cudaBindTexture(0, tex_vx, vx_soa, size_f); cudaBindTexture(0, tex_vy, vy_soa, size_f);
			cudaBindTexture(0, tex_m, m_soa, size_f);
			update_body_by_cuda_with_texture << < N / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > ();
			checkCUDAErrors("update_body_by_cuda_with_texture");
			cudaUnbindTexture(tex_x); cudaUnbindTexture(tex_y);
			cudaUnbindTexture(tex_vx); cudaUnbindTexture(tex_vy);
			cudaUnbindTexture(tex_m);
		}

		reset_d_densities << <1, 1 >> > ();
		checkCUDAErrors("reset_d_densities");
		calc_densities_by_cuda << < 1, 1 >> > ();
		cudaDeviceSynchronize();
		checkCUDAErrors("calc_densities_by_cuda");

		cudaEventRecord(stop, 0);cudaEventSynchronize(stop);cudaEventElapsedTime(&time, start, stop);
		printf("\nExecution time was %f ms\n", time);
		cudaEventDestroy(start);cudaEventDestroy(stop);
	}
}


/**
 * Compute fore of every body by serial mode
 */
void update_body_by_serial() {
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
			double s3 = pow(s1.x * s1.x + s1.y * s1.y + SOFTENING * SOFTENING, 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		f.x = G * body_j->m * f.x;
		f.y = G * body_j->m * f.y;
		forces[j].x = f.x;
		forces[j].y = f.y;

		// Update location and velocity of n bodies
		// calc the acceleration of body
		vector a = { forces[j].x / body_j->m, forces[j].y / body_j->m };
		// new velocity
		vector v_new = { body_j->vx + dt * a.x, body_j->vy + dt * a.y };
		// new location
		vector l_new = { body_j->x + dt * v_new.x, body_j->y + dt * v_new.y };
		body_j->x = l_new.x;
		body_j->y = l_new.y;
		body_j->vx = v_new.x;
		body_j->vy = v_new.y;
	}
}

/**
 * Compute fore of every body by paralle mode (using OPENMP)
 */
void update_body_by_openmp() {
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
			double s3 = pow(s1.x * s1.x + s1.y * s1.y + SOFTENING * SOFTENING, 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		f.x = G * body_j->m * f.x;
		f.y = G * body_j->m * f.y;
		forces[j].x = f.x;
		forces[j].y = f.y;

		// Update location and velocity of n bodies
		// calc the acceleration of body
		vector a = { forces[j].x / body_j->m, forces[j].y / body_j->m };
		// new velocity
		vector v_new = { body_j->vx + dt * a.x, body_j->vy + dt * a.y };
		// new location
		vector l_new = { body_j->x + dt * v_new.x, body_j->y + dt * v_new.y };
		body_j->x = l_new.x;
		body_j->y = l_new.y;
		body_j->vx = v_new.x;
		body_j->vy = v_new.y;
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

__global__ void update_body_by_cuda() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		//printf("\n(x:%f,y:%f,vx:%f,vy:%f,m:%f)", d_bodies_soa->x[i], d_bodies_soa->y[i], d_bodies_soa->vx[i], d_bodies_soa->vy[i], d_bodies_soa->m[i]);
		// compute the force of every body
		vector f = { 0, 0 };
		for (int k = 0; k < d_N; k++) {
			// skip the influence of body on itself
			if (k == i) continue;
			vector s1 = { d_bodies_soa->x[k] - d_bodies_soa->x[i], d_bodies_soa->y[k] - d_bodies_soa->y[i] };
			vector s2 = { s1.x * d_bodies_soa->m[k], s1.y * d_bodies_soa->m[k] };
			double s3 = powf(s1.x * s1.x + s1.y * s1.y + SOFTENING * SOFTENING, 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		f.x = G * d_bodies_soa->m[i] * f.x;
		f.y = G * d_bodies_soa->m[i] * f.y;
		d_forces[i].x = f.x;
		d_forces[i].y = f.y;

		// calc the acceleration of body
		vector a = { d_forces[i].x / d_bodies_soa->m[i], d_forces[i].y / d_bodies_soa->m[i] };
		// new velocity
		vector v_new = { d_bodies_soa->vx[i] + dt * a.x, d_bodies_soa->vy[i] + dt * a.y };
		// new location
		vector l_new = { d_bodies_soa->x[i] + dt * v_new.x, d_bodies_soa->y[i] + dt * v_new.y };
		d_bodies_soa->x[i] = l_new.x;
		d_bodies_soa->y[i] = l_new.y;
		d_bodies_soa->vx[i] = v_new.x;
		d_bodies_soa->vy[i] = v_new.y;
	}
}

__global__ void update_body_by_cuda_with_texture() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		//printf("\n(x:%f,y:%f,vx:%f,vy:%f,m:%f)", d_bodies_soa->x[i], d_bodies_soa->y[i], d_bodies_soa->vx[i], d_bodies_soa->vy[i], d_bodies_soa->m[i]);
		// compute the force of every body
		//printf("\n:tex1Dfetch:%f,real:%f", tex1Dfetch(tex_x, i), d_bodies_soa->x[i]);
		vector f = { 0, 0 };
		for (int k = 0; k < d_N; k++) {
			// skip the influence of body on itself
			if (k == i) continue;
			vector s1 = { tex1Dfetch(tex_x, k) - tex1Dfetch(tex_x, i), tex1Dfetch(tex_y, k) - tex1Dfetch(tex_y, i) };
			vector s2 = { s1.x * d_bodies_soa->m[k], s1.y * d_bodies_soa->m[k] };
			double s3 = powf(s1.x * s1.x + s1.y * s1.y + SOFTENING * SOFTENING, 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		f.x = G * d_bodies_soa->m[i] * f.x;
		f.y = G * d_bodies_soa->m[i] * f.y;
		d_forces[i].x = f.x;
		d_forces[i].y = f.y;

		// calc the acceleration of body
		vector a = { d_forces[i].x / d_bodies_soa->m[i], d_forces[i].y / d_bodies_soa->m[i] };
		// new velocity
		vector v_new = { d_bodies_soa->vx[i] + dt * a.x, d_bodies_soa->vy[i] + dt * a.y };
		// new location
		vector l_new = { tex1Dfetch(tex_x, i) + dt * v_new.x,  tex1Dfetch(tex_y, i) + dt * v_new.y };
		d_bodies_soa->x[i] = l_new.x;
		d_bodies_soa->y[i] = l_new.y;
		d_bodies_soa->vx[i] = v_new.x;
		d_bodies_soa->vy[i] = v_new.y;
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
		printf("\nerror: No more than one thread.");
		return;
	}
	for (int i = 0; i < d_N; i++) {
		double scale = 1.0 / d_D;
		// x-axis coordinate of D*D locations
		int x = (int)ceil(d_bodies_soa->x[i] / scale) - 1;
		// y-axis coordinate of D*D locations
		int y = (int)ceil(d_bodies_soa->y[i] / scale) - 1;
		// the index of one dimensional array
		int index = y * d_D + x;
		//d_densities[index] = d_densities[index] + 1.0 * d_D / d_N;
		atomicAdd(&d_densities[index], (float)(1.0 * d_D / d_N));
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
	return UNKNOWN;
}

/**
 * Processes the command line arguments
 */
void parse_parameter(int argc, char* argv[]) {
	if (!(argc == 4 || argc == 6 || argc == 8)) {
		fprintf(stderr, "\nInput parameter format is incorrect, please check.\n");
		exit(1);
	}
	N = atoi(argv[1]);
	D = atoi(argv[2]);
	M = str2enum(argv[3]);
	if (M == UNKNOWN) {
		fprintf(stderr, "\nUnknown mode, please check!.\n");
		exit(1);
	}
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