﻿#include <stdio.h>
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
__constant__ int d_N = 0; // N of device
int D = 0;	// the integer dimension of the activity grid
__constant__ int d_D = 0; // D of device
int I = 0;	// the number of simulation iterations
MODE M;	// operation mode
char* input_file = NULL;	// input file with an initial N bodies of data
nbody* bodies = NULL;	// AOS structure of n bodies
nbody_soa* bodies_soa = NULL; // SOA structure of n bodies
__device__ nbody_soa* d_bodies_soa = NULL;
nbody_soa* hd_bodies_soa = NULL;
float* x_soa, * y_soa, * vx_soa, * vy_soa, * m_soa;

float* densities = NULL;	// store the density values of the D*D locations (acitvity map)
__device__ float* d_densities = NULL;
float* hd_densities = NULL;

vector* accelerations = NULL;	// acceleration of every body
__device__ vector* d_accelerations = NULL;
vector* hd_accelerations = NULL;

int N_FLOAT_SIZE = 0; // N * sizeof(float)

int show_verbose = 0; // if show the cost time for each iteration. (0: false, >=1: true)

// cuda texure variables 
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
void calc_accelerations_by_serial();
void calc_accelerations_by_openmp();
void update_location_velocity();
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

__global__ void calc_accelerations_by_cuda_with_global();
__global__ void calc_accelerations_by_cuda_with_texture();
__global__ void calc_accelerations_by_cuda_with_shared();
__global__ void calc_accelerations_by_cuda_with_readonly(nbody_soa const* __restrict__ nbodies);
__global__ void update_bodies_by_cuda_with_global();
__global__ void update_bodies_by_cuda_with_texture();
__global__ void update_bodies_by_cuda_with_global(nbody_soa const* __restrict__ nbodies, vector const* __restrict__ accelerations);

int main(int argc, char* argv[]) {
	// Processes the command line arguments
	parse_parameter(argc, argv);
	// Allocate any heap memory
	bodies = (nbody*)malloc(N * sizeof(nbody));

	// Initialize all values are zero
	densities = (float*)calloc(D * D, sizeof(float));
	accelerations = (vector*)malloc(N * sizeof(vector));

	N_FLOAT_SIZE = N * sizeof(float);

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

	// Allocation for cuda
	if (M == CUDA) {
		// Set value for cuda constant variables
		cudaMemcpyToSymbol(d_N, &N, sizeof(int));
		cudaMemcpyToSymbol(d_D, &D, sizeof(int));

		// Set for accelerations
		cudaMalloc((void**)&hd_accelerations, N * sizeof(vector));
		checkCUDAErrors("cuda malloc d_accelerations");
		cudaMemcpyToSymbol(d_accelerations, &hd_accelerations, sizeof(hd_accelerations));

		// Set for densities
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

/**
 * Perform simulation for n bodies.
 */
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
			if (!show_verbose) {
				printf("\n\nIteration epoch:%d...", i, i);
				step();
			}
			else {
				double begin = omp_get_wtime();
				step();
				double elapsed = omp_get_wtime() - begin;
				printf("\n\nIteration epoch:%d, Complete in %d seconds %f milliseconds", i, (int)elapsed, 1000 * (elapsed - (int)elapsed));
			}
		}
		// stop timer
		double total = omp_get_wtime() - begin_outer;
		printf("\nFully Complete in %d seconds %f milliseconds", (int)total, 1000 * (total - (int)total));
	}
}

/**
 * Allocation for bodies_soa in device mainly.
 */
void set_bodies_soa() {
	bodies_soa = (nbody_soa*)malloc(sizeof(nbody_soa));
	bodies_soa->m = (float*)malloc(N_FLOAT_SIZE);
	bodies_soa->x = (float*)malloc(N_FLOAT_SIZE); bodies_soa->y = (float*)malloc(N_FLOAT_SIZE);
	bodies_soa->vx = (float*)malloc(N_FLOAT_SIZE); bodies_soa->vy = (float*)malloc(N_FLOAT_SIZE);
	for (int i = 0; i < N; i++) {
		bodies_soa->m[i] = bodies[i].m;
		bodies_soa->x[i] = bodies[i].x; bodies_soa->y[i] = bodies[i].y;
		bodies_soa->vx[i] = bodies[i].vx; bodies_soa->vy[i] = bodies[i].vy;
	}

	cudaMalloc((void**)&hd_bodies_soa, sizeof(nbody_soa));
	cudaMemcpyToSymbol(d_bodies_soa, &hd_bodies_soa, sizeof(hd_bodies_soa));
	cudaMemcpy(hd_bodies_soa, bodies_soa, sizeof(bodies_soa), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&x_soa, N_FLOAT_SIZE); cudaMemcpy(x_soa, bodies_soa->x, N_FLOAT_SIZE, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->x), &x_soa, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&y_soa, N_FLOAT_SIZE); cudaMemcpy(y_soa, bodies_soa->y, N_FLOAT_SIZE, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->y), &y_soa, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&vx_soa, N_FLOAT_SIZE); cudaMemcpy(vx_soa, bodies_soa->vx, N_FLOAT_SIZE, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->vx), &vx_soa, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&vy_soa, N_FLOAT_SIZE); cudaMemcpy(vy_soa, bodies_soa->vy, N_FLOAT_SIZE, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->vy), &vy_soa, sizeof(float*), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&m_soa, N_FLOAT_SIZE); cudaMemcpy(m_soa, bodies_soa->m, N_FLOAT_SIZE, cudaMemcpyHostToDevice); cudaMemcpy(&(hd_bodies_soa->m), &m_soa, sizeof(float*), cudaMemcpyHostToDevice);
	checkCUDAErrors("cuda allocate bodies_soa");
}

/**
 * Free memroies.
 */
void cleanup() {
	printf("\nClean memory...\n");
	free(accelerations);
	free(bodies);
	free(densities);

	cudaFree(x_soa); cudaFree(y_soa); cudaFree(vx_soa); cudaFree(vy_soa); cudaFree(m_soa);
	cudaFree(hd_bodies_soa);
	cudaFree(hd_densities);
	cudaFree(hd_accelerations);
	checkCUDAErrors("CUDA cleanup");
}


/**
 * Perform the main simulation of the NBody system (Simulation within a single iteration)
 */
void step(void) {
	if (M == CPU) {
		// Compute the acceleration of every body with serial way
		calc_accelerations_by_serial();
		// Update locations and velocities
		update_location_velocity();
		// Calculate density for the D*D locations (activity map)
		calc_densities();
	}
	else if (M == OPENMP) {
		calc_accelerations_by_openmp();
		update_location_velocity();
		calc_densities();
	}
	else if (M == CUDA) {
		float time; cudaEvent_t start, stop;
		if (show_verbose) {
			cudaEventCreate(&start); cudaEventCreate(&stop);
			cudaEventRecord(start, 0);
		}
		cudaMemset(hd_densities, 0, size_t(D * D) * sizeof(float));
		checkCUDAErrors("cudaMemset");

		CUDA_OPT_MODE opt_mode = GLOBAL;
		if (N <= THREADS_PER_BLOCK) {
			opt_mode = TEXTURE;
		}
		else {
			opt_mode = READ_ONLY;
		}

		opt_mode = GLOBAL;
		//CUDA_OPT_MODE opt_mode = TEXTURE;
		//CUDA_OPT_MODE opt_mode = GLOBAL;

		int BLOCKS_PER_GRID = N / THREADS_PER_BLOCK + 1;

		switch (opt_mode) {
		case GLOBAL:
			calc_accelerations_by_cuda_with_global << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > ();
			checkCUDAErrors("calc_accelerations_by_cuda");
			update_bodies_by_cuda_with_global << < BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > ();
			checkCUDAErrors("update_bodies_by_cuda_with_global");
			break;
		case SHARED:
			calc_accelerations_by_cuda_with_shared << < BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > ();
			checkCUDAErrors("calc_accelerations_by_cuda_with_shared");
			update_bodies_by_cuda_with_global << < BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > ();
			checkCUDAErrors("update_bodies_by_cuda_with_global");
			break;
		case TEXTURE:
			cudaBindTexture(0, tex_x, x_soa, N_FLOAT_SIZE); cudaBindTexture(0, tex_y, y_soa, N_FLOAT_SIZE);
			cudaBindTexture(0, tex_vx, vx_soa, N_FLOAT_SIZE); cudaBindTexture(0, tex_vy, vy_soa, N_FLOAT_SIZE);
			cudaBindTexture(0, tex_m, m_soa, N_FLOAT_SIZE);
			checkCUDAErrors("cudaBindTexture");
			calc_accelerations_by_cuda_with_texture << < BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > ();
			checkCUDAErrors("calc_accelerations_by_cuda_with_texture");
			update_bodies_by_cuda_with_texture << < BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > ();
			checkCUDAErrors("update_bodies_by_cuda_with_texture");
			cudaUnbindTexture(tex_x); cudaUnbindTexture(tex_y);
			cudaUnbindTexture(tex_vx); cudaUnbindTexture(tex_vy);
			cudaUnbindTexture(tex_m);
			break;
		case READ_ONLY:
			calc_accelerations_by_cuda_with_readonly << < BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (hd_bodies_soa);
			checkCUDAErrors("calc_accelerations_by_cuda_with_readonly");
			update_bodies_by_cuda_with_global << < BLOCKS_PER_GRID, THREADS_PER_BLOCK >> > (hd_bodies_soa, hd_accelerations);
			checkCUDAErrors("update_bodies_by_cuda_with_global");
			break;
		default:
			break;
		}

		if (show_verbose) {
			cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop);
			cudaEventDestroy(start); cudaEventDestroy(stop);
			printf("\nCUDA execution time was %f ms", time);
		}
		cudaDeviceSynchronize();
	}
}


/**
 * Compute accelerations of every body by serial mode
 */
void calc_accelerations_by_serial() {
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

		// calc the acceleration of body
		accelerations[j].x = f.x / body_j->m;
		accelerations[j].y = f.y / body_j->m;
	}
}

/**
 * Compute accelerations of every body by paralle mode (using OPENMP)
 */
void calc_accelerations_by_openmp() {
	// compute the force of every body
	int j;
#pragma omp parallel for default(none) shared(N, bodies, accelerations) schedule(dynamic)
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

		// calc the acceleration of body
		accelerations[j].x = f.x / body_j->m;
		accelerations[j].y = f.y / body_j->m;
	}
}

/**
 * Update location and velocity of n bodies (Using in CPU and OPENMP mode)
 */
void update_location_velocity() {
	for (int i = 0; i < N; i++) {
		nbody* body = &bodies[i];
		// new velocity
		vector v_new = { body->vx + dt * accelerations[i].x, body->vy + dt * accelerations[i].y };
		// new location
		vector l_new = { body->x + dt * v_new.x, body->y + dt * v_new.y };
		body->x = l_new.x;
		body->y = l_new.y;
		body->vx = v_new.x;
		body->vy = v_new.y;
	}
}

/**
 * Calculate accelerations by CUDA with global memory
 */
__global__ void calc_accelerations_by_cuda_with_global() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
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
		// calc the acceleration of body
		d_accelerations[i].x = G * f.x;
		d_accelerations[i].y = G * f.y;
	}
}

/**
 * Update bodies and densities by CUDA with global memory
 */
__global__ void update_bodies_by_cuda_with_global() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		// new velocity
		vector v_new = { d_bodies_soa->vx[i] + dt * d_accelerations[i].x, d_bodies_soa->vy[i] + dt * d_accelerations[i].y };
		// new location
		vector l_new = { d_bodies_soa->x[i] + dt * v_new.x, d_bodies_soa->y[i] + dt * v_new.y };
		d_bodies_soa->x[i] = l_new.x;
		d_bodies_soa->y[i] = l_new.y;
		d_bodies_soa->vx[i] = v_new.x;
		d_bodies_soa->vy[i] = v_new.y;

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

/**
 * Calculate accelerations by CUDA with texture memory
 */
__global__ void calc_accelerations_by_cuda_with_texture() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		// compute the force of every body
		vector f = { 0, 0 };
		for (int k = 0; k < d_N; k++) {
			// skip the influence of body on itself
			if (k == i) continue;
			vector s1 = { tex1Dfetch(tex_x, k) - tex1Dfetch(tex_x, i), tex1Dfetch(tex_y, k) - tex1Dfetch(tex_y, i) };
			vector s2 = { s1.x * tex1Dfetch(tex_m, k), s1.y * tex1Dfetch(tex_m, k) };
			double s3 = powf(s1.x * s1.x + s1.y * s1.y + SOFTENING * SOFTENING, 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		// calc the acceleration of body
		d_accelerations[i].x = G * f.x;
		d_accelerations[i].y = G * f.y;
	}
}

/**
 * Update bodies and densities by CUDA with texture memory
 */
__global__ void update_bodies_by_cuda_with_texture() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		// new velocity
		vector acc = d_accelerations[i];
		vector v_new = { tex1Dfetch(tex_vx, i) + dt * acc.x, tex1Dfetch(tex_vy, i) + dt * acc.y };
		// new location
		vector l_new = { tex1Dfetch(tex_x, i) + dt * v_new.x,  tex1Dfetch(tex_y, i) + dt * v_new.y };
		d_bodies_soa->x[i] = l_new.x;
		d_bodies_soa->y[i] = l_new.y;
		d_bodies_soa->vx[i] = v_new.x;
		d_bodies_soa->vy[i] = v_new.y;

		double scale = 1.0 / d_D;
		// x-axis coordinate of D*D locations
		int x = (int)ceil(tex1Dfetch(tex_x, i) / scale) - 1;
		// y-axis coordinate of D*D locations
		int y = (int)ceil(tex1Dfetch(tex_y, i) / scale) - 1;
		// the index of one dimensional array
		int index = y * d_D + x;
		//d_densities[index] = d_densities[index] + 1.0 * d_D / d_N;
		atomicAdd(&d_densities[index], (float)(1.0 * d_D / d_N));
	}
}

/**
 * Calculate accelerations by CUDA with shared memory(when N <= BLOCK_SIZE)
 */
__global__ void calc_accelerations_by_cuda_with_shared() {
	if (d_N > THREADS_PER_BLOCK) {
		fprintf(stderr, "\nIn this shared case, N is no more than %d.\n", THREADS_PER_BLOCK);
		exit(EXIT_FAILURE);
	}
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float x_arr[THREADS_PER_BLOCK];
	__shared__ float y_arr[THREADS_PER_BLOCK];
	__shared__ float m_arr[THREADS_PER_BLOCK];
	int idx = threadIdx.x;
	if (i < d_N) {
		x_arr[idx] = d_bodies_soa->x[i];
		y_arr[idx] = d_bodies_soa->y[i];
		m_arr[idx] = d_bodies_soa->m[i];
		__syncthreads();

		// compute the force of every body
		vector f = { 0, 0 };
		for (int k = 0; k < d_N; k++) {
			// skip the influence of body on itself
			if (k == i) continue;
			vector s1 = { x_arr[k] - x_arr[i],y_arr[k] - y_arr[i] };
			vector s2 = { s1.x * m_arr[k], s1.y * m_arr[k] };
			double s3 = powf(s1.x * s1.x + s1.y * s1.y + SOFTENING * SOFTENING, 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		// calc the acceleration of body
		d_accelerations[i].x = G * f.x;
		d_accelerations[i].y = G * f.y;
	}
}

/**
 * Calculate accelerations by CUDA with readonly memory
 */
__global__ void calc_accelerations_by_cuda_with_readonly(nbody_soa const* __restrict__ nbodies) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		// compute the force of every body
		vector f = { 0, 0 };
		for (int k = 0; k < d_N; k++) {
			// skip the influence of body on itself
			if (k == i) continue;
			vector s1 = { nbodies->x[k] - nbodies->x[i], nbodies->y[k] - nbodies->y[i] };
			vector s2 = { s1.x * nbodies->m[k], s1.y * nbodies->m[k] };
			double s3 = powf(s1.x * s1.x + s1.y * s1.y + SOFTENING * SOFTENING, 1.5);
			f.x = f.x + s2.x / s3;
			f.y = f.y + s2.y / s3;
		}
		// calc the acceleration of body
		d_accelerations[i].x = G * f.x;
		d_accelerations[i].y = G * f.y;
	}
}

/**
 * Update bodies and densities by CUDA with read-only memory
 */
__global__ void update_bodies_by_cuda_with_readonly(nbody_soa const* __restrict__ nbodies, vector const* __restrict__ accelerations) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < d_N) {
		// new velocity 
		vector acc = accelerations[i];
		vector v_new = { nbodies->vx[i] + dt * acc.x, d_bodies_soa->vy[i] + dt * acc.y };
		// new location
		vector l_new = { nbodies->x[i] + dt * v_new.x, nbodies->y[i] + dt * v_new.y };
		nbodies->x[i] = l_new.x;
		nbodies->y[i] = l_new.y;
		nbodies->vx[i] = v_new.x;
		nbodies->vy[i] = v_new.y;

		double scale = 1.0 / d_D;
		// x-axis coordinate of D*D locations
		int x = (int)ceil(nbodies->x[i] / scale) - 1;
		// y-axis coordinate of D*D locations
		int y = (int)ceil(nbodies->y[i] / scale) - 1;
		// the index of one dimensional array
		int index = y * d_D + x;
		//d_densities[index] = d_densities[index] + 1.0 * d_D / d_N;
		atomicAdd(&d_densities[index], (float)(1.0 * d_D / d_N));
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
#pragma omp parallel for default(none) shared(densities, scale, N, D, bodies)
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
#pragma omp parallel for default(none) shared(densities, scale, N, D, bodies)
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
		exit(EXIT_FAILURE);
	}
	N = atoi(argv[1]);
	D = atoi(argv[2]);
	M = str2enum(argv[3]);
	if (M == UNKNOWN) {
		fprintf(stderr, "\nUnknown mode, please check!.\n");
		exit(EXIT_FAILURE);
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
		exit(EXIT_FAILURE);
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
		exit(EXIT_SUCCESS);
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

/**
 * Check cuda errors
 */
void checkCUDAErrors(const char* msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "\nCUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}