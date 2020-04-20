#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "NBody.h"
#include "NBodyVisualiser.h"

#define USER_NAME "Xiaowei Zhu" 

int N = 0;	// the number of bodies to simulate
int D = 0;	// the integer dimension of the activity grid
int I = 0;	// the number of simulation iterations
MODE M;	// operation mode
char* input_file = NULL;	// input file with an initial N bodies of data
nbody* bodies = NULL;
float* densities;	// store the density values of the D*D locations
vector* forces;	// force(F) of every body

// declaration of all functions
void print_help();
void parse_parameter(int argc, char* argv[]);
void init_data_by_random();
void load_data_from_file();
void step(void);
void calc_forces_by_serial();
void calc_forces_by_parallel();
void calc_densities();
void update_location_velocity();
void print_bodies();
void print_densities();
char* get_string_in_range(char string[], int start, int end);
char** split(const char* string, char dim, int size);

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
	print_bodies();
	// Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	if (I == 0) {
		printf("\n\nStart simulate by visualisation mode with %s computing...", M == CPU ? "CPU" : "OPENMP");
		initViewer(N, D, M, step);
		setNBodyPositions(bodies);
		//setActivityMapData(densities);
		setHistogramData(densities);
		startVisualisationLoop();
	}
	else {
		printf("\n\nStart simulate by console mode with %s computing...", M == CPU ? "CPU" : "OPENMP");
		// start timer
		double begin_outer = omp_get_wtime();
		for (int i = 0; i < I; i++) {
			double begin = omp_get_wtime();
			step();
			double elapsed = omp_get_wtime() - begin;
			printf("\n\nIteration epoch:%d, Complete in %d seconds %f milliseconds", i, (int)elapsed, 1000 * (elapsed - (int)elapsed));
			/*print_bodies();
			print_densities();*/
		}
		/*print_bodies();
		print_densities();*/
		// stop timer
		double total = omp_get_wtime() - begin_outer;
		printf("\n\nFully Complete in %d seconds %f milliseconds\n", (int)total, 1000 * (total - (int)total));

		// free global variables
		free(bodies);
		free(densities);
		free(forces);
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
	}
	else if (M == OPENMP) {
		calc_forces_by_parallel();
	}
	else {
		fprintf(stderr, "\n%d mode is not supported", M);
	}
	// Calculate density for the D*D locations (density)
	calc_densities();
	// Update location and velocity of n bodies
	update_location_velocity();
}

/**
 * Compute by serial mode
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
 * Compute by paralle mode (using OPENMP)
 */
void calc_forces_by_parallel() {
	// compute the force of every body
	int j;
#pragma omp parallel for default(none) shared(N, bodies, forces) 
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

/**
 * Calculate density for the D*D locations (density)
 */
void calc_densities() {
	// reset the value to zero
	for (int i = 0; i < D * D; i++) {
		densities[i] = 0;
	}
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