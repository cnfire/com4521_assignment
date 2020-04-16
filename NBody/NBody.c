#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "NBody.h"
#include "NBodyVisualiser.h"

#define USER_NAME "Xiaowei Zhu"        //replace with your username

// dimension of parameters
#define P  5
//delimiter of parameters in csv file
#define DELIMITER  ','
#define LARGER_N 10

// the number of bodies to simulate
int N = 0;
// the integer dimension of the activity grid
int D = 0;
// the number of simulation iterations
int I = 0;
// operation mode
MODE M;
// input file with an initial N bodies of data
char* input_file = NULL;
nbody* bodies = NULL;
// store the values of the D*D locations
float* density;
// force of every body
vector* f_arr;

void print_help();
void parse_parameter(int argc, char* argv[]);
void init_data_by_random(nbody* bodies);
void load_data_from_file(nbody* bodies);
char* get_string_in_range(char string[], int start, int end);
void step(void);
void serial_compute(void);
void parallel_compute(void);
void calc_density(void);
void update_location_velocity(void);
void print_bodies();

int main(int argc, char* argv[]) {
	/*print_help();
	printf("\n");*/

	// Processes the command line arguments
	// argc in the count of the command arguments
	// argv is an array (of length argc) of the arguments. The first argument is always the executable name (including path)
	parse_parameter(argc, argv);

	// Allocate any heap memory
	bodies = (nbody*)malloc(N * sizeof(nbody));
	// initialize all values are zero
	density = (float*)calloc(D * D, sizeof(float));
	f_arr = (vector*)malloc(N * sizeof(vector));

	// Depending on program arguments, either read initial data from file or generate random data.
	if (input_file == NULL) {
		printf("\n\nGenerate random data...");
		init_data_by_random(bodies);
	}
	else {
		printf("\n\nLoad data from csv...");
		load_data_from_file(bodies);
	}
	print_bodies();

	// Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	if (I == 0) {
		printf("\n\nStart simulate by visualisation mode with %s computing...", M == CPU ? "CPU" : "OPENMP");
		initViewer(N, D, M, step);
		setNBodyPositions(bodies);
		setActivityMapData(density);
		//setHistogramData(density);
		startVisualisationLoop();
	}
	else {
		printf("\n\nStart simulate by console mode with %s computing...", M == CPU ? "CPU" : "OPENMP");
		// start timer
		double begin_outer = omp_get_wtime();
		for (int i = 0; i < I; i++) {
			double begin = omp_get_wtime();
			step();
			double end = omp_get_wtime();
			double elapsed = end - begin;
			printf("\n\nIteration epoch:%d, Complete in %d seconds %f milliseconds", i, (int)elapsed, 1000 * (elapsed - (int)elapsed));

			print_bodies();

			/*for (int i = 0; i < D * D; i++) {
				printf("%f,", density[i]*N);
			}*/
			//printf("\n");
		}
		// stop timer
		double end_outer = omp_get_wtime();
		double total = end_outer - begin_outer;
		printf("\n\nFully Complete in %d seconds %f milliseconds\n", (int)total, 1000 * (total - (int)total));
	}
	return 0;
}

void step(void) {
	// Perform the main simulation of the NBody system
	if (M == CPU) {
		serial_compute();
	}
	else if (M == OPENMP) {
		parallel_compute();
	}
	else {
		fprintf(stderr, "\n%d mode is not supported", M);
	}
}

void serial_compute() {
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
		f_arr[j].x = f.x;
		f_arr[j].y = f.y;
	}
	calc_density();
	update_location_velocity();
}

void parallel_compute() {
	// compute the force of every body
	int j;
#pragma omp parallel for default(none) shared(N, bodies, f_arr) 
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
			f.x = (float)(f.x + s2.x / s3);
			f.y = (float)(f.y + s2.y / s3);
		}
		f.x = G * body_j->m * f.x;
		f.y = G * body_j->m * f.y;
		f_arr[j].x = f.x;
		f_arr[j].y = f.y;
	}
#pragma omp barrier
#pragma omp master

	calc_density();
	update_location_velocity();
}

// calculate the values of the D*D locations (density)
void calc_density() {
	// reset the value to zero
	for (int i = 0; i < D * D; i++) {
		density[i] = 0.0f;
	}
	for (int i = 0; i < N; i++) {
		nbody* body = &bodies[i];
		double scale = 1.0 / D;
		// x-axis coordinate of D*D locations
		int x = (int)ceil(body->x / scale) - 1;
		// y-axis coordinate of D*D locations
		int y = (int)ceil(body->y / scale) - 1;
		// the index of one dimensional array
		int index = x * D + (y + 1);
		// For large values of N
		int a = N;
		if (N > LARGER_N) {
			a = N * D;
		}
		density[index] = (density[index] * N + 1) / N;
		// density[index] = density[index] + 1;
		// printf("\nindex:%d: density:%f", index, density[index]);
	}
}

// // update location and velocity of nbodies
void update_location_velocity() {
	for (int i = 0; i < N; i++) {
		nbody* body = &bodies[i];
		// calc the acceleration of body
		vector a = { f_arr[i].x / body->m, f_arr[i].y / body->m };
		//printf("\na:(%f,%f)", a.x, a.y);
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

MODE str2enum(char* str) {
	//    MODE myEnum = (MODE)enum.Parse(typeof(MODE), str);
	if (strcmp(str, "CPU") == 0) return CPU;
	if (strcmp(str, "OPENMP") == 0) return OPENMP;
	if (strcmp(str, "CUDA") == 0) return CUDA;
}

void parse_parameter(int argc, char* argv[]) {
	//printf("\n params len:%s", argv[1]);
	/*for (int i = 1; i < argc; i++) {
		printf("Argument %d: %s\n", i, argv[i]);
	}*/
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
 * generate random data from 0 ~ 1
 */
float random_float() {
	//srand(NULL);
	// Keep only two decimal place
	float result = rand() % 100 / (float)100;
	return result;
}

void init_data_by_random(nbody* bodies) {
	for (int i = 0; i < N; i++) {
		bodies[i].x = random_float();
		bodies[i].y = random_float();
		bodies[i].vx = 0.0f;
		bodies[i].vy = 0.0f;
		bodies[i].m = (float)1.0 / N;
	}
}

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

void load_data_from_file(nbody* bodies) {
	char line[1000];
	FILE* file = NULL;
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
		char** res = split(line, ',', 5);
		bodies[i].x = res[0] == NULL ? random_float() : atof(res[0]);
		bodies[i].y = res[1] == NULL ? random_float() : atof(res[1]);
		bodies[i].vx = res[2] == NULL ? 0 : atof(res[2]);
		bodies[i].vy = res[3] == NULL ? 0 : atof(res[3]);
		bodies[i].m = res[4] == NULL ? 1.0 / N : atof(res[4]);
		i++;
		// free the res memory
		for (int i = 0; i < 5; i++) {
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

void print_bodies() {
	for (int i = 0; i < N; i++) {
		printf("\nx:%f, y:%f, vx:%f, vy:%f, m:%f", bodies[i].x, bodies[i].y, bodies[i].vx, bodies[i].vy, bodies[i].m);
	}
	printf("\n");
}

void get_openGL_info() {
	//glutInit(&argc, argv);
	glutCreateWindow("GLUT");
	glewInit();
	printf("OpenGL version supported by this platform (%s): \n", glGetString(GL_VERSION));
}
