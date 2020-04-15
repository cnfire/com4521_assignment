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
void step(void);
void serial_compute(void);
void parallel_compute(void);
void calc_density(void);
void update_location_velocity(void);

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
		printf("\nGenerate random data...");
		init_data_by_random(bodies);
	}
	else {
		printf("\nLoad data from csv...");
		load_data_from_file(bodies);
	}
	// print_bodies();

	// Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	if (I == 0) {
		printf("\nStart simulate by visualisation mode with %s computing...", M == CPU ? "CPU" : "OPENMP");
		initViewer(N, D, M, step);
		setNBodyPositions(bodies);
		//setActivityMapData(density);
		setHistogramData(density);
		startVisualisationLoop();
	}
	else {
		printf("\nStart simulate by console mode with %s computing...", M == CPU ? "CPU" : "OPENMP");
		// start timer
		double begin_outer = omp_get_wtime();
		for (int i = 0; i < I; i++) {
			double begin = omp_get_wtime();
			step();
			double end = omp_get_wtime();
			double elapsed = end - begin;
			printf("\nIteration epoch:%d, Complete in %f seconds\n", i, elapsed);
		}
		// stop timer
		double end_outer = omp_get_wtime();
		double total_elapsed = end_outer - begin_outer;
		printf("\nFully Complete in %f seconds\n", total_elapsed);
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
			f.x = (float)(f.x + s2.x / s3);
			f.y = (float)(f.y + s2.y / s3);
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
	for (int j = 0; j < D * D; j++) {
		density[j] = 0.0f;
	}
	for (int j = 0; j < N; j++) {
		nbody* body = &bodies[j];
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
		density[index] = (density[index] * a + 1) / a;
		//            density[index] = density[index] + 1;
		//            printf("\nindex:%d: density:%f", index, density[index]);
	}
}

// // update location and velocity of nbodies
void update_location_velocity() {
	for (int j = 0; j < N; j++) {
		nbody* body = &bodies[j];
		vector a = { f_arr[j].x / body->m, f_arr[j].y / body->m };
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
	printf("\nN:%d, D:%d, M:%d, I:%d, f:%s", N, D, M, I, input_file);
}

/**
 * generate random data from 0 ~ 1
 */
float random_float() {
	// 设置随机数种子，使每次产生的随机序列不同
	//    srand(NULL);
		// Keep only one decimal place
	float result = rand() % 10 / (float)10;
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

/**判断str1是否以str2开头
 * 如果是返回1
 * 不是返回0
 * 出错返回-1
 * */
int is_begin_with(const char* str1, char* str2) {
	if (str1 == NULL || str2 == NULL)
		return -1;
	int len1 = strlen(str1);
	int len2 = strlen(str2);
	if ((len1 < len2) || (len1 == 0 || len2 == 0))
		return -1;
	char* p = str2;
	int i = 0;
	while (*p != '\0') {
		if (*p != str1[i])
			return 0;
		p++;
		i++;
	}
	return 1;
}

// todo 字符串分割时剩下的2个问题：1：",,,"
float* split(char* str) {
	float* result = (float*)malloc(P * sizeof(float));
	char* ptr = strtok(str, ",");
	int i = 0;
	while (ptr != NULL) {
		//        todo 默认值设置
		result[i] = (float)atof(ptr);
		//        printf("split:%f", atof(ptr));
		ptr = strtok(NULL, ",");
		i++;
	}
	return result;
}

void load_data_from_file(nbody* bodies) {
	char buff[255];
	FILE* fp = NULL;
	fp = fopen(input_file, "rb");
	if (fp == NULL) {
		fprintf(stderr, "Error: Could not find file:%s \n", input_file);
		exit(1);
	}
	int i = 0;
	while (!feof(fp)) {
		fgets(buff, 255, fp);
		//        skip the comment line
		if (is_begin_with(buff, "#") == 1) {
			continue;
		}
		//        todo empty line skip
		//        printf("\nbuff:%s", buff);
		float* res = split(buff);
		bodies[i].x = res[0];
		bodies[i].y = res[1];
		bodies[i].vx = res[2];
		bodies[i].vy = res[3];
		bodies[i].m = res[4];
		i++;
	}
	fclose(fp);
	if (i != N) {
		fprintf(stderr, "\nN is %d and the number of body in csv file is %d, not equal!", N, i);
		exit(0);
	}
}

void print_bodies() {
	for (int i = 0; i < N; i++) {
		printf("\nx:%f, y:%f, vx:%f, vy:%f, m:%f", bodies[i].x, bodies[i].y, bodies[i].vx, bodies[i].vy, bodies[i].m);
	}
}

void get_openGL_info() {
	//glutInit(&argc, argv);
	glutCreateWindow("GLUT");
	glewInit();
	printf("OpenGL version supported by this platform (%s): \n", glGetString(GL_VERSION));
}
