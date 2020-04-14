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

void print_help();

void step(void);

void step_with_parallel(void);

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

MODE str2enum(char* str) {
	//    MODE myEnum = (MODE)enum.Parse(typeof(MODE), str);
	if (strcmp(str, "CPU") == 0) return CPU;
	if (strcmp(str, "OPENMP") == 0) return OPENMP;
	if (strcmp(str, "CUDA") == 0) return CUDA;
}

void parse_parameter(int argc, char* argv[]) {
	//printf("\n params len:%s", argv[1]);
	for (int i = 1; i < argc; i++) {
		printf("Argument %d: %s\n", i, argv[i]);
	}
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

int main(int argc, char* argv[]) {
	print_help();
	printf("\n");

	//TODO: Processes the command line arguments
	//argc in the count of the command arguments
	//argv is an array (of length argc) of the arguments. The first argument is always the executable name (including path)
	parse_parameter(argc, argv);

	//TODO: Allocate any heap memory
//    nbody arr[N];
	bodies = (nbody*)malloc(N * sizeof(nbody));
	// initialize all values are zero
	density = (float*)calloc(D * D, sizeof(float));

	//TODO: Depending on program arguments, either read initial data from file or generate random data.
	if (input_file == NULL) {
		printf("\ngenerate random data...");
		init_data_by_random(bodies);
	}
	else {
		printf("\nload data from csv...");
		//        load_data_from_file(bodies);
		init_data_by_random(bodies);
	}
	//    print_bodies();


		//TODO: Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	if (M == CPU) {
		// step();
		initViewer(N, D, M, step);
		setNBodyPositions(bodies);
		//setActivityMapData(density);
		setHistogramData(density);
		startVisualisationLoop();
	}
	else if (M == OPENMP) {
		printf("\ncompute by openmp");
		step_with_parallel();
	}
	else {
		fprintf(stderr, "\n%d mode is not supported", M);
	}
	return 0;
}

void step(void) {
	//TODO: Perform the main simulation of the NBody system
	printf("\ncompute by serial");
	vector* f_arr = (vector*)malloc(N * sizeof(vector));

	for (int i = 0; i < I; i++) {
		//start timer
		double begin = omp_get_wtime();
		printf("\niteration epoch:%d", i);
		//        vector f_arr[N];
		for (int l = 0; l < N; l++) {
			nbody* body_l = &bodies[l];
			vector f = { 0, 0 };
			for (int j = 0; j < N; j++) {
				// skip the influence of body on itself
				if (j == l) continue;
				nbody* body_j = &bodies[j];
				vector s1 = { body_j->x - body_l->x, body_j->y - body_l->y };
				vector s2 = { s1.x * body_j->m, s1.y * body_j->m };
				double s3 = pow(pow(s1.x, 2) + pow(s1.y, 2) + pow(SOFTENING, 2), 1.5);
				f.x = (float)(f.x + s2.x / s3);
				f.y = (float)(f.y + s2.y / s3);
			}
			f.x = G * body_l->m * f.x;
			f.y = G * body_l->m * f.y;
			//                printf("\nvector %d = (%f,%f)", j, f.x, f.y);
			f_arr[l].x = f.x;
			f_arr[l].y = f.y;
		}
		//        print_bodies();

				//calculate the values of the D*D locations (density)
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
		//        for (int j = 0; j < D * D; j++) {
		//            printf("\nindex:%d: density:%f", j, density[j]);
		//        }

				// update location,velocity,
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
		//stop timer
		double end = omp_get_wtime();

		double elapsed = end - begin;
		printf("\nComplete in %f seconds\n", elapsed);
	}
}

void step_with_parallel(void) {
	printf("\ncompute by parallel");
	vector* f_arr = (vector*)malloc(N * sizeof(vector));

	for (int i = 0; i < I; i++) {
		//start timer
		double begin = omp_get_wtime();
		printf("\niteration epoch:%d", i);
		int l;
#pragma omp parallel for default(none) shared(N, bodies, f_arr) 
		for (l = 0; l < N; l++) {
			nbody* body_l = &bodies[l];
			vector f = { 0, 0 };
			//#pragma omp parallel for default(none) private(l) shared(body_l,N,bodies)
			for (int j = 0; j < N; j++) {
				// skip the influence of body on itself
				if (j == l) continue;
				nbody* body_j = &bodies[j];
				vector s1 = { body_j->x - body_l->x, body_j->y - body_l->y };
				vector s2 = { s1.x * body_j->m, s1.y * body_j->m };
				double s3 = pow(pow(s1.x, 2) + pow(s1.y, 2) + pow(SOFTENING, 2), 1.5);
				f.x = f.x + s2.x / s3;
				f.y = f.y + s2.y / s3;
			}
			f.x = G * body_l->m * f.x;
			f.y = G * body_l->m * f.y;
			f_arr[l].x = f.x;
			f_arr[l].y = f.y;
		}
		//        print_bodies();

#pragma omp barrier
#pragma omp master

		// reset the value to zero
		for (int j = 0; j < D * D; j++) {
			density[j] = 0.0f;
		}

		//#pragma omp parallel for num_threads(2) default(none) shared(D, N, bodies, f_arr, density)
		for (int j = 0; j < N; j++) {
			nbody* body = &bodies[j];

			// calculate the values of the D*D locations (density)
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

			// update location,velocity
			vector v = { f_arr[j].x / body->m, f_arr[j].y / body->m };
			// new velocity
			vector v_new = { body->vx + dt * v.x, body->vy + dt * v.y };
			// new location
			vector l_new = { body->x + dt * v_new.x, body->y + dt * v_new.y };
			body->x = l_new.x;
			body->y = l_new.y;
			body->vx = v_new.x;
			body->vy = v_new.y;
		}
		//stop timer
		double end = omp_get_wtime();
		double elapsed = end - begin;
		printf(" Complete in %f seconds\n", elapsed);
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
