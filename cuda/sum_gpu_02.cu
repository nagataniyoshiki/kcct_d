/* DATA_SIZE 個の浮動小数点の積演算を CPU および GPU でおこなう - 2 */
/*  - rev.201905 by Yoshiki NAGATANI */

/*  - CPU 側の OpenMP による計算とも比較する
		Visual Studio では [プロジェクト]-[xxのプロパティ]-
		[CUDA C/C++]-[Host]-[Additional Compiler Options]
		に「 -Xcompiler "/openmp" 」を追加。
/*  - 時間も計測する（ただし秒単位） */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

/* DATA_SIZE = BLOCK_SIZE * GRID_SIZE で割り切れること(プログラム側ではノーチェック) */
#define DATA_SIZE 1048576
#define BLOCK_SIZE 256
#define GRID_SIZE (DATA_SIZE/BLOCK_SIZE)

/* 速度比較のため同じ計算を REPEAT 回繰り返す */
#define REPEAT 10000

/*-----------------------------------------------------------*/
/* CPU側で積演算 R=A*B をおこなう関数(単一コア) */
void MultiplyOnCPU_Single(float* h_data_A, float* h_data_B, float* h_data_R) {
	long i;

	/* CPU ではデータの数だけ for 文をまわす */
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_R[i] = h_data_A[i] * h_data_B[i];
	}
}

/*-----------------------------------------------------------*/
/* CPU側で積演算 R=A*B をおこなう関数(OpenMP) */
void MultiplyOnCPU_OpenMP(float* h_data_A, float* h_data_B, float* h_data_R) {
	long i;

	/* CPU ではデータの数だけ for 文をまわす */
	#pragma omp parallel for
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_R[i] = h_data_A[i] * h_data_B[i];
	}
}

/*-----------------------------------------------------------*/
/* GPU側で積演算 R=A*B をおこなう関数 */
__global__ void MultiplyOnGPU(float* d_data_A, float* d_data_B, float* d_data_R) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	/* GPU では for 文ではなく，自分の担当のデータ(id)だけ計算すれば OK */
	d_data_R[id] = d_data_A[id] * d_data_B[id];
}

/*-----------------------------------------------------------*/
int main(void) {

	int i;
	time_t time_start_cpu_single, time_end_cpu_single;
	time_t time_start_cpu_openmp, time_end_cpu_openmp;
	time_t time_start_gpu, time_end_gpu;

	printf("DATA_SIZE(%d) = BLOCK_SIZE(%d) x GRID_SIZE(%d).\n", DATA_SIZE, BLOCK_SIZE, GRID_SIZE);

	float* h_data_A;   /* Host(CPU)側メモリ */
	float* h_data_B;   /* Host(CPU)側メモリ */
	float* h_data_R;   /* Host(CPU)側メモリ */
	float* h_data_R_fromGPU;   /* Host(CPU)側メモリ（結果のチェック専用） */

	float* d_data_A;   /* Devive(GPU)側メモリ */
	float* d_data_B;   /* Devive(GPU)側メモリ */
	float* d_data_R;   /* Devive(GPU)側メモリ */

	/* ホスト(CPU)側メモリ領域の確保（可読性重視のためエラーチェック無しなので注意） */
	h_data_A = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_B = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_R = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_R_fromGPU = (float*)malloc(DATA_SIZE * sizeof(float));

	/* デバイス(GPU)側メモリ領域の確保（可読性重視のためエラーチェック無しなので注意） */
	cudaMalloc((void**)& d_data_A, DATA_SIZE * sizeof(float));
	cudaMalloc((void**)& d_data_B, DATA_SIZE * sizeof(float));
	cudaMalloc((void**)& d_data_R, DATA_SIZE * sizeof(float));

	/* データ生成(この例ではCPU側で生成している) */
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_A[i] = (double)(rand()) / 32768.0;
		h_data_B[i] = (double)(rand()) / 32768.0;
		h_data_R[i] = 0.0;
	}

	/* デバイスにメモリ内容をコピー(CPU→GPU) */
	cudaMemcpy(d_data_A, h_data_A, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data_B, h_data_B, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	/* ホスト(Single CPU)で積演算を実行（速度計測のため REPEAT 回繰り返し） */
	printf("Start calculation on Single CPU for %d times...", REPEAT);
	time_start_cpu_single = time(NULL);
	for (i = 0; i < REPEAT; i++) {
		MultiplyOnCPU_Single(h_data_A, h_data_B, h_data_R);
	}
	time_end_cpu_single = time(NULL);
	printf("done!! (Time: %d s)\n", time_end_cpu_single - time_start_cpu_single);

	/* ホスト(CPU with OpenMP)で積演算を実行（速度計測のため REPEAT 回繰り返し） */
	printf("Start calculation on CPU with OpenMP for %d times...", REPEAT);
	time_start_cpu_openmp = time(NULL);
	for (i = 0; i < REPEAT; i++) {
		MultiplyOnCPU_OpenMP(h_data_A, h_data_B, h_data_R);
	}
	time_end_cpu_openmp = time(NULL);
	printf("done!! (Time: %d s)\n", time_end_cpu_openmp - time_start_cpu_openmp);

	/* デバイス(GPU)で積演算を実行（速度計測のため REPEAT 回繰り返し） */
	printf("Start calculation on GPU for %d times...", REPEAT);
	time_start_gpu = time(NULL);
	for (i = 0; i < REPEAT; i++) {
		MultiplyOnGPU << <GRID_SIZE, BLOCK_SIZE >> > (d_data_A, d_data_B, d_data_R);
	}
	time_end_gpu = time(NULL);
	printf("done!! (Time: %d s)\n", time_end_gpu - time_start_gpu);

	/* デバイスからメモリ内容をコピー(CPU←GPU) */
	cudaMemcpy(h_data_R_fromGPU, d_data_R, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	/* 結果の比較(CPU上で)(画面表示の都合上，最初と最後のデータだけ表示) */
	printf("Comparison of the Results:\n");
	printf(" %8d: CPU:%f vs GPU:%f\n", 0, h_data_R[0], h_data_R_fromGPU[0]);
	printf(" %8d: CPU:%f vs GPU:%f\n", DATA_SIZE - 1, h_data_R[DATA_SIZE - 1], h_data_R_fromGPU[DATA_SIZE - 1]);

	cudaDeviceReset();
	return 0;
}
