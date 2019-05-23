/* CPU→GPUへのデータコピー */
/*  - rev.201905 by Yoshiki NAGATANI */

#include <stdio.h>

/* DATA_SIZE = BLOCK_SIZE * GRID_SIZE で割り切れること(プログラム側ではノーチェック) */
#define DATA_SIZE 8
#define BLOCK_SIZE 4
#define GRID_SIZE (DATA_SIZE/BLOCK_SIZE)

/*-----------------------------------------------------------*/
/* GPU側でデータ内容を2倍して表示する関数 */
__global__ void DoubleOnGPU(float* d_data) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	/* GPU では for 文ではなく，自分の担当のデータ(id)だけ計算すれば OK */
	printf("My target is d_data[%d] : %f * 2.0 = %f.\n", id, d_data[id], d_data[id] * 2.0);
}

/*-----------------------------------------------------------*/
int main(void) {

	float* h_data;    /* Host(CPU)側メモリ */
	float* d_data;    /* Devive(GPU)側メモリ */

	/* ホスト(CPU)側メモリ領域の確保（可読性重視のためエラーチェック無しなので注意） */
	h_data = (float*)malloc(DATA_SIZE * sizeof(float));

	/* デバイス(GPU)側メモリ領域の確保（可読性重視のためエラーチェック無しなので注意） */
	cudaMalloc((void**)&d_data, DATA_SIZE * sizeof(float));

	/* 初期値の代入(CPU側で生成) */
	printf("Data before processing: ");
	for (int i = 0; i < DATA_SIZE; i++) {
		h_data[i] = (float)(i) * 10.0;
		printf("%f, ", h_data[i]);
	}
	printf("\n");

	/* デバイスにメモリ内容をコピー(CPU→GPU) */
	cudaMemcpy(d_data, h_data, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	/* デバイス(GPU)で2倍処理を実行 */
	DoubleOnGPU <<<GRID_SIZE, BLOCK_SIZE>>> (d_data);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
