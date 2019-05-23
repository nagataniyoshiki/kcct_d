#include <stdio.h>

/* DATA_SIZE = BLOCK_SIZE * GRID_SIZE で割り切れること(プログラム側ではノーチェック) */
#define DATA_SIZE 16
#define BLOCK_SIZE 8
#define GRID_SIZE (DATA_SIZE/BLOCK_SIZE)

__global__ void helloFromGPU() {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	printf("I am blockDim.x=%3d, blockIdx.x=%3d, threadIdx.x=%3d. My target is %3d.\n", blockDim.x, blockIdx.x, threadIdx.x, id);
}

int main(void) {
	printf("Hello World from CPU!  DATA_SIZE(%d) = BLOCK_SIZE(%d) x GRID_SIZE(%d).\n", DATA_SIZE, BLOCK_SIZE, GRID_SIZE);

	helloFromGPU <<<GRID_SIZE, BLOCK_SIZE>>> ();
	cudaDeviceSynchronize();

	printf("Goodbye World from CPU!\n");

	cudaDeviceReset();
	return 0;
}
