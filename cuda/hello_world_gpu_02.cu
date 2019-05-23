#include <stdio.h>

__global__ void helloFromGPU() {
	printf("Hello World from GPU (Thread #%d)!\n", threadIdx.x);
}

int main(void){
	printf("Hello World from CPU!\n");

	helloFromGPU<<<1,10>>>();
	cudaDeviceSynchronize();

	printf("Goodbye World from CPU!\n");

	cudaDeviceReset();
	return 0;
}
