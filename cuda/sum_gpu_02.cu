/* DATA_SIZE �̕��������_�̐ω��Z�� CPU ����� GPU �ł����Ȃ� - 2 */
/*  - rev.201905 by Yoshiki NAGATANI */

/*  - CPU ���� OpenMP �ɂ��v�Z�Ƃ���r����
		Visual Studio �ł� [�v���W�F�N�g]-[xx�̃v���p�e�B]-
		[CUDA C/C++]-[Host]-[Additional Compiler Options]
		�Ɂu -Xcompiler "/openmp" �v��ǉ��B
/*  - ���Ԃ��v������i�������b�P�ʁj */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

/* DATA_SIZE = BLOCK_SIZE * GRID_SIZE �Ŋ���؂�邱��(�v���O�������ł̓m�[�`�F�b�N) */
#define DATA_SIZE 1048576
#define BLOCK_SIZE 256
#define GRID_SIZE (DATA_SIZE/BLOCK_SIZE)

/* ���x��r�̂��ߓ����v�Z�� REPEAT ��J��Ԃ� */
#define REPEAT 10000

/*-----------------------------------------------------------*/
/* CPU���Őω��Z R=A*B �������Ȃ��֐�(�P��R�A) */
void MultiplyOnCPU_Single(float* h_data_A, float* h_data_B, float* h_data_R) {
	long i;

	/* CPU �ł̓f�[�^�̐����� for �����܂킷 */
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_R[i] = h_data_A[i] * h_data_B[i];
	}
}

/*-----------------------------------------------------------*/
/* CPU���Őω��Z R=A*B �������Ȃ��֐�(OpenMP) */
void MultiplyOnCPU_OpenMP(float* h_data_A, float* h_data_B, float* h_data_R) {
	long i;

	/* CPU �ł̓f�[�^�̐����� for �����܂킷 */
	#pragma omp parallel for
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_R[i] = h_data_A[i] * h_data_B[i];
	}
}

/*-----------------------------------------------------------*/
/* GPU���Őω��Z R=A*B �������Ȃ��֐� */
__global__ void MultiplyOnGPU(float* d_data_A, float* d_data_B, float* d_data_R) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	/* GPU �ł� for ���ł͂Ȃ��C�����̒S���̃f�[�^(id)�����v�Z����� OK */
	d_data_R[id] = d_data_A[id] * d_data_B[id];
}

/*-----------------------------------------------------------*/
int main(void) {

	int i;
	time_t time_start_cpu_single, time_end_cpu_single;
	time_t time_start_cpu_openmp, time_end_cpu_openmp;
	time_t time_start_gpu, time_end_gpu;

	printf("DATA_SIZE(%d) = BLOCK_SIZE(%d) x GRID_SIZE(%d).\n", DATA_SIZE, BLOCK_SIZE, GRID_SIZE);

	float* h_data_A;   /* Host(CPU)�������� */
	float* h_data_B;   /* Host(CPU)�������� */
	float* h_data_R;   /* Host(CPU)�������� */
	float* h_data_R_fromGPU;   /* Host(CPU)���������i���ʂ̃`�F�b�N��p�j */

	float* d_data_A;   /* Devive(GPU)�������� */
	float* d_data_B;   /* Devive(GPU)�������� */
	float* d_data_R;   /* Devive(GPU)�������� */

	/* �z�X�g(CPU)���������̈�̊m�ہi�ǐ��d���̂��߃G���[�`�F�b�N�����Ȃ̂Œ��Ӂj */
	h_data_A = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_B = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_R = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_R_fromGPU = (float*)malloc(DATA_SIZE * sizeof(float));

	/* �f�o�C�X(GPU)���������̈�̊m�ہi�ǐ��d���̂��߃G���[�`�F�b�N�����Ȃ̂Œ��Ӂj */
	cudaMalloc((void**)& d_data_A, DATA_SIZE * sizeof(float));
	cudaMalloc((void**)& d_data_B, DATA_SIZE * sizeof(float));
	cudaMalloc((void**)& d_data_R, DATA_SIZE * sizeof(float));

	/* �f�[�^����(���̗�ł�CPU���Ő������Ă���) */
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_A[i] = (double)(rand()) / 32768.0;
		h_data_B[i] = (double)(rand()) / 32768.0;
		h_data_R[i] = 0.0;
	}

	/* �f�o�C�X�Ƀ��������e���R�s�[(CPU��GPU) */
	cudaMemcpy(d_data_A, h_data_A, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data_B, h_data_B, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	/* �z�X�g(Single CPU)�Őω��Z�����s�i���x�v���̂��� REPEAT ��J��Ԃ��j */
	printf("Start calculation on Single CPU for %d times...", REPEAT);
	time_start_cpu_single = time(NULL);
	for (i = 0; i < REPEAT; i++) {
		MultiplyOnCPU_Single(h_data_A, h_data_B, h_data_R);
	}
	time_end_cpu_single = time(NULL);
	printf("done!! (Time: %d s)\n", time_end_cpu_single - time_start_cpu_single);

	/* �z�X�g(CPU with OpenMP)�Őω��Z�����s�i���x�v���̂��� REPEAT ��J��Ԃ��j */
	printf("Start calculation on CPU with OpenMP for %d times...", REPEAT);
	time_start_cpu_openmp = time(NULL);
	for (i = 0; i < REPEAT; i++) {
		MultiplyOnCPU_OpenMP(h_data_A, h_data_B, h_data_R);
	}
	time_end_cpu_openmp = time(NULL);
	printf("done!! (Time: %d s)\n", time_end_cpu_openmp - time_start_cpu_openmp);

	/* �f�o�C�X(GPU)�Őω��Z�����s�i���x�v���̂��� REPEAT ��J��Ԃ��j */
	printf("Start calculation on GPU for %d times...", REPEAT);
	time_start_gpu = time(NULL);
	for (i = 0; i < REPEAT; i++) {
		MultiplyOnGPU << <GRID_SIZE, BLOCK_SIZE >> > (d_data_A, d_data_B, d_data_R);
	}
	time_end_gpu = time(NULL);
	printf("done!! (Time: %d s)\n", time_end_gpu - time_start_gpu);

	/* �f�o�C�X���烁�������e���R�s�[(CPU��GPU) */
	cudaMemcpy(h_data_R_fromGPU, d_data_R, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	/* ���ʂ̔�r(CPU���)(��ʕ\���̓s����C�ŏ��ƍŌ�̃f�[�^�����\��) */
	printf("Comparison of the Results:\n");
	printf(" %8d: CPU:%f vs GPU:%f\n", 0, h_data_R[0], h_data_R_fromGPU[0]);
	printf(" %8d: CPU:%f vs GPU:%f\n", DATA_SIZE - 1, h_data_R[DATA_SIZE - 1], h_data_R_fromGPU[DATA_SIZE - 1]);

	cudaDeviceReset();
	return 0;
}
