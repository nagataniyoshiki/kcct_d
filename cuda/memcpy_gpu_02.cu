/* CPU��GPU�ւ̃f�[�^�R�s�[ �� CPU��GPU�ւ̃f�[�^�R�s�[ */
/*  - rev.201905 by Yoshiki NAGATANI */

#include <stdio.h>

/* DATA_SIZE = BLOCK_SIZE * GRID_SIZE �Ŋ���؂�邱��(�v���O�������ł̓m�[�`�F�b�N) */
#define DATA_SIZE 8
#define BLOCK_SIZE 4
#define GRID_SIZE (DATA_SIZE/BLOCK_SIZE)

/*-----------------------------------------------------------*/
/* GPU���Ńf�[�^���e��2�{���ĕ\������֐� */
__global__ void DoubleOnGPU(float* d_data, float* d_data2) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	/* GPU �ł� for ���ł͂Ȃ��C�����̒S���̃f�[�^(id)�����v�Z����� OK */
	d_data2[id] = d_data[id] * 2.0;
	printf("My target is d_data[%d] : %f * 2.0 = %f.\n", id, d_data[id], d_data2[id]);
}

/*-----------------------------------------------------------*/
int main(void) {

	float* h_data;    /* Host(CPU)�������� */
	float* h_data2;   /* Host(CPU)�������� */

	float* d_data;    /* Device(GPU)�������� */
	float* d_data2;   /* Device(GPU)�������� */

	/* �z�X�g(CPU)���������̈�̊m�ہi�ǐ��d���̂��߃G���[�`�F�b�N�����Ȃ̂Œ��Ӂj */
	h_data = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data2 = (float*)malloc(DATA_SIZE * sizeof(float));

	/* �f�o�C�X(GPU)���������̈�̊m�ہi�ǐ��d���̂��߃G���[�`�F�b�N�����Ȃ̂Œ��Ӂj */
	cudaMalloc((void**)& d_data, DATA_SIZE * sizeof(float));
	cudaMalloc((void**)& d_data2, DATA_SIZE * sizeof(float));

	/* �����l�̑��(CPU���Ő���) */
	printf("Data before processing: ");
	for (int i = 0; i < DATA_SIZE; i++) {
		h_data[i] = (float)(i) * 10.0;
		printf("%f, ", h_data[i]);
	}
	printf("\n");

	/* �f�o�C�X�Ƀ��������e���R�s�[(CPU��GPU) */
	cudaMemcpy(d_data, h_data, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	/* �f�o�C�X(GPU)��2�{���������s */
	DoubleOnGPU <<<GRID_SIZE, BLOCK_SIZE>>> (d_data, d_data2);

	/* �f�o�C�X���烁�������e���R�s�[(CPU��GPU) */
	cudaMemcpy(h_data2, d_data2, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	/* ���ʂ̕\��(CPU����) */
	printf("Data after processing: ");
	for (int i = 0; i < DATA_SIZE; i++) {
		printf("%f, ", h_data2[i]);
	}

	return 0;
}
