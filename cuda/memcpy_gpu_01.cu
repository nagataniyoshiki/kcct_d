/* CPU��GPU�ւ̃f�[�^�R�s�[ */
/*  - rev.201905 by Yoshiki NAGATANI */

#include <stdio.h>

/* DATA_SIZE = BLOCK_SIZE * GRID_SIZE �Ŋ���؂�邱��(�v���O�������ł̓m�[�`�F�b�N) */
#define DATA_SIZE 8
#define BLOCK_SIZE 4
#define GRID_SIZE (DATA_SIZE/BLOCK_SIZE)

/*-----------------------------------------------------------*/
/* GPU���Ńf�[�^���e��2�{���ĕ\������֐� */
__global__ void DoubleOnGPU(float* d_data) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	/* GPU �ł� for ���ł͂Ȃ��C�����̒S���̃f�[�^(id)�����v�Z����� OK */
	printf("My target is d_data[%d] : %f * 2.0 = %f.\n", id, d_data[id], d_data[id] * 2.0);
}

/*-----------------------------------------------------------*/
int main(void) {

	float* h_data;    /* Host(CPU)�������� */
	float* d_data;    /* Devive(GPU)�������� */

	/* �z�X�g(CPU)���������̈�̊m�ہi�ǐ��d���̂��߃G���[�`�F�b�N�����Ȃ̂Œ��Ӂj */
	h_data = (float*)malloc(DATA_SIZE * sizeof(float));

	/* �f�o�C�X(GPU)���������̈�̊m�ہi�ǐ��d���̂��߃G���[�`�F�b�N�����Ȃ̂Œ��Ӂj */
	cudaMalloc((void**)&d_data, DATA_SIZE * sizeof(float));

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
	DoubleOnGPU <<<GRID_SIZE, BLOCK_SIZE>>> (d_data);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
