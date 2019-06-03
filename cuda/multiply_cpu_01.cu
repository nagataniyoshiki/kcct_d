/* DATA_SIZE �̕��������_�̐ω��Z�� CPU �ł����Ȃ� */
/*  - rev.201905 by Yoshiki NAGATANI */

#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 1048576

/* ���x��r�̂��ߓ����v�Z�� REPEAT ��J��Ԃ� */
#define REPEAT 10000

/*-----------------------------------------------------------*/
/* �ω��Z R=A*B �������Ȃ��֐�(�P��R�A) */
void MultiplyOnCPU(float* h_data_A, float* h_data_B, float* h_data_R) {
	long i;

	/* CPU �ł̓f�[�^�̐����� for �����܂킷 */
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_R[i] = h_data_A[i] * h_data_B[i];
	}
}

/*-----------------------------------------------------------*/
int main(void) {

	int i;

	printf("DATA_SIZE(%d)\n", DATA_SIZE);

	float* h_data_A;   /* Host(CPU)�������� */
	float* h_data_B;   /* Host(CPU)�������� */
	float* h_data_R;   /* Host(CPU)�������� */

	/* �������̈�̊m�ہi�ǐ��d���̂��߃G���[�`�F�b�N�����Ȃ̂Œ��Ӂj */
	h_data_A = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_B = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_R = (float*)malloc(DATA_SIZE * sizeof(float));

	/* �f�[�^���� */
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_A[i] = (double)(rand()) / 32768.0;
		h_data_B[i] = (double)(rand()) / 32768.0;
		h_data_R[i] = 0.0;
	}

	/* �ω��Z�����s�i���x�v���̂��� REPEAT ��J��Ԃ��j */
	printf("Start calculation on CPU for %d times...", REPEAT);
	for (i = 0; i < REPEAT; i++) {
		MultiplyOnCPU(h_data_A, h_data_B, h_data_R);
	}
	printf("done!!\n");

	/* ���ʂ̕\��(��ʕ\���̓s����C�ŏ��ƍŌ�̃f�[�^�����\��) */
	printf("Results:\n");
	printf(" %8d: %f\n", 0, h_data_R[0]);
	printf(" %8d: %f\n", DATA_SIZE - 1, h_data_R[DATA_SIZE - 1]);

	return 0;
}
