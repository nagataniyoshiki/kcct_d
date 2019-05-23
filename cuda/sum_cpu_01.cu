/* DATA_SIZE 個の浮動小数点の積演算を CPU でおこなう */
/*  - rev.201905 by Yoshiki NAGATANI */

#include <stdio.h>
#include <stdlib.h>

#define DATA_SIZE 1048576

/* 速度比較のため同じ計算を REPEAT 回繰り返す */
#define REPEAT 10000

/*-----------------------------------------------------------*/
/* 積演算 R=A*B をおこなう関数(単一コア) */
void MultiplyOnCPU(float* h_data_A, float* h_data_B, float* h_data_R) {
	long i;

	/* CPU ではデータの数だけ for 文をまわす */
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_R[i] = h_data_A[i] * h_data_B[i];
	}
}

/*-----------------------------------------------------------*/
int main(void) {

	int i;

	printf("DATA_SIZE(%d)\n", DATA_SIZE);

	float* h_data_A;   /* Host(CPU)側メモリ */
	float* h_data_B;   /* Host(CPU)側メモリ */
	float* h_data_R;   /* Host(CPU)側メモリ */

	/* メモリ領域の確保（可読性重視のためエラーチェック無しなので注意） */
	h_data_A = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_B = (float*)malloc(DATA_SIZE * sizeof(float));
	h_data_R = (float*)malloc(DATA_SIZE * sizeof(float));

	/* データ生成 */
	for (i = 0; i < DATA_SIZE; i++) {
		h_data_A[i] = (double)(rand()) / 32768.0;
		h_data_B[i] = (double)(rand()) / 32768.0;
		h_data_R[i] = 0.0;
	}

	/* 積演算を実行（速度計測のため REPEAT 回繰り返し） */
	printf("Start calculation on CPU for %d times...", REPEAT);
	for (i = 0; i < REPEAT; i++) {
		MultiplyOnCPU(h_data_A, h_data_B, h_data_R);
	}
	printf("done!!\n");

	/* 結果の表示(画面表示の都合上，最初と最後のデータだけ表示) */
	printf("Results:\n");
	printf(" %8d: %f\n", 0, h_data_R[0]);
	printf(" %8d: %f\n", DATA_SIZE - 1, h_data_R[DATA_SIZE - 1]);

	return 0;
}
