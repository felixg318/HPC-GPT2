void mat_inplace_add (float *A, float *B, int y, int x) {
	for (int i = 0; i < y; ++i)
		for (int j = 0; j < x; ++j)
			A[i * x + j] += B[i * x + j];
}
