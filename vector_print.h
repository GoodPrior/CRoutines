
void vprint(double* x, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		printf("%g ", x[i]);
	}
	printf("\n");
}

void vprinti(int* x, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		printf("%d ", x[i]);
	}
	//  printf("\n");
}

void mprinti(int* x, int m, int n)
{
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%d ", x[i*n + j]);
		}
		printf("\n");
	}
}

void mprint(double* x, int m, int n)
{
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%g ", x[i*n + j]);
		}
		printf("\n");
	}
}

void vfprinti(FILE* stream, int* x, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		fprintf(stream, "%d\n", x[i]);
	}
}

void vfprint(FILE* stream, double* x, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		fprintf(stream, "%.5g\n", x[i]);
	}
}

void mfprint(FILE* stream, double* x, int m, int n)
{
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			fprintf(stream, "%.5g ", x[i*n + j]);
		}
		fprintf(stream, "\n");
	}
}

void mfprinti(FILE* stream, int* x, int m, int n)
{
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			fprintf(stream, "%d ", x[i*n + j]);
		}
		fprintf(stream, "\n");
	}
}

void mfprint(FILE* stream, double* x, int m, int stepm, int n, int stepn)
{
	int i, j;
	for (i = 0; i < m; i += stepm) {
		for (j = 0; j < n; j += stepn) {
			fprintf(stream, "%.5g ", x[i*n + j]);
		}
		fprintf(stream, "\n");
	}
}

void mtfprint(FILE* stream, double* x, int m, int n)
{
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < m; j++) {
			fprintf(stream, "%.5g ", x[j*m + i]);
		}
		fprintf(stream, "\n");
	}
}

void mfprintfile(char* file, double* x, int m, int n)
{
	FILE* f = fopen(file, "w");
	mfprint(f, x, m, n);
	fclose(f);
}

void mfprintfile(char* file, double* x, int m, int stepm, int n, int stepn)
{
	FILE* f = fopen(file, "w");
	mfprint(f, x, m, stepm, n, stepn);
	fclose(f);
}

void mtfprintfile(char* file, double* x, int m, int n)
{
	FILE* f = fopen(file, "w");
	mtfprint(f, x, m, n);
	fclose(f);
}

