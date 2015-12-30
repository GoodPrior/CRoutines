/*
 * matrix_math.h
 *
 *  Created on: Sep 30, 2013
 *      Author: wenlan
 */

#pragma once

#include <cstring>
#include <cmath>
#include "mkl.h"
#include <assert.h>

#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define MIN(a,b) (((a)<(b)) ? (a) : (b))

void vplus(double* x, int n, double* y);
void vminus(double* x, int n, double* y);
void vmulti(double* x, int n, double* y);
void mvrowmulti(double* x, int mx, int nx, double* y);
void vaxpby(int n, double a, double* x, double b, double* y);
void vcopy(double* x, int n, double* y);
void vcopyi(int* x, int n, int* y);
void vrrepmat(double* x, int t, double* r, int m, int n);
void vrep(double* x, int n, int t, double* r);
void mtrans(double* a, int m, int n);
void vcrepmat(double* x, int t, double* r, int m, int n);
void vstimes(double k, double* x, int n);
double vdot(double* x, int n, double* y);
double vnorm(double* x, int n);
void vlinspace(double min, double max, int n, double* r);
void vloglinspace(double min, double max, int n, double* r);
void mkronprod(double* x, int rx, int cx, double* y, int ry, int cy, double* r);
void vzeros(double* x, int n);
void vzerosi(int* x, int n);
void vones(double* x, int n);
void vonesi(int* x, int n);
void meye(double* x, int n);
void mcolsum(double *a, int m, int n, double* y);
void mrowsum(double *a, int m, int n, double* y);
void mrowcsum(double* a, int m, int n);
void mcolcsum(double* a, int m, int n);
void mcolNormalized(double *a, int m, int n);
void msolve(double* a, int m, int n, double* b, int nb, char trans);
void minv(double* a, int n);
void mmtimes(double* a, int ma, int na, double* b, int mb, int nb, double* c);
void mtmtimes(double* a, int ma, int na, double* b, int mb, int nb, double* c);
void mmttimes(double* a, int ma, int na, double* b, int mb, int nb, double* c);
void mvtimes(double* a, int m, int n, double* x, double* y);
void mstationaryDist(double* a, int n, double* y);
void mtrans(double* a, int m, int n);

void vplus(double* x, int n, double* y)
{
	cblas_daxpy(n, 1, x, 1, y, 1);
}

void vminus(double* x, int n, double* y)
{
	cblas_daxpy(n, -1, x, 1, y, 1);
}

void vmulti(double* x, int n, double* y)
{
	vdMul(n, x, y, y);
}

void mvrowmulti(double* x, int mx, int nx, double* y)
{
	/******************************************
	 * Extend a row vector to a matrix and multiply element by element
	 ******************************************/
	double* yExtended = new double[mx*nx];
	for (int i = 0; i < mx; ++i) {
		memcpy(yExtended + i*nx, y, nx*sizeof(double));
	}
	vmulti(yExtended, mx*nx, x);
}


void vaxpby(int n, double a, double* x, double b, double* y)
{
	cblas_daxpby(n, a, x, 1, b, y, 1);
}

void vcopy(double* x, int n, double* y) {
	cblas_dcopy(n, x, 1, y, 1);
}

void vcopyi(int* x, int n, int* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = x[i];
	}
}

void vrep(double* x, int n, int t, double* r)
{
	/* replicate x of length n for t times and store in r[n*t] */
	for (int i = 0; i < t; ++i) {
		vcopy(x, n, r + i*n);
	}
}

void vrrepmat(double* x, int t, double* r, int m, int n)
{
	/* repeat row vector x of lenth t m*n times and write to r */
	int i, j;
	/* copy to fill one row */
	for (i = 0; i < t*n; i += t)
	{
		cblas_dcopy(t, x, 1, r + i, 1);
	}
	/* copy to fill all rows */
	for (j = n; j < m*n; j += n)
	{
		cblas_dcopy(n, r, 1, r + j, 1);
	}
}

void mtrans(double* a, int m, int n)
{
	double* atemp = (double*)malloc(sizeof(double) * m * n);
	mkl_domatcopy('R', 'T', m, n, 1.0, a, n, atemp, m);
	memcpy(a, atemp, sizeof(double)*m*n);
	free(atemp);
}

void vcrepmat(double* x, int t, double* r, int m, int n)
{
	/* repeat column vector x of lenth t m*n times and write to r */
	/* conduct row copy first */
	vrrepmat(x, t, r, m, n);
}

void vstimes(double k, double* x, int n) {
	cblas_dscal(n, k, x, 1);
}

double vdot(double* x, int n, double* y)
{
	return cblas_ddot(n, x, 1, y, 1);
}

double vnorm(double* x, int n)
{
	return cblas_dasum(n, x, 1);
}

void vlinspace(double min, double max, int n, double* r)
{
	int i;
	double step = (max - min) / (n - 1);
	for (i = 0; i < n; i++) {
		r[i] = min + i*step;
	}
}

void vloglinspace(double min, double max, int n, double* r)
{
	int i;
	max = log(max);
	min = log(min);
	double step = (max - min) / (n - 1);
	for (i = 0; i < n; i++) {
		r[i] = exp(min + i*step);
	}
}

void mkronprod(double* x, int rx, int cx, double* y, int ry, int cy, double* r)
{
	for (int irx = 0; irx < rx; irx++) {
		for (int iry = 0; iry < ry; iry++) {
			for (int icx = 0; icx < cx; icx++) {
				for (int icy = 0; icy < cy; icy++) {
					r[(irx*ry + iry) * cx*cy + (icx*cy + icy)] =
						x[irx*cx + icx] * y[iry*cy + icy];
				}
			}
		}
	}
}

void vzeros(double* x, int n)
{
	memset(x, 0, sizeof(double) * n);
}

void vzerosi(int* x, int n)
{
	memset(x, 0, sizeof(int) * n);
}

void vones(double* x, int n)
{
	for (int i = 0; i < n; ++i)
	{
		x[i] = 1;
	}
}

void vonesi(int* x, int n)
{
	for (int i = 0; i < n; ++i)
	{
		x[i] = 1;
	}
}

void meye(double* x, int n)
{
	double* temp = (double*)malloc(n * sizeof(double));
	vones(temp, n);
	vzeros(x, n*n);
	cblas_dcopy(n, temp, 1, x, n + 1);
	free(temp);
}

void mcolsum(double *a, int m, int n, double* y)
{
	double* ones = (double*)malloc(n * sizeof(double));
	vones(ones, n);
	mvtimes(a, m, n, ones, y);
	free(ones);
}

void mrowsum(double *a, int m, int n, double* y)
{
	double* ones = (double*)malloc(m * sizeof(double));
	vones(ones, m);
	mmtimes(ones, 1, m, a, m, n, y);
	free(ones);
}

void mrowcsum(double* a, int m, int n)
{
	int i;
	double* y = (double*)malloc(m*n * sizeof(double));
	vcopy(a, n, y);
	for (i = 0; i < m - 1; i++) {
		mrowsum(a + i*n, 2, n, y + (i + 1)*n);
		vcopy(y + (i + 1)*n, n, a + (i + 1)*n);
	}
	free(y);
}

void mcolcsum(double* a, int m, int n)
{
	mtrans(a, m, n);
	mrowcsum(a, n, m);
	mtrans(a, n, m);
}

void mcolNormalized(double *a, int m, int n)
{
	/* calculate colsum first */
	double* colsum = (double*)malloc(m * sizeof(double));
	mcolsum(a, m, n, colsum);

	/* expand colsumExpand to the same size of a */
	double* colsumExpand = (double*)malloc(m*n * sizeof(double));
	double* ones = (double*)malloc(n * sizeof(double));
	vones(ones, n);

	mmtimes(colsum, m, 1, ones, 1, n, colsumExpand);
	vdDiv(m*n, a, colsumExpand, a);
	free(colsum);
	free(colsumExpand);
	free(ones);
}

void msolve(double* a, int m, int n, double* b, int nb, char trans)
{
	int* ipiv = (int*)malloc(MAX(1, MIN(m, n)) * sizeof(int));
	LAPACKE_dgetrf(101, m, n, a, n, ipiv);
	LAPACKE_dgetrs(101, trans, m, nb, a, n, ipiv, b, nb);
	free(ipiv);
}

void minv(double* a, int n)
{
	int* ipiv = (int*)malloc(n * sizeof(int));
	LAPACKE_dgetrf(101, n, n, a, n, ipiv);
	LAPACKE_dgetri(101, n, a, n, ipiv);
	free(ipiv);
}

void mmtimes(double* a, int ma, int na, double* b, int mb, int nb, double* c)
{
	assert(na == mb);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, ma, nb, na, 1.0, a, na, b, nb, 0.0, c, nb);
}

void mtmtimes(double* a, int ma, int na, double* b, int mb, int nb, double* c)
{
	assert(ma == mb);
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, na, nb, ma, 1.0, a, na, b, nb, 0.0, c, nb);
}

void mmttimes(double* a, int ma, int na, double* b, int mb, int nb, double* c)
{
	assert(na == nb);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ma, mb, na, 1.0, a, na, b, nb, 0.0, c, mb);
}

void mvtimes(double* a, int m, int n, double* x, double* y)
{
	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, a, n, x, 1, 0.0, y, 1);
}

void mstationaryDist(double* a, int n, double* y)
{
	double tol = 1e-8;
	double err = 1;

	double* oldx = (double*)malloc(n * sizeof(double));
	double* newx = y;
	vzeros(newx, n);
	vones(oldx, n);
	vstimes(1.0 / n, oldx, n);  // initiate prob vector

	while (err > tol)
	{
		mmtimes(oldx, 1, n, a, n, n, newx);

		vminus(newx, n, oldx);
		err = vnorm(oldx, n);

		vcopy(newx, n, oldx);
	}
	free(oldx);
}

#undef MAX
#undef MIN
