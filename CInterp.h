#pragma once

#define MAXDIM 10

namespace CInterp {
	/*
	* recursive evaluation doesn't allow compiler to optimize
	*/
	inline double receval4_1d(double* Coefs, double* XSite, int* CellOfSite, int Shift)
	{
		double r;
		double* pCoefs = Coefs + Shift + (*CellOfSite + 1) * 4;
		r = *(--pCoefs);
		r *= *XSite;
		r += *(--pCoefs);
		r *= *XSite;
		r += *(--pCoefs);
		r *= *XSite;
		r += *(--pCoefs);
		return r;
	}

	/*
	* recursive evaluation doesn't allow compiler to optimize
	*/
	inline double receval2_1d(double* Coefs, double* XSite, int* CellOfSite, int Shift)
	{
		double r;
		double* pCoefs = Coefs + Shift + (*CellOfSite + 1) * 2;
		r = *(--pCoefs);
		r *= *XSite;
		r += *(--pCoefs);
		return r;
	}

	inline double eval4_1d(double* XGrid, double* Coefs, int* CoefsSize,
		double* XSite, int* CellOfSite, int VecIdx)
	{
		double XSiteToLeft = *XSite - XGrid[*CellOfSite];
		return receval4_1d(Coefs, &XSiteToLeft, CellOfSite, VecIdx*CoefsSize[1]);
	}

	inline double receval4_1d_111(double* Coefs, double* XSite, int* CellOfSite, int Shift, double* g, double* h)
	{
		double r;

		double* pCoefs = Coefs + Shift + (*CellOfSite + 1) * 4;

		--pCoefs;
		r = *pCoefs;
		*g = *pCoefs * 3;
		*h = *pCoefs * 6;

		--pCoefs;
		r *= *XSite;
		r += *pCoefs;
		*g *= *XSite;
		*g += *pCoefs * 2;
		*h *= *XSite;
		*h += *pCoefs * 2;

		--pCoefs;
		r *= *XSite;
		r += *pCoefs;
		*g *= *XSite;
		*g += *pCoefs;

		--pCoefs;
		r *= *XSite;
		r += *pCoefs;
		return r;
	}

	inline double eval4_1d_111(double* XGrid, double* Coefs, int* CoefsSize,
		double* XSite, int* CellOfSite, int VecIdx, double* g, double* h)
	{
		double XSiteToLeft = *XSite - XGrid[*CellOfSite];
		return receval4_1d_111(Coefs, &XSiteToLeft, CellOfSite, VecIdx*CoefsSize[1], g, h);
	}

	/**
	* Evaluate cubic sline at one particular vector function recursively
	* \param[in] Coefs Coefficients of interpolation, order (vector function, X1, X2, ...).
	* \param[in] CoefsSize Size of coefficients of the above order.
	* \param[in] CurrentDim How many dimension left to be interpolated.
	* \param[in] XSite Evaluation site, already substracted from the left knot points.
	* \param[in] CellOfSite Which pieces does the i-th coordinate of XSite fall in.
	*/
	inline double receval4(double* Coefs, int* CoefsSize, int CurrentDim, double* XSite, int* CellOfSite, int Shift)
	{
		// recursive evaluation of a piece of coefficient at certain dimension
		double r;
		if (CurrentDim == 1) {
			// last dimension
			double* pCoefs = Coefs + Shift + (*CellOfSite + 1) * 4;
			r = *(--pCoefs);
			r *= *XSite;
			r += *(--pCoefs);
			r *= *XSite;
			r += *(--pCoefs);
			r *= *XSite;
			r += *(--pCoefs);
		}
		else {
			CurrentDim--;
			XSite++;
			Shift += (*CellOfSite + 1) * 4;
			CellOfSite++;
			CoefsSize++;
			r = receval4(Coefs, CoefsSize, CurrentDim, XSite, CellOfSite, (--Shift) * *CoefsSize);
			r *= *XSite;
			r += receval4(Coefs, CoefsSize, CurrentDim, XSite, CellOfSite, (--Shift) * *CoefsSize);
			r *= *XSite;
			r += receval4(Coefs, CoefsSize, CurrentDim, XSite, CellOfSite, (--Shift) * *CoefsSize);
			r *= *XSite;
			r += receval4(Coefs, CoefsSize, CurrentDim, XSite, CellOfSite, (--Shift) * *CoefsSize);
		}
		return r;
	}

	inline double eval4(int XDim, double** XGrid, double* Coefs, int* CoefsSize,
		double* XSite, int* CellOfSite, int VecIdx)
	{
		double XSiteToLeft[MAXDIM];
		for (int j = 0; j < XDim; ++j)
		{
			XSiteToLeft[j] = XSite[j] - XGrid[j][CellOfSite[j]];
		}
		return receval4(Coefs, CoefsSize, XDim, XSiteToLeft, CellOfSite, VecIdx*CoefsSize[1]);
	}

	inline void hunt(double xx[], int n, double x, int *jlo)
	{
		int jm, jhi, inc;

		if (*jlo <= 0 || *jlo > n) {
			*jlo = 0;
			jhi = n + 1;
		}
		else {
			inc = 1;
			if (x >= xx[*jlo]) {
				if (*jlo == n) return;
				jhi = (*jlo) + 1;
				while (x >= xx[jhi]) {
					*jlo = jhi;
					inc += inc;
					jhi = (*jlo) + inc;
					if (jhi > n) {
						jhi = n + 1;
						break;
					}
				}
			}
			else {
				if (*jlo == 1) {
					*jlo = 0;
					return;
				}
				jhi = (*jlo)--;
				while (x < xx[*jlo]) {
					jhi = (*jlo);
					inc <<= 1;
					if (inc >= jhi) {
						*jlo = 0;
						break;
					}
					else *jlo = jhi - inc;
				}
			}
		}
		while (jhi - (*jlo) != 1) {
			jm = (jhi + (*jlo)) >> 1;
			if (x >= xx[jm])
				*jlo = jm;
			else
				jhi = jm;
		}
		if (x == xx[n]) *jlo = n - 1;
		if (x == xx[1]) *jlo = 1;
	}

	inline void locate(double xx[], int n, double x, int *j)
	{
		int ju, jm, jl;

		jl = 0;
		ju = n + 1;
		while (ju - jl > 1) {
			jm = (ju + jl) >> 1;
			if (x >= xx[jm])
				jl = jm;
			else
				ju = jm;
		}
		if (x == xx[1]) *j = 1;
		else if (x == xx[n]) *j = n - 1;
		else *j = jl;
	}

	inline int locate2(double* xx, int n, double x)
	{
		// adjust xx[] to 1-based, 
		xx--;
		// return j if x \in [xx[j],xx[j+1])
		// exception, return n-1 if x==xx[n]; only return n if x > xx[n]
		int j;
		int ju, jm, jl;

		jl = 0;
		ju = n + 1;
		while (ju - jl > 1) {
			jm = (ju + jl) >> 1;
			if (x >= xx[jm])
				jl = jm;
			else
				ju = jm;
		}
		if (x == xx[1]) j = 1;
		else if (x == xx[n]) j = n - 1;
		else j = jl;

		return j;
	}

	/**** Matlab-alike Interface ***********/
	inline double search_eval4_1d(double* breaks, int nBreaks, double* coefs,
		double site, int idx)
	{
		int cellOfSite;
		locate(breaks, nBreaks - 2, site, &cellOfSite);
		double XSiteToLeft = site - breaks[cellOfSite];
		return receval4_1d(coefs, &XSiteToLeft, &cellOfSite, idx*(nBreaks - 1) * 4);
	}

	inline void search_1d(double* breaks, int nBreaks, double site, double* XSiteToLeft, int* cellOfSite)
	{
		locate(breaks, nBreaks - 2, site, cellOfSite);
		*XSiteToLeft = site - breaks[*cellOfSite];
	}
	
	inline double nosearch_eval2_1d(int nBreaks, double* coefs, double XSiteToLeft, int cellOfSite, int idx)
	{
		return receval2_1d(coefs, &XSiteToLeft, &cellOfSite, idx*(nBreaks - 1) * 2);
	}

	inline double search_eval2_1d(double* breaks, int nBreaks, double* coefs,
		double site, int idx)
	{
		int j;
		// locate(breaks, nBreaks - 2, site, &cellOfSite);
		int ju, jm, jl;
		int n = nBreaks - 1;

		jl = 0;
		ju = n + 1;
		while (ju - jl > 1) {
			jm = (ju + jl) >> 1;
			if (site >= breaks[jm])
				jl = jm;
			else
				ju = jm;
		}
		if (site == breaks[1]) j = 1;
		else if (site == breaks[n]) j = n - 1;
		else j = jl;

		double XSiteToLeft = site - breaks[j];

		// return receval2_1d(coefs, &XSiteToLeft, &j, idx*(nBreaks - 1) * 2);
		/*
		double r;
		double* pCoefs = Coefs + Shift + (*CellOfSite + 1) * 2;
		r = *(--pCoefs);
		r *= *XSite;
		r += *(--pCoefs);
		return r;
		*/

		int Shift = idx*(nBreaks - 1) * 2;
		double* pCoefs = coefs + Shift + j*2;
		double r = pCoefs[0] + pCoefs[1] * XSiteToLeft;
		// r = *(--pCoefs);
		// r *= *XSite;
		// r += *(--pCoefs);
		return r;
	}
}
