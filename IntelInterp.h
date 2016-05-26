#pragma once

#include <cstdlib>
#include <cstring>
#include "mkl.h"
#include "blitz/array.h"

using namespace blitz;

#define MAXDIM 10

namespace IntelInterp {
	inline void int_memcpy(int* des, int* src, int n)
	{
		memcpy(des, src, sizeof(int) * n);
	}

	inline void double_memcpy(double* des, double* src, int n)
	{
		memcpy(des, src, sizeof(double) * n);
	}

	inline int vector_prod(int* x, int n) {
		int s = x[0];
		for (int i = 1; i < n; ++i) {
			s *= x[i];
		}
		return s;
	}

	inline void construct(int XDim, int* XPts, double** XGrid, int VecDim, double* Value, int* SOrder, double* Coefs, int CoefsMemorySize) {
		double* LastCoefs = (double*)malloc(sizeof(double) * CoefsMemorySize);
		int LastCoefsSize[MAXDIM];
		int LastCoefsDim;

		// for the first coordinate, the LastCoefs is just Value
		int InitialCoefsSize[MAXDIM + 1];
		InitialCoefsSize[0] = VecDim;
		for (int i = 0; i < XDim; ++i) {
			InitialCoefsSize[i + 1] = XPts[i];
		}

		int CoefsSize[MAXDIM];
		int_memcpy(CoefsSize, InitialCoefsSize, XDim + 1);
		CoefsMemorySize = vector_prod(CoefsSize, XDim + 1);
		double_memcpy(Coefs, Value, CoefsMemorySize);

		// the following is a mimic of csape in Matlab
		MKL_INT s_type[MAXDIM];
		MKL_INT bc_type[MAXDIM];
		for (int i = 0; i < XDim; ++i) {
			switch (SOrder[i]) {
			case 4:
				s_type[i] = DF_PP_NATURAL;
				bc_type[i] = DF_BC_NOT_A_KNOT;
				break;
			case 2:
				s_type[i] = DF_PP_DEFAULT;
				bc_type[i] = DF_NO_BC;
				break;
			}
		}

		// Let me do MKL Interp at low level to boost performance
		DFTaskPtr Interp0;
		int CurrentVecSize = 1; // not used
		for (int i = XDim - 1; i >= 0; --i) {
			// carry out coordinatewise interpolation at coordinate i
			// the interpolation is always w.r.t. the last coordinate, w.r.t. to which y are stored adjacently
			// calculate new size of vector function, i.e. the total size except the last dimension
			CurrentVecSize = vector_prod(CoefsSize, XDim);

			// compute the new size of coeffs
			// the last dimension reduces by 1, and times by s_order
			CoefsSize[XDim] = CoefsSize[XDim] - 1;
			CoefsSize[XDim] = CoefsSize[XDim] * SOrder[i];
			// update last Coeff
			int_memcpy(LastCoefsSize, CoefsSize, XDim + 1);
			LastCoefsDim = vector_prod(LastCoefsSize, XDim + 1);

			// compute coefficients
			dfdNewTask1D(&Interp0, XPts[i], XGrid[i], DF_NO_HINT, CurrentVecSize, Coefs, DF_NO_HINT);
			dfdEditPPSpline1D(Interp0, SOrder[i], s_type[i], bc_type[i], 0, DF_NO_IC, 0, LastCoefs, DF_NO_HINT);
			dfdConstruct1D(Interp0, DF_PP_SPLINE, DF_METHOD_STD);
			dfDeleteTask(&Interp0);

			// forming new interpolation problem, by permuting the last dimension to the first one
			if (XDim > 1) {
				int vecShift = LastCoefsDim / VecDim;

				double* des = Coefs;
				double* src = LastCoefs;
				int transRow = vecShift / LastCoefsSize[XDim];
				int transCol = LastCoefsSize[XDim];
				for (int i = 0; i < VecDim; ++i) {
					mkl_domatcopy('R', 'T', transRow, transCol, 1.0, src, transCol, des, transRow);
					des += vecShift;
					src += vecShift;
				}
				// swap dimension
				int temp = LastCoefsSize[XDim];
				for (int i = XDim; i >= 2; --i) {
					LastCoefsSize[i] = LastCoefsSize[i - 1];
				}

				LastCoefsSize[1] = temp;
				int_memcpy(CoefsSize, LastCoefsSize, XDim + 1);
				CoefsMemorySize = vector_prod(CoefsSize, XDim + 1);
			}
			else {
				double_memcpy(Coefs, LastCoefs, LastCoefsDim);
			}
		}

		free(LastCoefs);
	}

	/**
	* compute coefficient memory size
	*/
	inline int compute_coefs_size(int XDim, int* XPts, double** XGrid, int VecDim, int* SOrder, int* CoefsSize)
	{
		// compute size for coefficients
		CoefsSize[0] = VecDim;
		for (int i = 0; i < XDim; ++i) {
			CoefsSize[i + 1] = (XPts[i] - 1) * SOrder[i];
		}
		return vector_prod(CoefsSize, XDim + 1);
	}

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

	// mkl evaluation for full vector function is still much faster
	inline void eval_mkl(int* XPts, double** XGrid, int VecDim, int* SOrder, double* Coefs,
		double* XSite, double* InterpValue)
	{
		DFTaskPtr interp;
		double tempy;
		dfdNewTask1D(&interp, XPts[0], XGrid[0], DF_NO_HINT, VecDim, &tempy, DF_NO_HINT);
		MKL_INT s_type;
		MKL_INT bc_type;
		switch (SOrder[0]) {
		case 4:
			s_type = DF_PP_NATURAL;
			bc_type = DF_BC_NOT_A_KNOT;
			break;
		case 2:
			s_type = DF_PP_DEFAULT;
			bc_type = DF_NO_BC;
			break;
		}
		dfdEditPPSpline1D(interp, SOrder[0], s_type, bc_type, 0, DF_NO_IC, 0, Coefs, DF_NO_HINT);
		int dorder = 1;
		dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, 1, XSite, DF_NO_HINT, 1, &dorder, NULL, InterpValue, DF_NO_HINT, 0);
		dfDeleteTask(&interp);
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

	inline double search_1d(double* breaks, int nBreaks, double site, double* XSiteToLeft, int* cellOfSite)
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


#ifdef INTERP_CLASS
	class CubicSplineRaw
	{
	public:
		int mXDim;
		CubicSplineRaw& set_XDim(int XDim) { mXDim = XDim; return *this; }

		int mVecDim;
		CubicSplineRaw& set_VecDim(int VecDim) { mVecDim = VecDim; return *this; }

		int mCoefsMemorySize;
		CubicSplineRaw& set_CoefsMemorySize(int CoefsMemorySize) { mCoefsMemorySize = CoefsMemorySize; return *this; }

		int mCoefsSize[MAXDIM];
		CubicSplineRaw& set_CoefsSize(int* CoefsSize) { memcpy(mCoefsSize, CoefsSize, MAXDIM*sizeof(int)); return *this; }

		int mSOrder[MAXDIM];
		CubicSplineRaw& set_SOrder(int* SOrder) { memcpy(mSOrder, SOrder, MAXDIM*sizeof(int)); return *this; }

		int mXPts[MAXDIM];
		CubicSplineRaw& set_XPts(int* XPts) { memcpy(mXPts, XPts, MAXDIM*sizeof(int)); return *this; }

		double* mXGrid[MAXDIM];
		CubicSplineRaw& set_XGrid(double** XGrid) { memcpy(mXGrid, XGrid, MAXDIM*sizeof(double*)); return *this; }

		double* mCoefs;
		CubicSplineRaw& set_Coefs(double* Coefs) { mCoefs = Coefs; return *this; }

		inline void alloc()
		{
			check();
			assert(!mCoefs);
			mCoefsMemorySize = IntelInterp::compute_coefs_size(mXDim, mXPts, mXGrid, mVecDim, mSOrder, mCoefsSize);
			mCoefs = (double*)malloc(mCoefsMemorySize * sizeof(double));
		}

		inline void dealloc()
		{
			free(mCoefs);
		}

		inline void locate(double* XSite, int* CellOfSite)
		{
			check();
			for (int j = 0; j < mXDim; ++j)
				IntelInterp::locate(mXGrid[j], mXPts[j] - 2, XSite[j], CellOfSite + j);
		}

		inline void hunt(double* XSite, int* CellOfSite)
		{
			check();
			for (int j = 0; j < mXDim; ++j)
				IntelInterp::hunt(mXGrid[j], mXPts[j] - 2, XSite[j], CellOfSite + j);
		}

		inline void eval_mkl(double* XSite, double* InterpValue)
		{
			check();
			assert(mCoefs);
			assert(mCoefsMemorySize);
			IntelInterp::eval_mkl(mXPts, mXGrid, mVecDim, mSOrder, mCoefs, XSite, InterpValue);
		}

		inline double eval_1d_111(double* XSite, int* CellOfSite, int VecIdx, double* g, double* h)
		{
			check();
			assert(mCoefs);
			assert(mCoefsMemorySize);
			return IntelInterp::eval4_1d_111(mXGrid[0], mCoefs, mCoefsSize, XSite, CellOfSite, VecIdx, g, h);
		}

		inline double eval_1d(double* XSite, int* CellOfSite, int VecIdx)
		{
			check();
			assert(mCoefs);
			assert(mCoefsMemorySize);
			return IntelInterp::eval4_1d(mXGrid[0], mCoefs, mCoefsSize, XSite, CellOfSite, VecIdx);
		}

		inline double eval(double* XSite, int* CellOfSite, int VecIdx)
		{
			check();
			assert(mCoefs);
			assert(mCoefsMemorySize);
			return IntelInterp::eval4(mXDim, mXGrid, mCoefs, mCoefsSize,
				XSite, CellOfSite, VecIdx);
		}

		void construct(double* Value)
		{
			check();
			assert(mCoefs);
			assert(mCoefsMemorySize);

			IntelInterp::construct(mXDim, mXPts, mXGrid, mVecDim, Value, mSOrder, mCoefs, mCoefsMemorySize);
		}

		inline void check()
		{
			assert(mXDim);
			assert(mVecDim);
			assert(mSOrder);
			assert(mXPts);
			assert(mXGrid[0]);
		}

	protected:
		inline void veceval_uniform_x(int NumOfSites, double* Sites, int* CellOfSite, int NumOfVec, int* VecIdx, double* InterpValue)
		{
			for (int i = 0; i < NumOfSites; ++i)
			{
				hunt(Sites + i*mXDim, CellOfSite + i*mXDim);
				for (int j = 0; j < NumOfVec; ++j)
				{
					InterpValue[i*NumOfVec + j] = eval(Sites + i*mXDim, CellOfSite + i*mXDim, VecIdx[j]);
				}
			}
		}

		void veceval_continuous(int NumOfSites, double* Sites, int* CellOfSite, int StartIdx, int EndIdx, double* InterpValue);

		CubicSplineRaw(){};
		~CubicSplineRaw(){};
	};

	class CubicSpline :
		public CubicSplineRaw
	{
	public:
		CubicSpline(){};
		~CubicSpline(){};

		inline Array<double, 2> veceval_uniform_x(Array<double, 2>* Sites, Array<int, 2>* CellOfSite, Array<int, 2>* VecIdx)
		{
			assert(Sites->rows() == VecIdx->rows());
			assert(Sites->rows() == CellOfSite->rows());
			assert(Sites->cols() == CellOfSite->cols());

			Array<double, 2> InterpValue(Sites->rows(), VecIdx->cols());
			CubicSplineRaw::veceval_uniform_x(Sites->rows(), Sites->data(), CellOfSite->data(), VecIdx->cols(), VecIdx->data(), InterpValue.data());
			return InterpValue;
		}

		Array<double, 2> veceval_continuous(Array<double, 2>* Sites, Array<int, 2>* CellOfSite, int StartIdx, int EndIdx);
	};
#endif
}
