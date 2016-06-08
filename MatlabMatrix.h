/*!
 * \file MatlabMatrix.h
 * \date 2016/01/07 16:24
 *
 * \author Wenlan
 * Contact: luowenlan@gmail.com
 *
 * \brief A set of macros to easy access variables from matlab
 *
 * TODO: long description
 *
 * \note
*/

#include "blitz/array.h"

using namespace blitz;

#ifndef NDEBUG
#define MMDEBUG
#endif

#define MACRO_GET_1(str, i) \
    (sizeof(str) > (i) ? str[(i)] : 0)

#define MACRO_GET_4(str, i) \
    MACRO_GET_1(str, i+0),  \
    MACRO_GET_1(str, i+1),  \
    MACRO_GET_1(str, i+2),  \
    MACRO_GET_1(str, i+3)

#define MACRO_GET_16(str, i) \
    MACRO_GET_4(str, i+0),   \
    MACRO_GET_4(str, i+4),   \
    MACRO_GET_4(str, i+8),   \
    MACRO_GET_4(str, i+12)

#define MACRO_GET_64(str, i) \
    MACRO_GET_16(str, i+0),  \
    MACRO_GET_16(str, i+16), \
    MACRO_GET_16(str, i+32), \
    MACRO_GET_16(str, i+48)

#define MACRO_GET_STR(str) MACRO_GET_64(str, 0), 0 //guard for longer strings

#define MMARRAY_OUT_OF_DIM mexErrMsgIdAndTxt("MMArray:OutOfDim","Access out of dimension, var declared at LINE %d", lineNum);
#define MMARRAY_OUT_OF_BOUND mexErrMsgIdAndTxt("MMArray:OutOfBound","Access out of bound, var declared at LINE %d", lineNum);
#define MMARRAY_BOUND_CHK_0 && i0 >= base(0) && (i0 - base(0)) < length_[0]
#define MMARRAY_BOUND_CHK_1 MMARRAY_BOUND_CHK_0 \
				&& i1 >= base(1) && (i1 - base(1)) < length_[1]
#define MMARRAY_BOUND_CHK_2 MMARRAY_BOUND_CHK_1 \
				&& i2 >= base(2) && (i2 - base(2)) < length_[2]
#define MMARRAY_BOUND_CHK_3 MMARRAY_BOUND_CHK_2 \
				&& i3 >= base(3) && (i3 - base(3)) < length_[3]
#define MMARRAY_BOUND_CHK_4 MMARRAY_BOUND_CHK_3 \
				&& i4 >= base(4) && (i4 - base(4)) < length_[4]
#define MMARRAY_BOUND_CHK_5 MMARRAY_BOUND_CHK_4 \
				&& i5 >= base(5) && (i5 - base(5)) < length_[5]
#define MMARRAY_BOUND_CHK_6 MMARRAY_BOUND_CHK_5 \
				&& i6 >= base(6) && (i6 - base(6)) < length_[6]
#define MMARRAY_BOUND_CHK_7 MMARRAY_BOUND_CHK_6 \
				&& i7 >= base(7) && (i7 - base(7)) < length_[7]
#define MMARRAY_BOUND_CHK_8 MMARRAY_BOUND_CHK_7 \
				&& i8 >= base(8) && (i8 - base(8)) < length_[8]
#define MMARRAY_BOUND_CHK_9 MMARRAY_BOUND_CHK_8 \
				&& i9 >= base(9) && (i9 - base(9)) < length_[9]
#define MMARRAY_BOUND_CHK_10 MMARRAY_BOUND_CHK_9 \
				&& i10 >= base(10) && (i10 - base(10)) < length_[10]

#if defined(MMDEBUG)
#define MMARRAY_BOUND_CHK_N(N) \
if (!(dim == N+1)) \
MMARRAY_OUT_OF_DIM; \
if (!(1 \
	MMARRAY_BOUND_CHK_##N \
	)) \
	MMARRAY_OUT_OF_BOUND;
#else
#define MMARRAY_BOUND_CHK_N(N)
#endif

namespace MatlabMatrix {
	template<typename T_numtype, int dim, int lineNum>
	class MMArray : public Array<T_numtype, dim>
	{
	public:
		using Array::Array;

		const T_numtype& restrict operator()(int i0) const
		{
			MMARRAY_BOUND_CHK_N(0);
			return data_[(i0)* stride_[0]];
		}

		T_numtype& restrict operator()(int i0)
		{
			MMARRAY_BOUND_CHK_N(0);
			return data_[(i0)* stride_[0]];
		}

		const T_numtype& restrict operator()(int i0, int i1) const
		{
			MMARRAY_BOUND_CHK_N(1);
			return data_[(i0)* stride_[0] + i1 * stride_[1]];
		}

		T_numtype& restrict operator()(int i0, int i1)
		{
			MMARRAY_BOUND_CHK_N(1);
			return data_[(i0)* stride_[0] + i1 * stride_[1]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2) const
		{
			MMARRAY_BOUND_CHK_N(2);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2)
		{
			MMARRAY_BOUND_CHK_N(2);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2, int i3) const
		{
			MMARRAY_BOUND_CHK_N(3);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2, int i3)
		{
			MMARRAY_BOUND_CHK_N(3);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4) const
		{
			MMARRAY_BOUND_CHK_N(4);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4)
		{
			MMARRAY_BOUND_CHK_N(4);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5) const
		{
			MMARRAY_BOUND_CHK_N(5);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5)
		{
			MMARRAY_BOUND_CHK_N(5);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6) const
		{
			MMARRAY_BOUND_CHK_N(6);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6)
		{
			MMARRAY_BOUND_CHK_N(6);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6, int i7) const
		{
			MMARRAY_BOUND_CHK_N(7);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6, int i7)
		{
			MMARRAY_BOUND_CHK_N(7);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6, int i7, int i8) const
		{
			MMARRAY_BOUND_CHK_N(8);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
				+ i8 * stride_[8]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6, int i7, int i8)
		{
			MMARRAY_BOUND_CHK_N(8);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
				+ i8 * stride_[8]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6, int i7, int i8, int i9) const
		{
			MMARRAY_BOUND_CHK_N(9);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
				+ i8 * stride_[8] + i9 * stride_[9]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6, int i7, int i8, int i9)
		{
			MMARRAY_BOUND_CHK_N(9);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
				+ i8 * stride_[8] + i9 * stride_[9]];
		}

		const T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6, int i7, int i8, int i9, int i10) const
		{
			MMARRAY_BOUND_CHK_N(10);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
				+ i8 * stride_[8] + i9 * stride_[9] + i10 * stride_[10]];
		}

		T_numtype& restrict operator()(int i0, int i1, int i2, int i3,
			int i4, int i5, int i6, int i7, int i8, int i9, int i10)
		{
			MMARRAY_BOUND_CHK_N(10);
			return data_[(i0)* stride_[0] + i1 * stride_[1]
				+ i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4]
				+ i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7]
				+ i8 * stride_[8] + i9 * stride_[9] + i10 * stride_[10]];
		}
	};

	template <class T> class Vector {
	public:
		int pts;
		T* data;
		Vector(int _pts) :pts(_pts) {}
		Vector(int _pts, T* _data) :pts(_pts), data(_data - 1) {}
		Vector(T* _data) : pts(0), data(_data - 1) {}
		T& operator()  (int idx)
		{
			// 1 based
			return data[idx];
		}

		const T& operator() (int idx) const
		{
			// 1 based
			return data[idx];
		}

		void operator >> (T* dest)
		{
			memcpy(dest, data + 1, sizeof(T)*pts);
		}

		void operator << (T* dest)
		{
			memcpy(data + 1, dest, sizeof(T)*pts);
		}
	};
}


#define DES(var) mxDestroyArray(__##var)
#define GET_INT(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsDouble(__##var)) mexErrMsgTxt("Not double: "#var); \
	int var = (int)*mxGetPr(__##var);

#define GET_DBL(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if (__##var == 0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsDouble(__##var)) mexErrMsgTxt("Not double: "#var); \
	double var = *mxGetPr(__##var);

#define GET_SGL(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if (__##var == 0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsDouble(__##var)) mexErrMsgTxt("Not double: "#var); \
	float var = (float)*mxGetPr(__##var);

#define GET_DMAT0(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsDouble(__##var)) mexErrMsgTxt("Not double: "#var); \
	double* _##var = mxGetPr(__##var)

#define GET_DMAT0_VIEW(var) const mxArray* __##var = mexGetVariablePtr("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsDouble(__##var)) mexErrMsgTxt("Not double: "#var); \
	double* _##var = mxGetPr(__##var)

#define GET_FMAT0(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsSingle(__##var)) mexErrMsgTxt("Not single: "#var); \
	float* _##var = (float*) mxGetData(__##var)

#define GET_FMAT0_VIEW(var) const mxArray* __##var = mexGetVariablePtr("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsSingle(__##var)) mexErrMsgTxt("Not single: "#var); \
	float* _##var = (float*) mxGetData(__##var)

#define GET_IMAT0(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsInt32(__##var)) mexErrMsgTxt("Not int32: "#var); \
	int* _##var = (int*) mxGetData(__##var)

#define GET_IMAT0_VIEW(var) const mxArray* __##var = mexGetVariablePtr("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsInt32(__##var)) mexErrMsgTxt("Not int32: "#var); \
	int* _##var = (int*) mxGetData(__##var)

// GET_DMAT gets data from matlab caller's workspace. One needs to specify number of dimensions and each dimension.
#define GET_DMAT(var,N,...) GET_DMAT0(var); \
	if (mxGetNumberOfDimensions(__##var)!=N) mexErrMsgTxt("No. of Dimensions Error: "#var); \
	Array<double, N> var(_##var, shape(__VA_ARGS__), neverDeleteData, ColumnMajorArray<N>()); \
	for (int i = 0; i < N ; i++) \
																	{ \
		if (*(mxGetDimensions(__##var)+i) != var.extent(i)) { \
			char errMsg[100]; \
			sprintf(errMsg,"Dimension Error: "#var" at //d",i); \
			mexErrMsgTxt(errMsg); \
																															} \
														}
// GET_DM is the most convenient way to get data from matlab caller's workspace. One just needs to specify the number of dimenions.
#define GET_DM(var,N) GET_DMAT0(var); \
	if (mxGetNumberOfDimensions(__##var)!=N) mexErrMsgTxt("No. of Dimensions Error: "#var); \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,N> _##var##dim; \
	for (int i = 0; i < N ; i++) \
		{ \
	_##var##dim[i] = *(mxGetDimensions(__##var)+i); \
		} \
	MatlabMatrix::MMArray<double, N, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

#define GET_DM_VIEW(var,N) GET_DMAT0_VIEW(var); \
	if (mxGetNumberOfDimensions(__##var)!=N) mexErrMsgTxt("No. of Dimensions Error: "#var); \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,N> _##var##dim; \
	for (int i = 0; i < N ; i++) \
		{ \
	_##var##dim[i] = *(mxGetDimensions(__##var)+i); \
		} \
	MatlabMatrix::MMArray<double, N, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

#define GET_FM(var,N) GET_FMAT0(var); \
	if (mxGetNumberOfDimensions(__##var)!=N) mexErrMsgTxt("No. of Dimensions Error: "#var); \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,N> _##var##dim; \
	for (int i = 0; i < N ; i++) \
		{ \
	_##var##dim[i] = *(mxGetDimensions(__##var)+i); \
		} \
	MatlabMatrix::MMArray<float, N, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

#define GET_FM_VIEW(var,N) GET_FMAT0_VIEW(var); \
	if (mxGetNumberOfDimensions(__##var)!=N) mexErrMsgTxt("No. of Dimensions Error: "#var); \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,N> _##var##dim; \
	for (int i = 0; i < N ; i++) \
		{ \
	_##var##dim[i] = *(mxGetDimensions(__##var)+i); \
		} \
	MatlabMatrix::MMArray<float, N, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

#define GET_IM(var,N) GET_IMAT0(var); \
	if (mxGetNumberOfDimensions(__##var)!=N) mexErrMsgTxt("No. of Dimensions Error: "#var); \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,N> _##var##dim; \
	for (int i = 0; i < N ; i++) \
		{ \
	_##var##dim[i] = *(mxGetDimensions(__##var)+i); \
		} \
	MatlabMatrix::MMArray<int, N, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

#define GET_IM_VIEW(var,N) GET_IMAT0_VIEW(var); \
	if (mxGetNumberOfDimensions(__##var)!=N) mexErrMsgTxt("No. of Dimensions Error: "#var); \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,N> _##var##dim; \
	for (int i = 0; i < N ; i++) \
		{ \
	_##var##dim[i] = *(mxGetDimensions(__##var)+i); \
		} \
	MatlabMatrix::MMArray<int, N, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

// GET_DV is routine to treat input as a one dimensional vector
// Since matlab always treats vector as a matrix, sometimes forcing it to be one-dimenion has great convenience for indexing
#define GET_DV(var) GET_DMAT0(var); \
	GeneralArrayStorage<1> _##var##storage = ColumnMajorArray<1>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,1> _##var##dim; \
	_##var##dim[0] = mxGetNumberOfElements(__##var); \
	MatlabMatrix::MMArray<double, 1, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

/*
#define GET_DV_VIEW(var) GET_DMAT0_VIEW(var); \
	GeneralArrayStorage<1> _##var##storage = ColumnMajorArray<1>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,1> _##var##dim; \
	_##var##dim[0] = mxGetNumberOfElements(__##var); \
	Array<double, 1> var(_##var,_##var##dim,neverDeleteData,_##var##storage);
	*/
#define GET_DV_VIEW(var) GET_DMAT0_VIEW(var); \
	MatlabMatrix::DV var(_##var)

#define GET_FV(var) GET_FMAT0(var); \
	GeneralArrayStorage<1> _##var##storage = ColumnMajorArray<1>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,1> _##var##dim; \
	_##var##dim[0] = mxGetNumberOfElements(__##var); \
	MatlabMatrix::MMArray<float, 1, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

#define GET_FV_VIEW(var) GET_FMAT0_VIEW(var); \
	GeneralArrayStorage<1> _##var##storage = ColumnMajorArray<1>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,1> _##var##dim; \
	_##var##dim[0] = mxGetNumberOfElements(__##var); \
	MatlabMatrix::MMArray<float, 1, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

#define GET_IV_VIEW(var) GET_IMAT0_VIEW(var); \
	GeneralArrayStorage<1> _##var##storage = ColumnMajorArray<1>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,1> _##var##dim; \
	_##var##dim[0] = mxGetNumberOfElements(__##var); \
	MatlabMatrix::MMArray<int, 1, __LINE__> var(_##var,_##var##dim,neverDeleteData,_##var##storage);


	// SET_DM is the most convenient way to prepare output data to matlab.
#define SET_DMAT(var,N,...) \
	mxArray* __##var; \
	{ \
	mwSize dim[] = {__VA_ARGS__}; \
	__##var = mxCreateNumericArray(N, dim, mxDOUBLE_CLASS, mxREAL); \
	} \
	double* _##var = mxGetPr(__##var); \
	MatlabMatrix::MMArray<double, N, __LINE__> var(_##var, shape(__VA_ARGS__), neverDeleteData, ColumnMajorArray<N>());

// PUT is put the variable to matlab's caller workspace
// #define PUT(var,...)  memcpy(_##var, var.data(), sizeof(double) * var.numElements()); mexPutVariable("caller",#var,__##var);
#define PUT(var,...) mexPutVariable("caller",#var,__##var);
// PUT_ is to put the variable to matlab's caller workspace under the name var_
#define PUT_(var,...)  mexPutVariable("caller",#var"_",__##var);

#define PUT_SCALAR(var,...) {mxArray* __##var = mxCreateDoubleScalar( (double) var); mexPutVariable( "caller", #var, __##var );}

// DV is a double vector
#define DV Vector<double>
#define IV Vector<int>


// DM is to create view of double array
#define DM(var,data,N,...) \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	Array<double,N> var(data,shape(__VA_ARGS__),neverDeleteData,_##var##storage);

#define DM0(var,N,...) \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	Array<double,N> var(__VA_ARGS__,_##var##storage); \
	memset(var.data(),0,sizeof(double)*var.numElements());

// FM is to create view of single array
#define FM(var,data,N,...) \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	Array<float,N> var(data,shape(__VA_ARGS__),neverDeleteData,_##var##storage);

// IM is to declare integer array
// #define IM(N) Array<int,N>
#define IM0(var,N,...) \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	Array<int,N> var(__VA_ARGS__,_##var##storage); \
	memset(var.data(),0,sizeof(int)*var.numElements());

#define ALL Range::all()

// Zerofy any Array
template<class T, int dim> void zeros(blitz::Array<T, dim> v)
{
	memset(v.data(), 0, sizeof(T)*v.numElements());
}

// Run matlab code
#define MATLAB(command) \
	mexEvalString(#command)
