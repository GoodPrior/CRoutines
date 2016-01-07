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

#define DES(var) mxDestroyArray(__##var)
#define GET_INT(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsDouble(__##var)) mexErrMsgTxt("Not double: "#var); \
	int var = (int)*mxGetPr(__##var);

#define GET_DBL(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if (__##var == 0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsDouble(__##var)) mexErrMsgTxt("Not double: "#var); \
	double var = *mxGetPr(__##var);

#define GET_DMAT0(var) mxArray* __##var = mexGetVariable("caller",#var); \
	if(__##var==0) mexErrMsgTxt("Variable doesn't exist: "#var); \
	if (!mxIsDouble(__##var)) mexErrMsgTxt("Not double: "#var); \
	double* _##var = mxGetPr(__##var)

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
	Array<double, N> var(_##var,_##var##dim,neverDeleteData,_##var##storage);

// GET_DV is routine to treat input as a one dimensional vector
// Since matlab always treats vector as a matrix, sometimes forcing it to be one-dimenion has great convenience for indexing
#define GET_DV(var) GET_DMAT0(var); \
	GeneralArrayStorage<1> _##var##storage = ColumnMajorArray<1>(); \
	_##var##storage.base() = 1; \
	TinyVector<int,1> _##var##dim; \
	Array<double, 1> var(_##var,_##var##dim,neverDeleteData,_##var##storage);


// SET_DM is the most convenient way to prepare output data to matlab.
#define SET_DMAT(var,N,...) \
	mxArray* __##var; \
	{ \
	mwSize dim[] = {__VA_ARGS__}; \
	__##var = mxCreateNumericArray(N, dim, mxDOUBLE_CLASS, mxREAL); \
	} \
	double* _##var = mxGetPr(__##var); \
	Array<double, N> var(_##var, shape(__VA_ARGS__), neverDeleteData, ColumnMajorArray<N>());

// PUT is put the variable to matlab's caller workspace
#define PUT(var,...)  memcpy(_##var, var.data(), sizeof(double) * var.numElements()); mexPutVariable("caller",#var,__##var);
// PUT_ is to put the variable to matlab's caller workspace under the name var_
#define PUT_(var,...)  memcpy(_##var, var.data(), sizeof(double) * var.numElements()); mexPutVariable("caller",#var"_",__##var);

// DV is a double vector
template <class T> class Vector{
public:
	int pts;
	T* data;
	Vector(int _pts) :pts(_pts) {}
	Vector(int _pts, T* _data) :pts(_pts), data(_data - 1){}
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
#define DV Vector<double>
#define IV Vector<int>


// DM is to declare double array
// #define DM(N) Array<double,N>
#define DM0(var,N,...) \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	Array<double,N> var(__VA_ARGS__,_##var##storage); \
	memset(var.data(),0,sizeof(double)*var.numElements());

// DI is to declare integer array
// #define IM(N) Array<int,N>
#define IM0(var,N,...) \
	GeneralArrayStorage<N> _##var##storage = ColumnMajorArray<N>(); \
	_##var##storage.base() = 1; \
	Array<int,N> var(__VA_ARGS__,_##var##storage); \
	memset(var.data(),0,sizeof(int)*var.numElements());

#define ALL Range::all()
