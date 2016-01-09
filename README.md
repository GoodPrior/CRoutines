# MatlabMatrix.h - a set of macros to convert matlab code to mex

## Introduction
We prototype models in matlab and translate the time-consuming part to CMex for production.
Translation is sometimes painful. For a multi-dimension array, usually
we need to mannualy extract the data pointer, the dimension information,
check consistency, deal with strides, construct output etc..

MatlabMatrix.h provides several macros to automate these processes.
The basic uses can be illustrated as below.

The matlab code:
```matlab
function Test
Dim1 = 10;
Dim2 = 20;
Dim3 = 30;

rng(0823);
A = rand(Dim1,Dim2,Dim3);
B = rand(Dim1,Dim2);
C = zeros(Dim1,Dim2,Dim3); % intentionally allocate space in advance for mex call

C = DoSomething(A,B,Dim1,Dim2,Dim3);
end

function C = DoSomething(A,B,Dim1,Dim2,Dim3)
C = zeros(Dim1,Dim2,Dim3);
for i3=1:Dim3
    for i2=1:Dim2
        for i1=1:Dim1
            C(i1,i2,i3) = A(i1,i2,i3) * B(i1,i2);
        end
        C(:,i2,i3) = A(:,i2,i3);
    end
end
end
```

With MatlabMatrix.h, the function DoSomething() and the line calling DoSomething()
```matlab
C = DoSomething(A,B,Dim1,Dim2,Dim3);
```

can be replaced by the following mex file:
```C++
#include "MatlabMatrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	GET_INT(Dim1);
	GET_INT(Dim2);
	GET_INT(Dim3);
	GET_DM(A, 3);
	GET_DM(B, 2);
	GET_DM(C, 3);

	for (int i3 = 1; i3 <= Dim3 ; i3++)
	{
		for (int i2 = 1; i2 <= Dim2 ; i2++)
		{
			for (int i1 = 1; i1 <= Dim1 ; i1++)
			{
				C(i1, i2, i3) = A(i1, i2, i3)*B(i1, i2);
			}
			C(ALL, i2, i3) = A(ALL, i2, i3);
		}
	}

	PUT(C);
}
```

One may have noticed some features in the mex file:

1. Arrays are assesed with braces (instead of brackets), 1-based (instead of 0-based),
column-major storage (instead of row-major). This keeps things close to matlab
and therefore most of codes can be directly copied.
2. With keyword ALL, slicing can be done similary as : in matlab.
3. Output variable (C here) should be constructed beforehand. And a PUT(C) is used to post output to the workspace.

## Documentation and Implementation
These macros greatly facilicate production, but they are compactly implemented.
Therefore it's convenient to describe the usage of each macro and make sure users understand what's behind.

The key implementation is through (1) [Blitz++](http://sourceforge.net/projects/blitz/),
a powerful (and the best IMO) array template in C++. (2) The mex function mexGetVariable(), mexPutVariable().

#### GET_DBL(var), GET_INT(var)
These two macros get variables var from the caller workspace,
convert it to double (GET_DBL()) or int (GET_INT()),
and assign it to a scalar with same name in C++ code.

More specifically, GET_DBL(var) does the following:
```c++
mxArray* __var = mexGetVariable("caller","var");
if (__var == 0) mexErrMsgTxt("Variable doesn't exist: var");
if (!mxIsDouble(__var)) mexErrMsgTxt("Not double: var");
double var = *mxGetPr(__var);
```
Similary for GET_INT(). Notice GET_INT() still reads a double scalar from matlab,
and only converts it to integer in C. (Matlab default type is double, and we usually do not
convert it to (int32) before usage even if it's used as int).

#### GET_DM(var, numOfDims), GET_IM(var, numOfDims)
GET_DM is short for (get a double matrix).
These two macros get a multi-dimension array with number of dimensions = numOfDims from the caller workspace,
and assign it to an array with same name in C++ code. The numOfDims
must be statically determined at compilation time, and can't excceed 11.

The multi-dimension array is actually a blitz array object, and has many great features.
But the most useful features are perhaps
indexing and slicing as shown in the example.
```c++
...
C(i1, i2, i3) = A(i1, i2, i3)*B(i1, i2);
...
C(ALL, i2, i3) = A(ALL, i2, i3)*B(i1, i2);
...
```

As expected, the indexing returns a reference and therefore the array object can be used
as helper class
to facilitate memory access as well. With a simple class
Vector<T> provided with the header file, one can simply create a memory view
of the first dimension of the array:

```c++
Vector<double> firstDimensionOfC(Dim1, &C(1,i2,i3))
```

And the loop body of the mex file can be rewritten as following:
```c++
	for (int i3 = 1; i3 <= Dim3 ; i3++)
	{
		for (int i2 = 1; i2 <= Dim2 ; i2++)
		{
			Vector<double> firstDimensionOfC(Dim1, &(1,i2,i3));
			for (int i1 = 1; i1 <= Dim1 ; i1++)
			{
				firstDimensionOfC(i1) = A(i1, i2, i3)*B(i1, i2);
			}
			C(ALL, i2, i3) = A(ALL, i2, i3);
		}
	}
```
This reduces overhead of computing stride and simplifies coding effort for large arrays.
Notice it should be clear that the array is column-major.

What GET_DM() actually does is to get variable from the caller workspace,
extract the data pointer, and construct a blitz array object reference to the data.
Notice mexGetVariable() always copies data; therefore, changes to data are not posted
unless PUT(var) is called.

#### PUT(var), PUT_(var)
PUT(var) posts the changes in var to the caller workspace.

PUT_(var) post the changes to a new variable var_.

#### GET_DV(var), GET_IV(var)
GET_DV(var) gets var and treats it as a one dimension vector.

This macro is provided because matlab always treats a vector as two-dimensional.
Sometimes it's convenient to force the vector read to be single dimension.

#### Error Checking and Parallel
Compile with -DBZ_DEBUG to enable bounds checking

Compile with -DBZ_THREADSAFE when using OpenMP



