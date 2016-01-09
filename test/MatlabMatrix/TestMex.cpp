#include "mex.h"
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
