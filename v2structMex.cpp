
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs != 1 & mxIsStruct(prhs[0]))
		mexErrMsgTxt("Input error.");

	const mxArray* inputStruct = prhs[0];


	// Get number of fields
	int nFileds = mxGetNumberOfFields(inputStruct);
	// For each field in the structure, put variable to the caller work space
	for (int j = 0; j < nFileds ; j++)
	{
		const char* fieldName = mxGetFieldNameByNumber(inputStruct, j);
		const mxArray* field = mxGetFieldByNumber(inputStruct, 0, j);
		mexPutVariable("caller", fieldName, field);
	}
}
