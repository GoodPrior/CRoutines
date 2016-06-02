#include "mex.h"
#include "string.h"

#define MAXNAMELENGTH 20

/// v2structMex(inputStruct)
/// Unpack fields in inputStruct to caller's workspace
/// @Author: Wenlan Luo (luowenlan@gmail.com)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs != 1 | !mxIsChar(prhs[0]))
		mexErrMsgTxt("Input error.");


	int lengthStructName = mxGetNumberOfElements(prhs[0]);
	char structName[MAXNAMELENGTH];
	mxGetString(prhs[0], structName, lengthStructName + 1);

	const mxArray* inputStruct = mexGetVariablePtr("caller", structName);
	// const mxArray* inputStruct = prhs[0];

	// Get number of fields
	int nFields = mxGetNumberOfFields(inputStruct);

	//
	int totalLength = (MAXNAMELENGTH + 1)*nFields + strlen("persistent ");
	int totalLength2 = (lengthStructName + MAXNAMELENGTH * 2 + 3)*nFields;

	// allocate command space
	char* command = (char*)malloc(sizeof(char)*totalLength);
	char* p = command;
	sprintf(p, "persistent ");
	p += strlen("persistent ");

	// For each field in the structure, put variable to the caller work space
	for (int j = 0; j < nFields ; j++)
	{
		const char* fieldName = mxGetFieldNameByNumber(inputStruct, j);
		if (!mexGetVariablePtr("caller", fieldName)) {
			sprintf(p, "%s ", fieldName);
			p += strlen(fieldName) + 1;
		}
	}

	char* command2 = (char*)malloc(sizeof(char)*totalLength2);
	p = command2;
	for (int j = 0; j < nFields; j++)
	{
		const char* fieldName = mxGetFieldNameByNumber(inputStruct, j);
		sprintf(p, "%s=%s.%s;", fieldName, structName, fieldName);
		p += lengthStructName + 2 * strlen(fieldName) + 3;
	}

	// mexPrintf("%s", command);
	plhs[0] = mxCreateString(command);
	plhs[1] = mxCreateString(command2);
	free(command);
	free(command2);

	/*
	for (int j = 0; j < nFields ; j++)
	{
		const char* fieldName = mxGetFieldNameByNumber(inputStruct, j);
		const mxArray* field = mxGetFieldByNumber(inputStruct, 0, j);
		mexPutVariable("caller", fieldName, field);
	}
	*/
}
