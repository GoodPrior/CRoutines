
/**
 * @file myppualMKL_CMEX.c
 * @author Jinhui Bai jinhui.bai@gmail.com
 * @author Wenlan Luo luowenlan@gmail.com
 *
 * This is the cmex source file calling intel MKL's spline construction and evaluation, which is part of general spline evaluation routine myppual in matlab. Please call through myppual gateway in matlab and see a detailed documentation shipped with myppual.
 *
 * @date Dec 21, 2013
 *
 * @version 0.1.0
 *
 * At this stage this routine supports:
 * 1. vector-function multi-dimension spline construction, allow for different order at each dimension.
 * 2. multi-dimension spline evaluation at specified vector function dimension of each site.
 * 3. partial construction and evaluation at specified vector function dimension (extension of GriddedInterpolant's cell-form), but only supports upper dimension be scalar, and vector-site be 1 dimension.
 * 4. if has input index, will use it as initial guess and use hunt() to search. Can output search result.
 */

/**
 * Keep a patch list here:
 *
 * 6/5/2014 0.1.1:
 * 1. Fix a bug for mixed evaluation (evaluation with order different at each coordinate).
 *
 * 12/21/2013 0.1.0:
 * 1. reorganize the code for fewer switches and better performance, (more maintanance cost though)
 * 2. now can take input index as initial guess and use hunt() to search.
 * 3. now can output search index.
 *
 * predated 0.0.0:
 *
 * 12/17/2013 0.0.0:
 * 1. vector-function multi-dimension spline construction, allow for different order at each dimension.
 * 2. multi-dimension spline evaluation at specified vector function dimension of each site.
 * 3. partial construction and evaluation at specified vector function dimension (extension of GriddedInterpolant's cell-form), but only supports upper dimension be scalar, and vector-site be 1 dimension.
 *
 */

#include "mkl.h"
#include "mex.h"
#include <string.h>
#include <omp.h>

#define MAXDIM 10

#define MAX(a,b) ((a>b) ? a : b)
#define MIN(a,b) ((a<b) ? a : b)

// get arguments from matlab rhs arguments
#define flag_           prhs[0]
#define breaks_         prhs[1]
#define values_         prhs[2]
#define pieces_         prhs[3]
#define order_          prhs[4]
#define dim_            prhs[5]
#define method_         prhs[6]
#define x_              prhs[7]
#define left_           prhs[8]
#define indexDim_       prhs[9]
#define index_x_        prhs[10]

// get arguments from matlab lhs
#define result_         plhs[0]
#define outputIndex_    plhs[1]

// get options from MKL_flag
#define GET_OPTION(i)   (*((int*) mxGetData(flag_) + i))
#define NUM_THREADS GET_OPTION(0)

void error(char* message)
{
  mexErrMsgTxt(message);
}

void errorCheck(int condition, char* message)
{
  if (!condition)
    error(message);
}

// HEADER
// major switch
void fullConstruction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void partialConstructEval(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void eval2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void eval4(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void evalN(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void partialConstructEvalIn(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void eval2In(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void eval4In(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void evalNIn(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void partialConstructEvalOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void eval2Out(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void eval4Out(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void evalNOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void partialConstructEvalInOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void eval2InOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void eval4InOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void evalNInOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
// Nr3
void locate(double xx[], int n, double x, int *j);
void hunt(double xx[], int n, double x, int *jlo);
// NdInterp
void mkl_partial_construction_evaluation(
    int xDim, double** xGrid, int* xPts,
    double* y, int yDim,
    int s_order,
    int nsite, double* siteEval, double* siteReduction, int siteReductionDim, 
    int nvec, int* ivec,
    double* result);
void mkl_start(int xDim, int* xPts, double** xGrid, int valDim, double* y, int* s_order, double* coeff, int coeffn);
double evalNoSearch2(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns);
double recevalNoSearch2(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns);
double evalNoSearch4(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns);
double recevalNoSearch4(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns);
double recevalNoSearchN(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns, int* s_order);
// misc
void intmemcpy(int* des, int* src, int n);
void doublememcpy(double* des, double* src, int n);
int vectorProd(int* x, int n);
int indexMklToNr3(int oldidx, int xPts);

/**
   Major routine with cmex interface.

   One should call in matlab in the form of
 */


/**

   <b> [v, outputIndex] = myppualMKL_CMEX(flag, breaks,value,pieces,order,dim,method,x,left,indexDim,index_x) </b>
 */


/**
 *
   - There's some light error check. Compile with -DNOCHECK to disable it (it's disabled by default with the shipped makefile).

   - All double passed in should be in the type of double (matlab default).

   - All integer passed in should be in the type of int32.

   @param[in] flag Some options.
   - Absolute value gives number of threads for openmp.
   - Positive for evaluation or negative for construction.
   - Pass in the type of int32.
   @param[in] breaks breaks in ppform. Grid points of each dimension.
   @param[in] value
   - If flag is negative, this is treated as value.
   - If flag is positive and x is cell, this is treated as value. Make sure it's C-sytle (row-major) when passed as value.
   - If flag is positive and x is matrix, this is treated as coefficients.
   @param[in] pieces pieces in ppform. Number of pieces of each dimension of coefficients. Currently not used, routine will read relevant information from breaks.
   @param[in] order order in ppform. Order of interpolation, 2 for linear and 4 for cubic. Needs to be passed as int32.
   - Construction and evaluation supports any combination of 2 or 4 for different dimensions.
   - Partial construction and evaluation supports only 4.
   - Notes: there's performance gain if orders are all 2 or 4 (compared with mixed orders at different dimensions). A specific routine with unwinding loop boosts the performance in either case.
   @param[in] dim dim in ppform. Number of vector functions. Pass in the type of int32.
   @param[in] method method in ppform, a placeholder. Currently, only the default method "not-a-knot" is supported.
   @param[in] x Interpolation sites. Will use only in evaluation switch (flag be positive).
   - If it's passed as cell, then will do partial construction and evaluation, currently only higher dimension be scalar, first dimension be vector site is supported.
   - If it's passed as matrix, then will do evaluation; make sure memory goes along dimension first, then vector site. (i.e. in matlab, the matrix should has the size [xDim, nsite])
   @param[in] left Placeholder.
   @param[in] indexDim At which vector function to be interpolated for each site. Needs to be passed as int32 with size [nvec, nsite]. Index should be adjusted to start from 0.
   @param[in] index_x Cell index of sites. If it's not empty, will use it as initial guess and use hunt() to search.
   - Pass in as int32.
   - If x is passed as cell, then index_x should be passed as cell. Now only the first element of the cell is used. (only the vector site uses hunt() to search, not the scalar site)
   - If x is passed as matrix, then index_x should be passed as matrix with the same size as x.
   @param[out] v
   - If flag is negative, return coefficients.
   - If flag is positive, return interplated values.
   @param[out] outputIndex Search results.
   - Can only be returned in evaluation switch (flag be positive).
   - Return format follows the same rule as index_x.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int numThreads = NUM_THREADS;
  if (numThreads > 0) {
    // deal with input and output index
    int hasInputIndex = 0;
    int hasOutputIndex = 0;
    if (nlhs > 1) {
      hasOutputIndex = 1;
    }
    if (mxGetNumberOfElements(index_x_)) {
      hasInputIndex = 1;
    }

    if (mxIsCell(x_)) {
      // cell-form
      if (hasInputIndex && !hasOutputIndex) {
        partialConstructEvalIn(nlhs, plhs, nrhs, prhs);
      }
      else if (!hasInputIndex && hasOutputIndex) {
        partialConstructEvalOut(nlhs, plhs, nrhs, prhs);
      }
      else if (hasInputIndex && hasOutputIndex) {
        partialConstructEvalInOut(nlhs, plhs, nrhs, prhs);
      }
      else {
        partialConstructEval(nlhs, plhs, nrhs, prhs);
      }
      return;
    } // mxIsCell
    else if (mxIsDouble(x_)) {
      // matrix-form
      int xDim = mxGetNumberOfElements(breaks_);
      int* s_order = (int*) mxGetData(order_);
      // check is s_order all equal to 4 or 2
      // is all cubic?
      int isEqualTo4 = 1;
      for (int i = 0; i < xDim; ++i) {
        if (s_order[i] != 4) {
          isEqualTo4 = 0;
          break;
        }
      } // i
      if (isEqualTo4) {
        if (hasInputIndex && !hasOutputIndex) {
          eval4In(nlhs, plhs, nrhs, prhs);
        }
        else if (!hasInputIndex && hasOutputIndex) {
          eval4Out(nlhs, plhs, nrhs, prhs);
        }
        else if (hasInputIndex && hasOutputIndex) {
          eval4InOut(nlhs, plhs, nrhs, prhs);
        }
        else {
          eval4(nlhs, plhs, nrhs, prhs);
        }
        return;
      } // isEqualTo4
      // is all linear?
      int isEqualTo2 = 1;
      for (int i = 0; i < xDim; ++i) {
        if (s_order[i] != 2) {
          isEqualTo2 = 0;
          break;
        }
      }
      if (isEqualTo2) {
        if (hasInputIndex && !hasOutputIndex) {
          eval2In(nlhs, plhs, nrhs, prhs);
        }
        else if (!hasInputIndex && hasOutputIndex) {
          eval2Out(nlhs, plhs, nrhs, prhs);
        }
        else if (hasInputIndex && hasOutputIndex) {
          eval2InOut(nlhs, plhs, nrhs, prhs);
        }
        else {
          eval2(nlhs, plhs, nrhs, prhs);
        }
        return;
      } // isEqualTo2
      // else, combined
      {
        if (hasInputIndex && !hasOutputIndex) {
          evalNIn(nlhs, plhs, nrhs, prhs);
        }
        else if (!hasInputIndex && hasOutputIndex) {
          evalNOut(nlhs, plhs, nrhs, prhs);
        }
        else if (hasInputIndex && hasOutputIndex) {
          evalNInOut(nlhs, plhs, nrhs, prhs);
        }
        else {
          evalN(nlhs, plhs, nrhs, prhs);
        }
      } // combined
      return;
    } // mxIsDouble
  } // numThreads > 0
  else if (numThreads < 0) {
    fullConstruction(nlhs, plhs, nrhs, prhs);
    return;
  } // numThreads < 0
  else {
    error("numThreads == 0, is a placeholder");
  } // numThreads == 0
}

void fullConstruction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // extract vector function from values
  double* y = mxGetPr(values_);

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
  int yDim = dim[0];

  // deal with x

  // deal with left

  // deal with indexDim

  // deal with index_x

  // allocate result, now the result is coeffs
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // fortran order
  coeffns[xDim] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[xDim-1-i] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[xDim-1-i];
  }
  result_ = mxCreateNumericArray(xDim+1, coeffns, mxDOUBLE_CLASS, mxREAL);
  double* coeff = mxGetPr(result_);

  // set num of threads
  mkl_set_num_threads(-NUM_THREADS);
  omp_set_num_threads(-NUM_THREADS);

  // call major routine
  mkl_start(xDim, xPts, xGrid, yDim, y, s_order, coeff, coeffn);
}

void eval2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (xDim == 1) {
    // has better routine for one dim case
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite;
        double xsite;
        locate(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = evalNoSearch2(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    }// nvec > 0
    else {
      // use MKL's evaluation
      DFTaskPtr interp;
      dfdNewTask1D(&interp, xPts[0], xGrid[0], DF_NO_HINT, yDim, 0, DF_NO_HINT);
      MKL_INT s_type;
      MKL_INT bc_type;
      switch (s_order[0]) {
      case 4:
        s_type = DF_PP_NATURAL;
        bc_type = DF_BC_NOT_A_KNOT;
        break;
      case 2:
        s_type = DF_PP_DEFAULT;
        bc_type = DF_NO_BC;
        break;
      }
      dfdEditPPSpline1D(interp, s_order[0], s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
      int dorder = 1;
      dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, nsite, site, DF_NO_HINT, 1, &dorder, NULL, result, DF_NO_HINT, 0);
      dfDeleteTask(&interp);
    }
  } // xDim == 1
  else {
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = recevalNoSearch2(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
      // full vector function
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = recevalNoSearch2(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim > 1
}

void eval2In(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x
#ifndef NOCHECK
  errorCheck(mxIsInt32(index_x_), "in matrix form: index_x should be int32 matrix");
  errorCheck(mxGetM(index_x_) == xDim, "in matrix form: row of index_x doesn't agree with xDim");
  errorCheck(mxGetN(index_x_) == nsite, "in matrix form: column of index_x doesn't agree with nsite");
#endif
  int* inputIndex = (int*) mxGetData(index_x_);

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (xDim == 1) {
    // has better routine for one dim case
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite = inputIndex[i];
        hunt(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = evalNoSearch2(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite = inputIndex[i];
        hunt(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = evalNoSearch2(0, &xsite, &cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim == 1
  else {
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          cellofsite[j] = inputIndex[i*xDim + j];
          hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = recevalNoSearch2(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
      // full vector function
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          cellofsite[j] = inputIndex[i*xDim + j];
          hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = recevalNoSearch2(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim > 1
}

void eval2Out(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // deal with outputIndex, fortran order
  outputIndex_ = mxCreateNumericArray(2, (int[]){xDim, nsite}, mxINT32_CLASS, mxREAL);
  int* outputIndex = (int*) mxGetData(outputIndex_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (xDim == 1) {
    // has better routine for one dim case
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite;
        locate(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // store search result
        outputIndex[i] = cellofsite;
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = evalNoSearch2(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite;
        locate(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // store search result
        outputIndex[i] = cellofsite;
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = evalNoSearch2(0, &xsite, &cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim == 1
  else {
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // store search result
          outputIndex[i*xDim + j] = cellofsite[j];
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = recevalNoSearch2(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
      // full vector function
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // store search result
          outputIndex[i*xDim + j] = cellofsite[j];
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = recevalNoSearch2(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim > 1
}

void eval2InOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x
#ifndef NOCHECK
  errorCheck(mxIsInt32(index_x_), "in matrix form: index_x should be int32 matrix");
  errorCheck(mxGetM(index_x_) == xDim, "in matrix form: row of index_x doesn't agree with xDim");
  errorCheck(mxGetN(index_x_) == nsite, "in matrix form: column of index_x doesn't agree with nsite");
#endif
  int* inputIndex = (int*) mxGetData(index_x_);

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // deal with outputIndex, fortran order
  outputIndex_ = mxCreateNumericArray(2, (int[]){xDim, nsite}, mxINT32_CLASS, mxREAL);
  int* outputIndex = (int*) mxGetData(outputIndex_);
  intmemcpy(outputIndex, inputIndex, xDim*nsite);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (xDim == 1) {
    // has better routine for one dim case
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite = outputIndex[i];
        hunt(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // store search result
        outputIndex[i] = cellofsite;
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = evalNoSearch2(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite = outputIndex[i];
        hunt(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // store search result
        outputIndex[i] = cellofsite;
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = evalNoSearch2(0, &xsite, &cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim == 1
  else {
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int* cellofsite = outputIndex + i*xDim;
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = recevalNoSearch2(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
      // full vector function
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int* cellofsite = outputIndex + i*xDim;
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = recevalNoSearch2(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim > 1
}

void eval4(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (xDim == 1) {
    // has better routine for one dim case
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite;
        locate(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    }// nvec > 0
    else {
      // use MKL's evaluation
      DFTaskPtr interp;
      double tempy;
      dfdNewTask1D(&interp, xPts[0], xGrid[0], DF_NO_HINT, yDim, &tempy, DF_NO_HINT);
      MKL_INT s_type;
      MKL_INT bc_type;
      switch (s_order[0]) {
      case 4:
        s_type = DF_PP_NATURAL;
        bc_type = DF_BC_NOT_A_KNOT;
        break;
      case 2:
        s_type = DF_PP_DEFAULT;
        bc_type = DF_NO_BC;
        break;
      }
      dfdEditPPSpline1D(interp, s_order[0], s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
      int dorder = 1;
      double* resultTemp = (double*) malloc (sizeof(double) * nsite * yDim);
      dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, nsite, site, DF_NO_HINT, 1, &dorder, NULL, resultTemp, DF_NO_HINT, 0);
      mkl_domatcopy('R', 'T', yDim, nsite, 1.0, resultTemp, nsite, result, yDim);
      free(resultTemp);
      dfDeleteTask(&interp);
    }
  } // xDim == 1
  else {
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = recevalNoSearch4(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
      // full vector function
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = recevalNoSearch4(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim > 1
}

void eval4In(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x
#ifndef NOCHECK
  errorCheck(mxIsInt32(index_x_), "in matrix form: index_x should be int32 matrix");
  errorCheck(mxGetM(index_x_) == xDim, "in matrix form: row of index_x doesn't agree with xDim");
  errorCheck(mxGetN(index_x_) == nsite, "in matrix form: column of index_x doesn't agree with nsite");
#endif
  int* inputIndex = (int*) mxGetData(index_x_);

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (xDim == 1) {
    // has better routine for one dim case
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite = inputIndex[i];
        hunt(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite = inputIndex[i];
        hunt(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = evalNoSearch4(0, &xsite, &cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim == 1
  else {
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          cellofsite[j] = inputIndex[i*xDim + j];
          hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = recevalNoSearch4(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
      // full vector function
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          cellofsite[j] = inputIndex[i*xDim + j];
          hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = recevalNoSearch4(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim > 1
}

void eval4Out(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // deal with outputIndex, fortran order
  outputIndex_ = mxCreateNumericArray(2, (int[]){xDim, nsite}, mxINT32_CLASS, mxREAL);
  int* outputIndex = (int*) mxGetData(outputIndex_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (xDim == 1) {
    // has better routine for one dim case
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite;
        locate(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // store search result
        outputIndex[i] = cellofsite;
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite;
        locate(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // store search result
        outputIndex[i] = cellofsite;
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = evalNoSearch4(0, &xsite, &cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim == 1
  else {
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // store search result
          outputIndex[i*xDim + j] = cellofsite[j];
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = recevalNoSearch4(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
      // full vector function
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite[MAXDIM];
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // store search result
          outputIndex[i*xDim + j] = cellofsite[j];
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = recevalNoSearch4(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim > 1
}

void eval4InOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x
#ifndef NOCHECK
  errorCheck(mxIsInt32(index_x_), "in matrix form: index_x should be int32 matrix");
  errorCheck(mxGetM(index_x_) == xDim, "in matrix form: row of index_x doesn't agree with xDim");
  errorCheck(mxGetN(index_x_) == nsite, "in matrix form: column of index_x doesn't agree with nsite");
#endif
  int* inputIndex = (int*) mxGetData(index_x_);

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // deal with outputIndex, fortran order
  outputIndex_ = mxCreateNumericArray(2, (int[]){xDim, nsite}, mxINT32_CLASS, mxREAL);
  int* outputIndex = (int*) mxGetData(outputIndex_);
  intmemcpy(outputIndex, inputIndex, xDim*nsite);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (xDim == 1) {
    // has better routine for one dim case
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite = outputIndex[i];
        hunt(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // store search result
        outputIndex[i] = cellofsite;
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int cellofsite = outputIndex[i];
        hunt(xGrid[0], xPts[0] - 2, site[i], &cellofsite);
        // x minus xleft
        double xsite = site[i] - xGrid[0][cellofsite];
        // store search result
        outputIndex[i] = cellofsite;
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = evalNoSearch4(0, &xsite, &cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim == 1
  else {
    if (nvec > 0) {
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int* cellofsite = outputIndex + i*xDim;
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < nvec; ++j) {
          result[i*nvec + j] = recevalNoSearch4(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec > 0
    else {
      // full vector function
#pragma omp parallel for private(i, j)
      for (i = 0; i < nsite; ++i) {
        // search once
        int* cellofsite = outputIndex + i*xDim;
        double xsite[MAXDIM];
        for (j = 0; j < xDim; ++j) {
          locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
          // x minus xleft
          xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
        } // j
        // evaluation
        for (j = 0; j < yDim; ++j) {
          result[i*yDim + j] = recevalNoSearch4(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns);
        } // j
      } // i
    } // nvec == 0
  } // xDim > 1
}

void evalN(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      // search once
      int cellofsite[MAXDIM];
      double xsite[MAXDIM];
      for (j = 0; j < xDim; ++j) {
        locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
        // x minus xleft
        xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
      } // j
      // evaluation
      for (j = 0; j < nvec; ++j) {
        result[i*nvec + j] = recevalNoSearchN(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns, s_order);
      } // j
    } // i
  } // nvec > 0
  else {
    // full vector function
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      // search once
      int cellofsite[MAXDIM];
      double xsite[MAXDIM];
      for (j = 0; j < xDim; ++j) {
        locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
        // x minus xleft
        xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
      } // j
      // evaluation
      for (j = 0; j < yDim; ++j) {
        result[i*yDim + j] = recevalNoSearchN(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns, s_order);
      } // j
    } // i
  } // nvec == 0
}

void evalNIn(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x
#ifndef NOCHECK
  errorCheck(mxIsInt32(index_x_), "in matrix form: index_x should be int32 matrix");
  errorCheck(mxGetM(index_x_) == xDim, "in matrix form: row of index_x doesn't agree with xDim");
  errorCheck(mxGetN(index_x_) == nsite, "in matrix form: column of index_x doesn't agree with nsite");
#endif
  int* inputIndex = (int*) mxGetData(index_x_);

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      // search once
      int cellofsite[MAXDIM];
      double xsite[MAXDIM];
      for (j = 0; j < xDim; ++j) {
        cellofsite[j] = inputIndex[i*xDim + j];
        hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
        // x minus xleft
        xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
      } // j
      // evaluation
      for (j = 0; j < nvec; ++j) {
        result[i*nvec + j] = recevalNoSearchN(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns, s_order);
      } // j
    } // i
  } // nvec > 0
  else {
    // full vector function
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      // search once
      int cellofsite[MAXDIM];
      double xsite[MAXDIM];
      for (j = 0; j < xDim; ++j) {
        cellofsite[j] = inputIndex[i*xDim + j];
        hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
        // x minus xleft
        xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
      } // j
      // evaluation
      for (j = 0; j < yDim; ++j) {
        result[i*yDim + j] = recevalNoSearchN(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns, s_order);
      } // j
    } // i
  } // nvec == 0
}

void evalNOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // deal with outputIndex, fortran order
  outputIndex_ = mxCreateNumericArray(2, (int[]){xDim, nsite}, mxINT32_CLASS, mxREAL);
  int* outputIndex = (int*) mxGetData(outputIndex_);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      // search once
      int cellofsite[MAXDIM];
      double xsite[MAXDIM];
      for (j = 0; j < xDim; ++j) {
        locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
        // store search result
        outputIndex[i*xDim + j] = cellofsite[j];
        // x minus xleft
        xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
      } // j
      // evaluation
      for (j = 0; j < nvec; ++j) {
        result[i*nvec + j] = recevalNoSearchN(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns, s_order);
      } // j
    } // i
  } // nvec > 0
  else {
    // full vector function
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      // search once
      int cellofsite[MAXDIM];
      double xsite[MAXDIM];
      for (j = 0; j < xDim; ++j) {
        locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
        // store search result
        outputIndex[i*xDim + j] = cellofsite[j];
        // x minus xleft
        xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
      } // j
      // evaluation
      for (j = 0; j < yDim; ++j) {
        result[i*yDim + j] = recevalNoSearchN(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns, s_order);
      } // j
    } // i
  } // nvec == 0
}

void evalNInOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with y

  // deal with pieces

  // deal with order
  int* s_order = (int*) mxGetData(order_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(order_), "in evaluation: order should be int32");
  errorCheck(mxGetNumberOfElements(order_) == xDim, "in evaluation: size of order should agree with size of breaks");
#endif

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
#ifndef NOCHECK
  errorCheck(mxIsInt32(dim_), "in evaluation: dim should be int32");
#endif
  int yDim = dim[0];

  // deal with x
  double* site = mxGetPr(x_);
  int nsite = mxGetN(x_);
#ifndef NOCHECK
  int siteDim = mxGetM(x_);
  errorCheck(siteDim == xDim, "in evaluation: xDim doesn't agree with siteDim");
#endif

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);
#ifndef NOCHECK
  if (nvec > 0) {
    errorCheck(mxIsInt32(indexDim_), "in evaluation: indexDim should be int32");
  }
#endif

  // deal with index_x
#ifndef NOCHECK
  errorCheck(mxIsInt32(index_x_), "in matrix form: index_x should be int32 matrix");
  errorCheck(mxGetM(index_x_) == xDim, "in matrix form: row of index_x doesn't agree with xDim");
  errorCheck(mxGetN(index_x_) == nsite, "in matrix form: column of index_x doesn't agree with nsite");
#endif
  int* inputIndex = (int*) mxGetData(index_x_);

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // deal with outputIndex, fortran order
  outputIndex_ = mxCreateNumericArray(2, (int[]){xDim, nsite}, mxINT32_CLASS, mxREAL);
  int* outputIndex = (int*) mxGetData(outputIndex_);
  intmemcpy(outputIndex, inputIndex, xDim*nsite);

  // determine coeff dimension
  int coeffn = yDim;
  int coeffns[MAXDIM];
  // C order
  coeffns[0] = yDim;
  for (int i = 0; i < xDim; ++i) {
    coeffns[i+1] = (xPts[i] - 1) * s_order[i];
    coeffn *= coeffns[i+1];
  }
  // values_ stores coeffs
  double* coeff = mxGetPr(values_);
#ifndef NOCHECK
  errorCheck(coeffn==mxGetNumberOfElements(values_), "in evaluation, coeffs has wrong numel");
#endif

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // evaluate
  int i, j;
  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      // search once
      int* cellofsite = outputIndex + i*xDim;
      double xsite[MAXDIM];
      for (j = 0; j < xDim; ++j) {
        hunt(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
        // x minus xleft
        xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
      } // j
      // evaluation
      for (j = 0; j < nvec; ++j) {
        result[i*nvec + j] = recevalNoSearchN(0, xsite, cellofsite, ivec[i*nvec + j] * coeffns[1], xDim, coeff, coeffns, s_order);
      } // j
    } // i
  } // nvec > 0
  else {
    // full vector function
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      // search once
      int* cellofsite = outputIndex + i*xDim;
      double xsite[MAXDIM];
      for (j = 0; j < xDim; ++j) {
        locate(xGrid[j], xPts[j] - 2, site[i*xDim + j], cellofsite + j);
        // x minus xleft
        xsite[j] = site[i*xDim + j] - xGrid[j][cellofsite[j]];
      } // j
      // evaluation
      for (j = 0; j < yDim; ++j) {
        result[i*yDim + j] = recevalNoSearchN(0, xsite, cellofsite, j * coeffns[1], xDim, coeff, coeffns, s_order);
      } // j
    } // i
  } // nvec == 0
}

void partialConstructEval(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with coefs
  // extract vector function from coefs
  double* y = mxGetPr(values_);

  // deal with pieces

  // deal with order
  int* order = (int*) mxGetData(order_);
  int s_order = order[0];

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
  int yDim = dim[0];

  // deal with x
  int siteDim = mxGetNumberOfElements(x_);
  int siteReductionDim = siteDim - 1;
  /*
#ifndef NOCHECK
  errorCheck(siteReductionDim > 0, "in partialConstructionEvaluation: siteReductionDim should > 0");
#endif
*/
  double* siteEval;
  double siteReduction[MAXDIM];
  // the first dimension is site vector
  mxArray* siteElement = mxGetCell(x_, 0);
  siteEval = mxGetPr(siteElement);
  int nsite = mxGetNumberOfElements(siteElement);
  // the last 2:end dimension would be scalar
  for (int i = 1; i < siteDim; ++i) {
    mxArray* siteElement = mxGetCell(x_, i);
    double* site_reduction_ptr = mxGetPr(siteElement);
    siteReduction[i-1] = *site_reduction_ptr;
  }

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);

  // deal with index_x

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

  // call major routine
  // iterator
  int i,j;

  // iteratively construct coeffs and evaluate using intel-mkl, from the highest dimension
  DFTaskPtr interp;
  MKL_INT s_type;
  MKL_INT bc_type;
  switch (s_order) {
  case 4:
    s_type = DF_PP_NATURAL;
    bc_type = DF_BC_NOT_A_KNOT;
    break;
  case 2:
    s_type = DF_PP_DEFAULT;
    bc_type = DF_NO_BC;
    break;
  }

  double* lastInterpResult;
  double* coeff;
  if (siteReductionDim > 0) {
    // construct the num of vector functions for each construction
    int sudoVecNum[MAXDIM];
    sudoVecNum[0] = yDim;
    for (i = 1; i < xDim; ++i) {
      sudoVecNum[i] = sudoVecNum[i-1] * xPts[i-1];
    }
    // do it once at the last scalar dimension
    int cDim = xDim - 1;
    lastInterpResult = y;
    dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
    coeff = (double*) malloc((xPts[cDim]-1) * sudoVecNum[cDim] * s_order * sizeof(double));
    dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
    dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
    int dorder = 1;
    double* interpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, NULL);
    dfDeleteTask(&interp);
    lastInterpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);

    // do the remaining scalar dimension
    for (i = 1; i < siteReductionDim; ++i) {
      int cDim = xDim - 1 - i;
      dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
      dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
      dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
      dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, NULL);
      dfDeleteTask(&interp);
      memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);
    }
    free(interpResult);
  } // siteReductionDim > 0
  else {
    lastInterpResult = (double*) malloc(sizeof(double) * yDim * xPts[0]);
    doublememcpy(lastInterpResult, y, yDim * xPts[0]);
    int coeffn = yDim * (xPts[0] - 1) * 4;
    coeff = (double*) malloc(sizeof(double) * coeffn);
  } // siteReductionDim == 0

  // do the vector site dimension
  dfdNewTask1D(&interp, xPts[0], xGrid[0], DF_NO_HINT, yDim, lastInterpResult, DF_NO_HINT);
  dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
  dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);

  int coeffns[MAXDIM];
  coeffns[0] = yDim;
  coeffns[1] = (xPts[0] - 1) * 4;

  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      int cellofsite;
      // search once
      locate(xGrid[0], xPts[0] - 2, siteEval[i], &cellofsite);
      double xsite = siteEval[i] - xGrid[0][cellofsite];
      for (j = 0; j < nvec; ++j) {
        // double recevalNoSearch4(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns)
        result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], 1, coeff, coeffns);
      }
    }
  } else {
    // full vector dimension
    double* resultTemp = (double*) malloc (sizeof(double) * nsite * yDim);
    int dorder = 1;
    dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, nsite, siteEval, DF_NO_HINT, 1, &dorder, NULL, resultTemp, DF_NO_HINT, NULL);
    mkl_domatcopy('R', 'T', yDim, nsite, 1.0, resultTemp, nsite, result, yDim);
    free(resultTemp);
  }

  // delete everything
  dfDeleteTask(&interp);
  free(coeff);
  free(lastInterpResult);
}

void partialConstructEvalIn(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with coefs
  // extract vector function from coefs
  double* y = mxGetPr(values_);

  // deal with pieces

  // deal with order
  int* order = (int*) mxGetData(order_);
  int s_order = order[0];

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
  int yDim = dim[0];

  // deal with x
  int siteDim = mxGetNumberOfElements(x_);
  int siteReductionDim = siteDim - 1;
  /*
#ifndef NOCHECK
  errorCheck(siteReductionDim > 0, "in partialConstructionEvaluation: siteReductionDim should > 0");
#endif
*/
  double* siteEval;
  double siteReduction[MAXDIM];
  // the first dimension is site vector
  mxArray* siteElement = mxGetCell(x_, 0);
  siteEval = mxGetPr(siteElement);
  int nsite = mxGetNumberOfElements(siteElement);
  // the last 2:end dimension would be scalar
  for (int i = 1; i < siteDim; ++i) {
    mxArray* siteElement = mxGetCell(x_, i);
    double* site_reduction_ptr = mxGetPr(siteElement);
    siteReduction[i-1] = *site_reduction_ptr;
  }

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);

  // deal with index_x
#ifndef NOCHECK
  errorCheck(mxIsCell(index_x_), "in cell form: index_x should be cell");
#endif
  mxArray* inputIndexMx = mxGetCell(index_x_, 0);
#ifndef NOCHECK
  errorCheck(mxIsInt32(inputIndexMx), "in cell form: each element of index_x should be int32 array");
  errorCheck(mxGetNumberOfElements(inputIndexMx) == nsite, "in cell form, dimension of element of index_x should agree with nsite");
#endif
  int* inputIndex = (int*) mxGetData(inputIndexMx);

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

    // iterator
  int i,j;

  // iteratively construct coeffs and evaluate using intel-mkl, from the highest dimension
  DFTaskPtr interp;
  MKL_INT s_type;
  MKL_INT bc_type;
  switch (s_order) {
  case 4:
    s_type = DF_PP_NATURAL;
    bc_type = DF_BC_NOT_A_KNOT;
    break;
  case 2:
    s_type = DF_PP_DEFAULT;
    bc_type = DF_NO_BC;
    break;
  }

  double* lastInterpResult;
  double* coeff;
  if (siteReductionDim > 0) {
    // construct the num of vector functions for each construction
    int sudoVecNum[MAXDIM];
    sudoVecNum[0] = yDim;
    for (i = 1; i < xDim; ++i) {
      sudoVecNum[i] = sudoVecNum[i-1] * xPts[i-1];
    }
    // do it once at the last scalar dimension
    int cDim = xDim - 1;
    lastInterpResult = y;
    dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
    coeff = (double*) malloc((xPts[cDim]-1) * sudoVecNum[cDim] * s_order * sizeof(double));
    dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
    dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
    int dorder = 1;
    double* interpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, NULL);
    dfDeleteTask(&interp);
    lastInterpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);

    // do the remaining scalar dimension
    for (i = 1; i < siteReductionDim; ++i) {
      int cDim = xDim - 1 - i;
      dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
      dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
      dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
      dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, NULL);
      dfDeleteTask(&interp);
      memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);
    }
    free(interpResult);
  } // siteReductionDim > 0
  else {
    lastInterpResult = (double*) malloc(sizeof(double) * yDim * xPts[0]);
    doublememcpy(lastInterpResult, y, yDim * xPts[0]);
    int coeffn = yDim * (xPts[0] - 1) * 4;
    coeff = (double*) malloc(sizeof(double) * coeffn);
  } // siteReductionDim == 0

  // do the vector site dimension
  dfdNewTask1D(&interp, xPts[0], xGrid[0], DF_NO_HINT, yDim, lastInterpResult, DF_NO_HINT);
  dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
  dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);

  int coeffns[MAXDIM];
  coeffns[0] = yDim;
  coeffns[1] = (xPts[0] - 1) * 4;

  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      int cellofsite = inputIndex[i];
      // search once
      hunt(xGrid[0], xPts[0] - 2, siteEval[i], &cellofsite);
      double xsite = siteEval[i] - xGrid[0][cellofsite];
      for (j = 0; j < nvec; ++j) {
        // double recevalNoSearch4(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns)
        result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], 1, coeff, coeffns);
      }
    }
  } else {
    // full vector dimension
    double* resultTemp = (double*) malloc (sizeof(double) * nsite * yDim);
    int dorder = 1;
    dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, nsite, siteEval, DF_NO_HINT, 1, &dorder, NULL, resultTemp, DF_NO_HINT, NULL);
    mkl_domatcopy('R', 'T', yDim, nsite, 1.0, resultTemp, nsite, result, yDim);
    free(resultTemp);
  }

  // delete everything
  dfDeleteTask(&interp);
  free(coeff);
  free(lastInterpResult);
}

void partialConstructEvalOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with coefs
  // extract vector function from coefs
  double* y = mxGetPr(values_);

  // deal with pieces

  // deal with order
  int* order = (int*) mxGetData(order_);
  int s_order = order[0];

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
  int yDim = dim[0];

  // deal with x
  int siteDim = mxGetNumberOfElements(x_);
  int siteReductionDim = siteDim - 1;
  /*
#ifndef NOCHECK
  errorCheck(siteReductionDim > 0, "in partialConstructionEvaluation: siteReductionDim should > 0");
#endif
*/
  double* siteEval;
  double siteReduction[MAXDIM];
  // the first dimension is site vector
  mxArray* siteElement = mxGetCell(x_, 0);
  siteEval = mxGetPr(siteElement);
  int nsite = mxGetNumberOfElements(siteElement);
  // the last 2:end dimension would be scalar
  for (int i = 1; i < siteDim; ++i) {
    mxArray* siteElement = mxGetCell(x_, i);
    double* site_reduction_ptr = mxGetPr(siteElement);
    siteReduction[i-1] = *site_reduction_ptr;
  }

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);

  // deal with index_x

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // deal with outputIndex, cell form
  outputIndex_ = mxCreateCellMatrix(xDim, 1);
  // vector site
  mxSetCell(outputIndex_, 0, mxCreateNumericArray(2, (int[]){nsite,1}, mxINT32_CLASS, mxREAL));
  // scalar site
  int* outputReductionIndex[MAXDIM];
  for (int i = 0; i < siteReductionDim; ++i) {
    mxSetCell(outputIndex_, i+1, mxCreateNumericArray(2, (int[]){1,1}, mxINT32_CLASS, mxREAL));
    outputReductionIndex[i] = (int*) mxGetData(mxGetCell(outputIndex_, i+1));
  }
  int* outputIndex = (int*) mxGetData(mxGetCell(outputIndex_, 0));

  // determine coeff dimension

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

    // iterator
  int i,j;

  // iteratively construct coeffs and evaluate using intel-mkl, from the highest dimension
  DFTaskPtr interp;
  MKL_INT s_type;
  MKL_INT bc_type;
  switch (s_order) {
  case 4:
    s_type = DF_PP_NATURAL;
    bc_type = DF_BC_NOT_A_KNOT;
    break;
  case 2:
    s_type = DF_PP_DEFAULT;
    bc_type = DF_NO_BC;
    break;
  }

  double* lastInterpResult;
  double* coeff;
  if (siteReductionDim > 0) {
    // construct the num of vector functions for each construction
    int sudoVecNum[MAXDIM];
    sudoVecNum[0] = yDim;
    for (i = 1; i < xDim; ++i) {
      sudoVecNum[i] = sudoVecNum[i-1] * xPts[i-1];
    }
    // do it once at the last scalar dimension
    int cDim = xDim - 1;
    lastInterpResult = y;
    dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
    coeff = (double*) malloc((xPts[cDim]-1) * sudoVecNum[cDim] * s_order * sizeof(double));
    dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
    dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
    int dorder = 1;
    double* interpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    dfdInterpolate1D(interp, DF_INTERP | DF_CELL, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, outputReductionIndex[cDim-1]);
    *outputReductionIndex[cDim-1] = indexMklToNr3(*outputReductionIndex[cDim-1], xPts[cDim]);
    dfDeleteTask(&interp);
    lastInterpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);

    // do the remaining scalar dimension
    for (i = 1; i < siteReductionDim; ++i) {
      int cDim = xDim - 1 - i;
      dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
      dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
      dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
      dfdInterpolate1D(interp, DF_INTERP | DF_CELL, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, outputReductionIndex[cDim-1]);
      *outputReductionIndex[cDim-1] = indexMklToNr3(*outputReductionIndex[cDim-1], xPts[cDim]);
      dfDeleteTask(&interp);
      memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);
    }
    free(interpResult);
  } // siteReductionDim > 0
  else {
    lastInterpResult = (double*) malloc(sizeof(double) * yDim * xPts[0]);
    doublememcpy(lastInterpResult, y, yDim * xPts[0]);
    int coeffn = yDim * (xPts[0] - 1) * 4;
    coeff = (double*) malloc(sizeof(double) * coeffn);
  } // siteReductionDim == 0

  // do the vector site dimension
  dfdNewTask1D(&interp, xPts[0], xGrid[0], DF_NO_HINT, yDim, lastInterpResult, DF_NO_HINT);
  dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
  dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);

  int coeffns[MAXDIM];
  coeffns[0] = yDim;
  coeffns[1] = (xPts[0] - 1) * 4;

  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      int cellofsite;
      // search once
      locate(xGrid[0], xPts[0] - 2, siteEval[i], &cellofsite);
      // store search result
      outputIndex[i] = cellofsite;
      double xsite = siteEval[i] - xGrid[0][cellofsite];
      for (j = 0; j < nvec; ++j) {
        // double recevalNoSearch4(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns)
        result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], 1, coeff, coeffns);
      }
    }
  } else {
    // full vector dimension
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      int cellofsite;
      // search once
      locate(xGrid[0], xPts[0] - 2, siteEval[i], &cellofsite);
      // store search result
      outputIndex[i] = cellofsite;
      double xsite = siteEval[i] - xGrid[0][cellofsite];
      for (j = 0; j < yDim; ++j) {
        result[i*yDim + j] = evalNoSearch4(0, &xsite, &cellofsite, j * coeffns[1], 1, coeff, coeffns);
      }
    }
  }

  // delete everything
  dfDeleteTask(&interp);
  free(coeff);
  free(lastInterpResult);
}

void partialConstructEvalInOut(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // deal with breaks
  // get dimension of grid points from breaks
  int xDim = mxGetNumberOfElements(breaks_);
  // extract grid points from breaks
  double* xGrid[MAXDIM];
  int xPts[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    mxArray* breaksElement = mxGetCell(breaks_, i);
    xGrid[i] = mxGetPr(breaksElement);
    xPts[i] = mxGetNumberOfElements(breaksElement);
  }

  // deal with coefs
  // extract vector function from coefs
  double* y = mxGetPr(values_);

  // deal with pieces

  // deal with order
  int* order = (int*) mxGetData(order_);
  int s_order = order[0];

  // deal with dim
  int* dim = (int*) mxGetData(dim_);
  int yDim = dim[0];

  // deal with x
  int siteDim = mxGetNumberOfElements(x_);
  int siteReductionDim = siteDim - 1;
  /*
#ifndef NOCHECK
  errorCheck(siteReductionDim > 0, "in partialConstructionEvaluation: siteReductionDim should > 0");
#endif
*/
  double* siteEval;
  double siteReduction[MAXDIM];
  // the first dimension is site vector
  mxArray* siteElement = mxGetCell(x_, 0);
  siteEval = mxGetPr(siteElement);
  int nsite = mxGetNumberOfElements(siteElement);
  // the last 2:end dimension would be scalar
  for (int i = 1; i < siteDim; ++i) {
    mxArray* siteElement = mxGetCell(x_, i);
    double* site_reduction_ptr = mxGetPr(siteElement);
    siteReduction[i-1] = *site_reduction_ptr;
  }

  // deal with left

  // deal with indexDim
  int nvec = mxGetM(indexDim_);
  int* ivec = (int*) mxGetData(indexDim_);

  // deal with index_x
#ifndef NOCHECK
  errorCheck(mxIsCell(index_x_), "in cell form: index_x should be cell");
#endif
  mxArray* inputIndexMx = mxGetCell(index_x_, 0);
#ifndef NOCHECK
  errorCheck(mxIsInt32(inputIndexMx), "in cell form: each element of index_x should be int32 array");
  errorCheck(mxGetNumberOfElements(inputIndexMx) == nsite, "in cell form, dimension of element of index_x should agree with nsite");
#endif
  int* inputIndex = (int*) mxGetData(inputIndexMx);

  // allocate result
  if (nvec > 0) {
    result_ = mxCreateDoubleMatrix(nvec, nsite, mxREAL);
  } else {
    result_ = mxCreateDoubleMatrix(yDim, nsite, mxREAL);
  }
  double* result = mxGetPr(result_);

  // deal with outputIndex, cell form
  outputIndex_ = mxCreateCellMatrix(xDim, 1);
  // vector site
  mxSetCell(outputIndex_, 0, mxCreateNumericArray(2, (int[]){nsite,1}, mxINT32_CLASS, mxREAL));
  // scalar site
  int* outputReductionIndex[MAXDIM];
  for (int i = 0; i < siteReductionDim; ++i) {
    mxSetCell(outputIndex_, i+1, mxCreateNumericArray(2, (int[]){1,1}, mxINT32_CLASS, mxREAL));
    outputReductionIndex[i] = (int*) mxGetData(mxGetCell(outputIndex_, i+1));
  }
  int* outputIndex = (int*) mxGetData(mxGetCell(outputIndex_, 0));

  // determine coeff dimension

  // set num of threads
  mkl_set_num_threads(NUM_THREADS);
  omp_set_num_threads(NUM_THREADS);

    // iterator
  int i,j;

  // iteratively construct coeffs and evaluate using intel-mkl, from the highest dimension
  DFTaskPtr interp;
  MKL_INT s_type;
  MKL_INT bc_type;
  switch (s_order) {
  case 4:
    s_type = DF_PP_NATURAL;
    bc_type = DF_BC_NOT_A_KNOT;
    break;
  case 2:
    s_type = DF_PP_DEFAULT;
    bc_type = DF_NO_BC;
    break;
  }

  double* lastInterpResult;
  double* coeff;
  if (siteReductionDim > 0) {
    // construct the num of vector functions for each construction
    int sudoVecNum[MAXDIM];
    sudoVecNum[0] = yDim;
    for (i = 1; i < xDim; ++i) {
      sudoVecNum[i] = sudoVecNum[i-1] * xPts[i-1];
    }
    // do it once at the last scalar dimension
    int cDim = xDim - 1;
    lastInterpResult = y;
    dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
    coeff = (double*) malloc((xPts[cDim]-1) * sudoVecNum[cDim] * s_order * sizeof(double));
    dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
    dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
    int dorder = 1;
    double* interpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    dfdInterpolate1D(interp, DF_INTERP | DF_CELL, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, outputReductionIndex[cDim-1]);
    *outputReductionIndex[cDim-1] = indexMklToNr3(*outputReductionIndex[cDim-1], xPts[cDim]);
    dfDeleteTask(&interp);
    lastInterpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);

    // do the remaining scalar dimension
    for (i = 1; i < siteReductionDim; ++i) {
      int cDim = xDim - 1 - i;
      dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
      dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
      dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
      dfdInterpolate1D(interp, DF_INTERP | DF_CELL, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, outputReductionIndex[cDim-1]);
      *outputReductionIndex[cDim-1] = indexMklToNr3(*outputReductionIndex[cDim-1], xPts[cDim]);
      dfDeleteTask(&interp);
      memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);
    }
    free(interpResult);
  } // siteReductionDim > 0
  else {
    lastInterpResult = (double*) malloc(sizeof(double) * yDim * xPts[0]);
    doublememcpy(lastInterpResult, y, yDim * xPts[0]);
    int coeffn = yDim * (xPts[0] - 1) * 4;
    coeff = (double*) malloc(sizeof(double) * coeffn);
  } // siteReductionDim == 0

  // do the vector site dimension
  dfdNewTask1D(&interp, xPts[0], xGrid[0], DF_NO_HINT, yDim, lastInterpResult, DF_NO_HINT);
  dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
  dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);

  int coeffns[MAXDIM];
  coeffns[0] = yDim;
  coeffns[1] = (xPts[0] - 1) * 4;

  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      int cellofsite = inputIndex[i];
      // search once
      hunt(xGrid[0], xPts[0] - 2, siteEval[i], &cellofsite);
      // store search result
      outputIndex[i] = cellofsite;
      double xsite = siteEval[i] - xGrid[0][cellofsite];
      for (j = 0; j < nvec; ++j) {
        result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], 1, coeff, coeffns);
      }
    }
  } else {
    // full vector dimension
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      int cellofsite = inputIndex[i];
      // search once
      hunt(xGrid[0], xPts[0] - 2, siteEval[i], &cellofsite);
      // store search result
      outputIndex[i] = cellofsite;
      double xsite = siteEval[i] - xGrid[0][cellofsite];
      for (j = 0; j < yDim; ++j) {
        result[i*yDim + j] = evalNoSearch4(0, &xsite, &cellofsite, j * coeffns[1], 1, coeff, coeffns);
      }
    }
  }

  // delete everything
  dfDeleteTask(&interp);
  free(coeff);
  free(lastInterpResult);
}

inline
void locate(double xx[], int n, double x, int *j)
{
  int ju,jm,jl;

  jl=0;
  ju=n+1;
  while (ju-jl > 1) {
    jm=(ju+jl) >> 1;
    if (x >= xx[jm])
      jl=jm;
    else
      ju=jm;
  }
  if (x == xx[1]) *j=1;
  else if(x == xx[n]) *j=n-1;
  else *j=jl;
}

inline
void hunt(double xx[], int n, double x, int *jlo)
{
  int jm,jhi,inc;

  if (*jlo <= 0 || *jlo > n) {
    *jlo=0;
    jhi=n+1;
  } else {
    inc=1;
    if (x >= xx[*jlo]) {
      if (*jlo == n) return;
      jhi=(*jlo)+1;
      while (x >= xx[jhi]) {
        *jlo=jhi;
        inc += inc;
        jhi=(*jlo)+inc;
        if (jhi > n) {
          jhi=n+1;
          break;
        }
      }
    } else {
      if (*jlo == 1) {
        *jlo=0;
        return;
      }
      jhi=(*jlo)--;
      while (x < xx[*jlo]) {
        jhi=(*jlo);
        inc <<= 1;
        if (inc >= jhi) {
          *jlo=0;
          break;
        }
        else *jlo=jhi-inc;
      }
    }
  }
  while (jhi-(*jlo) != 1) {
    jm=(jhi+(*jlo)) >> 1;
    if (x >= xx[jm])
      *jlo=jm;
    else
      jhi=jm;
  }
  if (x == xx[n]) *jlo=n-1;
  if (x == xx[1]) *jlo=1;
}

inline
double recevalNoSearch2(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns)
{
  // port from Wenlan's NdInterp::recevalNoSearch2 method, modified without using Matrix class
  // recursive evaluation of a piece of coefficient at certain dimension
  double r;
  if (idim == xDim-1) {
    // last dimension
    double* pcoeff = coeff + shift + (*cellofsite+1) * 2;
    r = *(--pcoeff);
    r *= *xsite;
    r += *(--pcoeff);
  } else {
    idim++;
    shift += (*(cellofsite++)+1) * 2;
    r = recevalNoSearch2(idim, xsite+1, cellofsite, (--shift) * coeffns[idim+1], xDim, coeff, coeffns);
    r *= *xsite;
    r += recevalNoSearch2(idim, xsite+1, cellofsite, (--shift) * coeffns[idim+1], xDim, coeff, coeffns);
  }
  return r;
}

inline
double evalNoSearch2(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns)
{
  // recursive evaluation of a piece of coefficient at certain dimension
  double r;
  double* pcoeff = coeff + shift + (*cellofsite+1) * 2;
  r = *(--pcoeff);
  r *= *xsite;
  r += *(--pcoeff);
  return r;
}

double recevalNoSearch4(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns)
{
  // port from Wenlan's NdInterp::recevalNoSearch4 method, modified without using Matrix class
  // recursive evaluation of a piece of coefficient at certain dimension
  double r;
  if (idim == xDim-1) {
    // last dimension
    double* pcoeff = coeff + shift + (*cellofsite+1) * 4;
    r = *(--pcoeff);
    r *= *xsite;
    r += *(--pcoeff);
    r *= *xsite;
    r += *(--pcoeff);
    r *= *xsite;
    r += *(--pcoeff);
  } else {
    idim++;
    shift += (*(cellofsite++)+1) * 4;
    r = recevalNoSearch4(idim, xsite+1, cellofsite, (--shift) * coeffns[idim+1], xDim, coeff, coeffns);
    r *= *xsite;
    r += recevalNoSearch4(idim, xsite+1, cellofsite, (--shift) * coeffns[idim+1], xDim, coeff, coeffns);
    r *= *xsite;
    r += recevalNoSearch4(idim, xsite+1, cellofsite, (--shift) * coeffns[idim+1], xDim, coeff, coeffns);
    r *= *xsite;
    r += recevalNoSearch4(idim, xsite+1, cellofsite, (--shift) * coeffns[idim+1], xDim, coeff, coeffns);
  }
  return r;
}

inline
double evalNoSearch4(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns)
{
  // recursive evaluation of a piece of coefficient at certain dimension
  double r;
  // last dimension
  double* pcoeff = coeff + shift + (*cellofsite+1) * 4;
  r = *(--pcoeff);
  r *= *xsite;
  r += *(--pcoeff);
  r *= *xsite;
  r += *(--pcoeff);
  r *= *xsite;
  r += *(--pcoeff);
  return r;
}

double recevalNoSearchN(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns, int* s_order)
{
  // port from Wenlan's NdInterp::recevalNoSearchN method, modified without using Matrix class
  // recursive evaluation of a piece of coefficient at certain dimension
  double r;
  if (idim == xDim-1) {
    // last dimension
    double* pcoeff = coeff + shift + (*cellofsite+1) * s_order[idim];
    r = *(--pcoeff);
    for (int i = 0; i < s_order[idim]-1; ++i) {
      r *= *xsite;
      r += *(--pcoeff);
    }
  } else {
    shift += (*(cellofsite++)+1) * s_order[idim];
    r = recevalNoSearchN(idim+1, xsite+1, cellofsite, (--shift) * coeffns[idim+2], xDim, coeff, coeffns, s_order);
    for (int i = 0; i < s_order[idim]-1; ++i) {
      r *= *xsite;
      r += recevalNoSearchN(idim+1, xsite+1, cellofsite, (--shift) * coeffns[idim+2], xDim, coeff, coeffns, s_order);
    }
  }
  return r;
}

void mkl_partial_construction_evaluation(
    int xDim, double** xGrid, int* xPts,
    double* y, int yDim,
    int s_order,
    int nsite, double* siteEval, double* siteReduction, int siteReductionDim, 
    int nvec, int* ivec,
    double* result)
{
  // iterator
  int i,j;

  // iteratively construct coeffs and evaluate using intel-mkl, from the highest dimension
  DFTaskPtr interp;
  MKL_INT s_type;
  MKL_INT bc_type;
  switch (s_order) {
  case 4:
    s_type = DF_PP_NATURAL;
    bc_type = DF_BC_NOT_A_KNOT;
    break;
  case 2:
    s_type = DF_PP_DEFAULT;
    bc_type = DF_NO_BC;
    break;
  }

  double* lastInterpResult;
  double* coeff;
  if (siteReductionDim > 0) {
    // construct the num of vector functions for each construction
    int sudoVecNum[MAXDIM];
    sudoVecNum[0] = yDim;
    for (i = 1; i < xDim; ++i) {
      sudoVecNum[i] = sudoVecNum[i-1] * xPts[i-1];
    }
    // do it once at the last scalar dimension
    int cDim = xDim - 1;
    lastInterpResult = y;
    dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
    coeff = (double*) malloc((xPts[cDim]-1) * sudoVecNum[cDim] * s_order * sizeof(double));
    dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
    dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
    int dorder = 1;
    double* interpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, NULL);
    dfDeleteTask(&interp);
    lastInterpResult = (double*) malloc(sudoVecNum[cDim] * sizeof(double));
    memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);

    // do the remaining scalar dimension
    for (i = 1; i < siteReductionDim; ++i) {
      int cDim = xDim - 1 - i;
      dfdNewTask1D(&interp, xPts[cDim], xGrid[cDim], DF_NO_HINT, sudoVecNum[cDim], lastInterpResult, DF_NO_HINT);
      dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
      dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);
      dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, 1, siteReduction+cDim-1, DF_NO_HINT, 1, &dorder, NULL, interpResult, DF_NO_HINT, NULL);
      dfDeleteTask(&interp);
      memcpy(lastInterpResult, interpResult, sizeof(double) * sudoVecNum[cDim]);
    }
    free(interpResult);
  } // siteReductionDim > 0
  else {
    lastInterpResult = (double*) malloc(sizeof(double) * yDim * xPts[0]);
    doublememcpy(lastInterpResult, y, yDim * xPts[0]);
    int coeffn = yDim * (xPts[0] - 1) * 4;
    coeff = (double*) malloc(sizeof(double) * coeffn);
  } // siteReductionDim == 0

  // do the vector site dimension
  dfdNewTask1D(&interp, xPts[0], xGrid[0], DF_NO_HINT, yDim, lastInterpResult, DF_NO_HINT);
  dfdEditPPSpline1D(interp, s_order, s_type, bc_type, 0, DF_NO_IC, 0, coeff, DF_NO_HINT);
  dfdConstruct1D(interp, DF_PP_SPLINE, DF_METHOD_STD);

  int coeffns[MAXDIM];
  coeffns[0] = yDim;
  coeffns[1] = (xPts[0] - 1) * 4;

  if (nvec > 0) {
#pragma omp parallel for private(i, j)
    for (i = 0; i < nsite; ++i) {
      int cellofsite;
      // search once
      locate(xGrid[0], xPts[0] - 2, siteEval[i], &cellofsite);
      double xsite = siteEval[i] - xGrid[0][cellofsite];
      for (j = 0; j < nvec; ++j) {
        // double recevalNoSearch4(int idim, double* xsite, int* cellofsite, int shift, int xDim, double* coeff, int* coeffns)
        result[i*nvec + j] = evalNoSearch4(0, &xsite, &cellofsite, ivec[i*nvec + j] * coeffns[1], 1, coeff, coeffns);
      }
    }
  } else {
    // full vector dimension
    double* resultTemp = (double*) malloc (sizeof(double) * nsite * yDim);
    int dorder = 1;
    dfdInterpolate1D(interp, DF_INTERP, DF_METHOD_PP, nsite, siteEval, DF_NO_HINT, 1, &dorder, NULL, resultTemp, DF_NO_HINT, NULL);
    mkl_domatcopy('R', 'T', yDim, nsite, 1.0, resultTemp, nsite, result, yDim);
    free(resultTemp);
  }

  // delete everything
  dfDeleteTask(&interp);
  free(coeff);
  free(lastInterpResult);
}

void mkl_start(int xDim, int* xPts, double** xGrid, int valDim, double* y, int* s_order, double* coeff, int coeffn)
{
  // copy from Wenlan's NdInterp::start(), modified without using Matrix class
  double* lastCoeff = (double*) malloc(sizeof(double) * coeffn);
  int lastCoeffns[MAXDIM];
  int lastCoeffn;

  // for the first coordinate, the lastCoeff is just y
  int initialCoeffSize[MAXDIM+1];
  initialCoeffSize[0] = valDim;
  for (int i = 0; i < xDim; ++i) {
    initialCoeffSize[i+1] = xPts[i];
  }

  int coeffns[MAXDIM];
  intmemcpy(coeffns, initialCoeffSize, xDim+1);
  coeffn = vectorProd(coeffns, xDim+1);
  doublememcpy(coeff, y, coeffn);

  // the following is a mimic of csape
  MKL_INT s_type[MAXDIM];
  MKL_INT bc_type[MAXDIM];
  for (int i = 0; i < xDim; ++i) {
    switch (s_order[i]) {
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

  // Let me do Mkl Interp at low level to boost performance
  DFTaskPtr interp0;
  int currentVecSize = 1; // not used
  for (int i = xDim-1; i >= 0; --i) {
    // carry out coordinatewise interpolation at coordinate i
    // the interpolation is always w.r.t. the last coordinate, w.r.t. to which y are stored adjacently
    // calculate new size of vector function, i.e. the total size except the last dimension
    currentVecSize = vectorProd(coeffns, xDim);

    // interp0 = new MklInterp(interpType, xGrid[i], xPts[i], lastCoeff->pt(), currentVecSize, 'n');
    // direct coefficients of MklInterp to the array in NdInterp

    // compute the new size of coeffs
    // the last dimension reduces by 1, and times by s_order
    coeffns[xDim] = coeffns[xDim] - 1;
    coeffns[xDim] = coeffns[xDim] * s_order[i];
    // update last Coeff
    intmemcpy(lastCoeffns, coeffns, xDim+1);
    lastCoeffn = vectorProd(lastCoeffns, xDim+1);

    // compute coefficients
    dfdNewTask1D(&interp0, xPts[i], xGrid[i], DF_NO_HINT, currentVecSize, coeff, DF_NO_HINT);
    dfdEditPPSpline1D(interp0, s_order[i], s_type[i], bc_type[i], 0, DF_NO_IC, 0, lastCoeff, DF_NO_HINT);
    dfdConstruct1D(interp0, DF_PP_SPLINE, DF_METHOD_STD);
    dfDeleteTask(&interp0);

    // forming new interpolation problem, by permuting the last dimension to the first one
    if (xDim > 1) {
      int vecShift = lastCoeffn / valDim;

      double* des = coeff;
      double* src = lastCoeff;
      int transRow = vecShift / lastCoeffns[xDim];
      int transCol = lastCoeffns[xDim];
      for (int i = 0; i < valDim; ++i) {
        mkl_domatcopy('R', 'T', transRow, transCol, 1.0, src, transCol, des, transRow);
        des += vecShift;
        src += vecShift;
      }
      // swap dimension
      int temp = lastCoeffns[xDim];
      for (int i = xDim; i >= 2; --i) {
        lastCoeffns[i] = lastCoeffns[i-1];
      }

      lastCoeffns[1] = temp;
      intmemcpy(coeffns, lastCoeffns, xDim+1);
      coeffn = vectorProd(coeffns, xDim+1);
    } else {
      doublememcpy(coeff, lastCoeff, lastCoeffn);
    }
  }

  free(lastCoeff);
}

inline
void intmemcpy(int* des, int* src, int n)
{
  memcpy(des, src, sizeof(int) * n);
}

inline
void doublememcpy(double* des, double* src, int n)
{
  memcpy(des, src, sizeof(double) * n);
}

inline
int vectorProd(int* x, int n) {
  int result = x[0];
  for (int i = 1; i < n; ++i) {
    result *= x[i];
  }
  return result;
}

inline
int indexMklToNr3(int oldidx, int xPts)
{
  return MIN(MAX(0, oldidx-1), xPts-2);
}



