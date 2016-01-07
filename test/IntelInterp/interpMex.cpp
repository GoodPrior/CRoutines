#include "mex.h"
#include "IntelInterp.h"
#include "MatlabMatrix.h"

using namespace IntelInterp;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	GET_INT(XPts);
	GET_DM(ppCoefs, 2);
	GET_DV(XGrid);
	GET_INT(SitePts);
	GET_DV(YSite);
	GET_DV(Idx);
	GET_DV(XSite);
	GET_INT(ppOrder);

	for (int i = 1; i <= SitePts ; i++)
	{
		int idx = (int)Idx(i) - 1;
		if (ppOrder == 2) {
			YSite(i) = search_eval2_1d(_XGrid, XPts, _ppCoefs, XSite(i), idx);
		}
		else if (ppOrder == 4){
			YSite(i) = search_eval4_1d(_XGrid, XPts, _ppCoefs, XSite(i), idx);
		}
	}

	PUT(YSite);
}
