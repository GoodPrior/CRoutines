#define BZ_THREADSAFE
#include "blitz/array.h"
#include "IntelInterp.h"

using namespace blitz;
using namespace IntelInterp;
using namespace blitz::tensor;

int main() 
{
	double XMax = 5;
	int XPts = 100;
	Array<double, 1> X(XPts);
	X = i / (double)XPts*XMax;

	double YMax = 6;
	int YPts = 100;
	Array<double, 1> Y(YPts);
	Y = i / (double)YPts*YMax;

	Array<double, 3> Z(2, XPts, YPts);
	Z(0, Range::all(), Range::all()) = X(i) * Y(j);
	Z(1, Range::all(), Range::all()) = pow(X(i), 2) * pow(Y(j), 2);
	// cout << Z << endl;

	Array<double, 1> XSite(4);
	XSite = 1.7, 2.7, 3.3, 4.2;
	Array<double, 1> YSite(4);
	YSite = 2.5, 3.3, 4.4, 5.1;
	Array<double, 2> ActualValue(4, 2);
	ActualValue(Range::all(), 0) = XSite(i) * YSite(i);
	ActualValue(Range::all(), 1) = pow(XSite(i), 2) * pow(YSite(i), 2);

	Array<double, 2> Sites(4, 2);
	Sites(Range::all(), 0) = XSite;
	Sites(Range::all(), 1) = YSite;

	cout << Sites(0) << endl;
	cout << Sites(3) << endl;
	cout << *Sites.data() << endl;
	cout << *(Sites.data() + 3) << endl;
	cout << *(Sites.data() + 6) << endl;

	/*
	CubicSpline TestSpline;
	{
		int CordPts[] = { XPts, YPts };
		double* XGrid[] = { X.data(), Y.data() };
		int SOrder[] = { 4, 4 };

		TestSpline.
			set_XDim(2).
			set_XPts(CordPts).
			set_XGrid(XGrid).
			set_VecDim(2).
			set_SOrder(SOrder);
	}

	// Allocate space for Interp
	TestSpline.alloc();
	// Construct coefficients
	TestSpline.construct(Z.data());

	// interpolate over sites
	int NumOfVec = 2;
	int NumOfSites = 4;

	Array<int, 2> CellOfSite(NumOfSites, 2);
	CellOfSite = 0;

	Array<int, 2> VecIdx(NumOfSites, NumOfVec);
	VecIdx(Range::all(), 0) = 0;
	VecIdx(Range::all(), 1) = 1;
	cout << VecIdx << endl;

	Array<double,2> InterpValue = TestSpline.veceval_uniform_x(&Sites, &CellOfSite, &VecIdx);
	cout << InterpValue << endl;
	cout << ActualValue << endl;

	TestSpline.dealloc();
	*/
};
