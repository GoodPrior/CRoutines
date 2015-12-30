/**********************************************************
 * This is a modified interface to NR3 optimizers.
 * It's a c interface.
 * It allows to pass function parameters.
 * Function should have the prototype
 * for golden and brent: void f(double x, double* fx, void* params)
 * for dbrent: void f(double x, double* fx, double* dfx, void* params)
 *********************************************************************/
#include <cmath>
#include <cstdio>

#define NRANSI
#define R 0.61803399
#define C (1.0-R)
#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);
#define MOV3(a,b,c, d,e,f) (a)=(d);(b)=(e);(c)=(f);
#define SHFT2(a,b,c) (a)=(b);(b)=(c);
#define SHFT3(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

double golden(double ax, double bx, double cx, double (*f)(double, void*), void* params, double tol, double *xmin)
{
    double f1,f2,x0,x1,x2,x3;
    int iter = 0;

    x0=ax;
    x3=cx;
    if (fabs(cx-bx) > fabs(bx-ax)) {
	x1=bx;
	x2=bx+C*(cx-bx);
    } else {
	x2=bx;
	x1=bx-C*(bx-ax);
    }
    f1=(*f)(x1, params);
    f2=(*f)(x2, params);
    while (fabs(x3-x0) > tol) {
    	iter++;
	if (f2 < f1) {
	    SHFT3(x0,x1,x2,R*x1+C*x3)
		SHFT2(f1,f2,(*f)(x2, params))
	} else {
	    SHFT3(x3,x2,x1,R*x2+C*x0)
		SHFT2(f2,f1,(*f)(x1, params))
	}
    }
    if (f1 < f2) {
	*xmin=x1;
	return f1;
    } else {
	*xmin=x2;
	return f2;
    }
}

double brent(double ax, double bx, double cx, double (*f)(double, void*), void* params, double tol, double *xmin)
{
    int iter;
    double a,b,d,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
    double e=0.0;

    a=(ax < cx ? ax : cx);
    b=(ax > cx ? ax : cx);
    x=w=v=bx;
    fw=fv=fx=(*f)(x, params);
    while (1) {
	xm=0.5*(a+b);
	tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
	if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
	    *xmin=x;
	    return fx;
	}
	if (fabs(e) > tol1) {
	    r=(x-w)*(fx-fv);
	    q=(x-v)*(fx-fw);
	    p=(x-v)*q-(x-w)*r;
	    q=2.0*(q-r);
	    if (q > 0.0) p = -p;
	    q=fabs(q);
	    etemp=e;
	    e=d;
	    if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
		d=CGOLD*(e=(x >= xm ? a-x : b-x));
	    else {
		d=p/q;
		u=x+d;
		if (u-a < tol2 || b-u < tol2)
		    d=SIGN(tol1,xm-x);
	    }
	} else {
	    d=CGOLD*(e=(x >= xm ? a-x : b-x));
	}
	u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
	fu=(*f)(u, params);
	if (fu <= fx) {
	    if (u >= x) a=x; else b=x;
	    SHFT(v,w,x,u)
		SHFT(fv,fw,fx,fu)
	} else {
	    if (u < x) a=u; else b=u;
	    if (fu <= fw || w == x) {
		v=w;
		w=u;
		fv=fw;
		fw=fu;
	    } else if (fu <= fv || v == x || v == w) {
		v=u;
		fv=fu;
	    }
	}
    }
    *xmin=x;
    return fx;
}

double dbrent(double ax, double bx, double cx, double (*f)(double, double*, void*), void* params, double tol, double *xmin)
{
    int iter,ok1,ok2;
    double a,b,d,d1,d2,du,dv,dw,dx,e=0.0;
    double fu,fv,fw,fx,olde,tol1,tol2,u,u1,u2,v,w,x,xm;
    double dfx, dfu;


    a=(ax < cx ? ax : cx);
    b=(ax > cx ? ax : cx);
    x=w=v=bx;

    fw=fv=fx=(*f)(x, &dfx, params);
    dw=dv=dx=dfx;
    for (iter=1;iter<=ITMAX;iter++) {
	xm=0.5*(a+b);
	tol1=tol*fabs(x)+ZEPS;
	tol2=2.0*tol1;
	if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
	    *xmin=x;
	    return fx;
	}
	if (fabs(e) > tol1) {
	    d1=2.0*(b-a);
	    d2=d1;
	    if (dw != dx) d1=(w-x)*dx/(dx-dw);
	    if (dv != dx) d2=(v-x)*dx/(dx-dv);
	    u1=x+d1;
	    u2=x+d2;
	    ok1 = (a-u1)*(u1-b) > 0.0 && dx*d1 <= 0.0;
	    ok2 = (a-u2)*(u2-b) > 0.0 && dx*d2 <= 0.0;
	    olde=e;
	    e=d;
	    if (ok1 || ok2) {
		if (ok1 && ok2)
		    d=(fabs(d1) < fabs(d2) ? d1 : d2);
		else if (ok1)
		    d=d1;
		else
		    d=d2;
		if (fabs(d) <= fabs(0.5*olde)) {
		    u=x+d;
		    if (u-a < tol2 || b-u < tol2)
			d=SIGN(tol1,xm-x);
		} else {
		    d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
		}
	    } else {
		d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
	    }
	} else {
	    d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
	}
	if (fabs(d) >= tol1) {
	    u=x+d;
	    fu=(*f)(u, &dfu, params);
	} else {
	    u=x+SIGN(tol1,d);
	    fu=(*f)(u, &dfu, params);
	    if (fu > fx) {
		*xmin=x;
		return fx;
	    }
	}
	du=dfu;
	if (fu <= fx) {
	    if (u >= x) a=x; else b=x;
	    MOV3(v,fv,dv, w,fw,dw)
		MOV3(w,fw,dw, x,fx,dx)
		MOV3(x,fx,dx, u,fu,du)
	} else {
	    if (u < x) a=u; else b=u;
	    if (fu <= fw || w == x) {
		MOV3(v,fv,dv, w,fw,dw)
		    MOV3(w,fw,dw, u,fu,du)
	    } else if (fu < fv || v == x || v == w) {
		MOV3(v,fv,dv, u,fu,du)
	    }
	}
    }
    return 0.0;
}

#undef C
#undef R
#undef SHFT2
#undef SHFT3
#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SHFT
#undef MOV3
#undef NRANSI
#undef SIGN
