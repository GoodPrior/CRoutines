inline double newton1D(double ax, double bx, double cx, double(*f)(double x, double* g, double* h, void*), void* params, double tol, int maxiter, double* xmin, int* exitflag, int* iter)
{
    *exitflag = -1; // unknown error
    *xmin = cx;
    double v;
    for (*iter = 0; *iter < maxiter; ++(*iter)) {
        // compute function value, gradient and hessian
        double g, h;
        v = f(*xmin, &g, &h, params);
        double newx = *xmin - g / h;
        // min max
        newx = (newx > ax) ? newx : ax;
        newx = (newx < bx) ? newx : bx;
        if (abs(newx - (*xmin)) < tol)
        {
            *exitflag = 1;
            *xmin = newx;
            return v;
        }
        // update *x
        *xmin = newx;
    }
    *exitflag = 0; // exceeds maxiter
    return v;
}
