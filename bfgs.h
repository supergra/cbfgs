double OptimizeBFGS(const int ndv,
                    double *x,
                    const double *lb,
                    const double *ub,
                    double *H,
                    double (*FGeval)(const int, const double *, double *));
