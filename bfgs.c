#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <assert.h>

/* Macros -- interp wants these */
#define MIN(A,B) (((A)>(B))?(B):(A))
#define MAX(A,B) (((A)>(B))?(A):(B))

#include "matrix.h"
#include "interp.h"
#include "bfgs.h"

/* Global variables */
int nfev; // Eval count

#define CONVEX_NOISY 0

#define SAVE_ALL_HESSIANS 0

/* BFGS Parameters */
#define STP0 1.0 /* scale first step to STP0/||g|| */
#define INIT_DX 1.0 /* 1 means start line search guess with full BFGS step */

/* Line search parameters */
#define MAX_LINE_STEPS 20
#define MAX_ZOOM_STEPS 30
#define MU_1      1.e-4   /* Sufficient decrease */
#define MU_2      0.99    /* Sufficient grad decrease */
#define FACTOR    2.0     /* Step size increase factor in line search */
#define POLY_SAFEGUARD_HI 0.66 /* GSEARCH factor */
#define POLY_SAFEGUARD_LO 0.01 /* Made up */
#define STRONGWOLFE 0 /* 0 is Weak Wolfe */

/* Convergence Criteria */
#define MAX_MAJOR_ITER  300 // Major iteration limit
#define EPS_A (1.e-15)           /* Absolute tolerance */
#define EPS_R (1.e-15)             /* Relative tolerance */
#define EPS_G (1.e-7)           /* Gradient tolerance */


/* Line search for convex function that only uses gradients, not
 * function values. */
double GradLineSearch(const int ndv, const double *x0, const double *lb, const double *ub, const double *sdir, const double init_dx, const double f0, const double *g0, double *f_final, double *g_final, double (*FGeval)(const int, const double *, double *))
{
  int i,k;
  double f, fl, fr;
  double dx, dxl, dxr;
  double gs, gsl, gsr, gs0;
  int phase_one = 1; /* Start with exponential growth until approaching max */
  int min_bracketed = 0;

  double x[ndv];
  double g[ndv];
  double gl[ndv]; /* Save full left grad to return */
  double gr[ndv]; /* Will never return right side so don't need this */

  /* Compute max step size based on min/max variable bounds */
  double bnd = 1.e100;
  for (i=0;i<ndv;i++){
    if      (sdir[i]>0)
      bnd = MIN(bnd,(ub[i]-x0[i])/sdir[i]); /* Only upper bound can be active */
    else if (sdir[i]<0)
      bnd = MIN(bnd,(lb[i]-x0[i])/sdir[i]); /* Only lower bound can be active */
  }
  const double dx_max = bnd;
  if (dx_max <= EPS_A){
    printf("Too close to bounds\n");
    return 0;
  }

  /* Left endpoint starts at 0 */
  dxl = 0.0;
  fl = f0;
  vcopy(g0, gl, ndv);
  gs0 = gsl = vdot(g0, sdir, ndv);
  /* Gradient must be negative in search direction */
  assert (gsl < 0); /* Already tested this outside */

  /* Right endpoint is unitialized */
  dxr = 1.e100;
  fr  = 1.e100;
  gsr = 1.e100;

  /* Set initial step */
  dx = init_dx; /* Usually this is just 1 */
  if (dx > dx_max){ /* Initial step violates bounds */
    phase_one = 0;
    dx = dx_max;
  }
  double sdir_mag = vmag(sdir,ndv);
  printf("  GLS: g0s=%.6g, ||sdir||=%.6g\n",gs0,sdir_mag);fflush(stdout);

  /* Keep increasing the step until zero derivative is bracketed,
   * and then work inward with interpolation. */

  *f_final = f0; /* Initialize these just in case we break early */
  vcopy(g0, g_final, ndv);

  for(k=0;k<MAX_LINE_STEPS;k++){
    assert(dx >= EPS_A);
    assert(dx <= dx_max);

    /* Apply step */
    for (i=0;i<ndv;i++)
      x[i] = x0[i] + dx*sdir[i];
    //printf("x = "); PrintVec(x,ndv);


    f = FGeval(ndv, x, g);  /* Compute functional and gradient */
    nfev++;
    gs = vdot(g,sdir,ndv);
    printf("  GLS %d: dx = %g : f=%.6g, gs=%.6g\n",k,dx,f,gs);fflush(stdout);

    /* For a convex function, the new gradient must fall between the current
     * left and right bounds.  */
    if ( gs < gsl || gs > gsr){
      printf("Warning: Function is non-convex along search direction.\n");
      if (0){ /* Whether to accept a nonconvex region */
        fflush(stdout);
        exit(1);
      }
    }

    /* New point replaces one of the previous ones */
    if (gs > 0 ){   /* New right side */
      min_bracketed = 1;
      dxr = dx;
      fr = f;
      vcopy(g, gr, ndv);  // Don't need full right hand gradient
      gsr = gs;
    }else{          /* New left side */
      dxl = dx;
      fl = f;
      vcopy(g, gl, ndv); /* Save full lefthand gradient */
      gsl = gs;
    }

    /* Check for sufficient gradient reduction (only on left side) */
    /* Note this is neither Weak nor Strong Wolfe
     * Sign change does not count as converged. */
    if (-gsl <= -MU_2*gs0){
      printf("GLS: Met gradient reduction criterion\n");
      break;
    }

    /* This would allow termination based on the Strong Wolfe curvature
     * condition from the right side -- not advisable for general functions
     * You can ping-pong back and forth, and you can get an arbitrarily
     * large function increase, especially for one-sided penalties. */
    // if (fabs(gsr) <= -MU_2*gs0){
    //   printf("BREAKING BAD\n");
    //   fl = fr;
    //   vcopy(gr, gl, ndv);
    //   dxl = dxr;
    //   break;
    // }
    //

    /* Once we've bracketed the minimum, work inward via interpolation */
    if (min_bracketed){
      assert(dxr > 0); /* Must have been initialized */

      /* Safeguarded polynomial interpolation */
      double dxlo = dxl + 0.1*(dxr-dxl);
      double dxhi = dxl + 0.5*(dxr-dxl); /* Min of quad and bisection */

      /* This is problematic, because it can take an initial big step to
         the right, and then slowly converge from the right, never triggering
         the sufficient grad reduction (which must be on the left). */
      dx = quad_g2(dxl,gsl,dxr,gsr,dxlo,dxhi);

      // Bisect dumbness
      //dx = MIN(0.5*(dxl+dxr), dx); /* This can work faster sometimes */

    }else{  /* Not bracketed yet -- increase step size and continue */
      if ((dx_max-dx) < EPS_A){ /* Prev. step already at upper bound */
        printf("GLS: Hit DV bounds.\n");fflush(stdout);
         /* Have to just exit, because we haven't decreased the gradient, and
          * so the BFGS update would be bad. Could also just return,
          * and have the calling function handle it. */
        exit(1);
      }
      dx = MIN(dx_max, dx*FACTOR); /* Bounded exponential expansion */
    }

    /* Numerical safety fuse -- dx didn't change sufficiently */
    if (fabs(dxl-dx) < EPS_A || dx < EPS_A){
       /* Have to just exit, because we haven't decreased the gradient, and
        * so the BFGS update would be bad. Could also just return,
        * and have the calling function handle it. */
      printf("GLS: dx changes are too small for numerical precision\n");
      exit(1);
    }
  }

  if (k >= MAX_LINE_STEPS-1){
    printf("Line search failed to converge in %d iterations\n",MAX_LINE_STEPS);
    fflush(stdout);
    exit(1);
  }

  /* Return final left state
   * [right state is not necessarily better, even for convex functions] */
  *f_final = fl;
  vcopy(gl, g_final, ndv);
  return dxl;
}

/* Zoom function for regular line search
 * Returns step length to take, and the function value and gradients at that
 * chosen step length:
 *   -- f_final returned by pointer and g_final as pointer to array
 *
 *   This also modifies glo, ghi arrays passed in as pointers
 */
double Zoom(const int ndv, const double *x0, const double *sdir, double dxlo, double dxhi, const double f0, const double *g0, double flo, double *glo,  double fhi, double *ghi, double *f_final, double *g_final, double (*FGeval)(const int, const double *, double *))
{
  int i,k;
  double dx, dxmin, dxmax;
  double x[ndv];
  double f;
  double g[ndv];
  /* Gradients projected into search direction: */
  double g_sdir, g0_sdir;

  assert (dxlo >= 0 && dxhi >= 0); /* only going in positive sdir */

  g0_sdir = vdot(g0, sdir, ndv); /* Get initial gradient */

  for(k=0;k<MAX_ZOOM_STEPS;k++){

    /* Safety exit for numerical reasons */
    if (fabs(dxhi - dxlo) < EPS_A){
      printf("Exited zoom because bracket became too tight\n");
      *f_final = flo; /* Save best so far as final */
      vcopy(glo, g_final, ndv);
      return dxlo;
    }

    /* Determine bounds on next delta-x in the uncertainty interval
     * Note lo/hi indicate the objective function value, not the x-value,
     * whereas we need to clamp to a lo-hi x-range.
     * Also safeguard -- don't let the dx be too close to either side,
     * otherwise polynomial interpolation can become interminable for
     * highly nonlinear functions. */
    if (dxlo < dxhi) {
      dxmin = dxlo + POLY_SAFEGUARD_LO*(dxhi-dxlo);
      dxmax = dxlo + POLY_SAFEGUARD_HI*(dxhi-dxlo);
    }else{
      dxmin = dxlo + POLY_SAFEGUARD_HI*(dxhi-dxlo);
      dxmax = dxlo + POLY_SAFEGUARD_LO*(dxhi-dxlo);;
    }

    printf("   Zoom %d: (%.8g,%.8g) safeguard (%.8g,%.8g)\n",k,dxlo,dxhi,dxmin,dxmax);fflush(stdout);

    double glo_sdir = vdot(glo, sdir, ndv); /* Grad at lo */
    double ghi_sdir = vdot(ghi, sdir, ndv); /* Grad at hi */

    /* Pick a new dx between low and high bounds */
    double dxf2g1lo, dxg2;
    dxf2g1lo = quad_f2g1(dxlo, flo, dxhi, fhi, dxlo, glo_sdir, dxmin, dxmax);
    dxg2     = quad_g2(dxlo, glo_sdir, dxhi, ghi_sdir, dxmin, dxmax);

    // double dxf2g1hi,  dxf3;
    // dxf3     = quad_f3(0, f0, dxlo, flo, dxhi, fhi, dxmin, dxmax);
    // dxf2g1hi = quad_f2g1(dxlo, flo, dxhi, fhi, dxhi, ghi_sdir, dxmin, dxmax);

    // printf("quad_f3      : %.5g\n",dxf3);
    printf("quad_f2g1[lo]: %.5g\n",dxf2g1lo);
    // printf("quad_f2g1[hi]: %.5g\n",dxf2g1hi);
    printf("quad_g2      : %.5g\n",dxg2);

#if CONVEX_NOISY
    dx = dxg2; // Should be noise-tolerant
#else
    dx = dxf2g1lo; // Gsearch uses this version
#endif
    // dx = dxf3; // This will often fall back on bisection
    // dx = dxf2g1hi;  // I don't think this is appropriate

    assert (dx <= dxmax && dx >= dxmin);

    /* If predicted minimum is at dxlo, we're done */

    for (i=0;i<ndv;i++)
      x[i] = x0[i] + dx*sdir[i]; /* Should check delta for numerics */

    /* Evaluate function at new test location
     * However, if we were smarter than bisecting, then we could use the
     * gradient also, no matter what */
    f = FGeval(ndv, x, g);
    g_sdir = vdot(g,sdir,ndv); /* Project new gradient into search direction */
    nfev++;

    /* Decide next delta x, before updating uncertainty interval */

    /* Check various conditions to determine how to update interval */

    /* Now update uncertainty interval (or break if Wolfe satisfied) */

#if (! CONVEX_NOISY)
    int f_increase = (f >= flo); /* relative to current best */
    int sufficient_decrease = (f < f0 + MU_1*dx*g0_sdir); /* rel to x0 */
    /* We can't rely on function changes in the noisy regime */
    if (!sufficient_decrease || f_increase){
      printf("--Hi [%e] replaced by dx [%e]\n",dxhi,dx);
      dxhi = dx;
      fhi = f;
      vcopy(g, ghi, ndv);
      continue;
    }
#endif

    /* If it also satisfies the Wolfe curvature condition, we're done */
#if (STRONGWOLFE)
    int strong_wolfe_curv_satisfied = (fabs(g_sdir) <= -MU_2*g0_sdir);
    if (strong_wolfe_curv_satisfied)
#else
    int weak_wolfe_curv_satisfied   = (    -g_sdir  <= -MU_2*g0_sdir);
    if (weak_wolfe_curv_satisfied)
#endif
    {
      *f_final = f;
      vcopy(g, g_final, ndv);
      return dx;
    }

    if ( g_sdir*(dxhi-dxlo) >= 0){
      printf("--Hi [%e] replaced by lo [%e]\n",dxhi,dxlo);
      dxhi = dxlo;
      fhi = flo;
      vcopy(glo, ghi, ndv);
    }
    printf("--lo [%e] replaced by dx [%e]\n",dxlo,dx);

    dxlo = dx;
    flo = f;
    vcopy(g, glo, ndv);
  };
  /* END Zoom loop */

  printf("Maxed out zoom steps\n");

  *f_final = f;
  vcopy(g, g_final, ndv);
  return dx;
}

/* Line search
 *
 * Returns step length to take, and the function value and gradients at that
 * chosen step length:
 *   -- f_final returned by pointer and g_final as pointer to array */
double LineSearch(const int ndv, const double *x0, const double *lb, const double *ub, const double *sdir, const double init_dx, const double f0, const double *g0, double *f_final, double *g_final, double (*FGeval)(const int, const double *, double *))
{
  int i,k;
  double f, f_prev;
  double g_sdir;
  double dx, dx_prev;
  int phase_one = 1; /* Start with exponential growth until approaching max */

  double x[ndv];
  double g[ndv];
  double g_prev[ndv];

  /* Compute max step size based on min/max variable bounds */
  double bnd = 1.e100;
  for (i=0;i<ndv;i++){
    if      (sdir[i]>0)
      bnd = MIN(bnd,(ub[i]-x0[i])/sdir[i]); /* Only upper bound can be active */
    else if (sdir[i]<0)
      bnd = MIN(bnd,(lb[i]-x0[i])/sdir[i]); /* Only lower bound can be active */
  }
  const double dx_max = bnd;
  if (dx_max <= EPS_A){
    printf("Too close to bounds\n");
    return 0;
  }

  const double g0_sdir = vdot(g0, sdir, ndv);/*Projected into NORMALIZED sdir*/
  /* Gradient must be negative in search direction */
  assert (g0_sdir < 0); /* Already tested this outside */

#if 0
  printf(" --> Line search: sdir = (%g,%g), g0_sdir = %g\n",sdir[0],sdir[1],g0_sdir);
#endif

  dx = init_dx; /* Usually this is just 1 */
  if (dx > dx_max){ /* Initial step violates bounds */
    phase_one = 0;
    dx = 0.5*(EPS_A+dx_max); /* Start halfway to max */
  }

  /* Loop to find a step satisfying strong Wolfe conditions
   * Keep increasing the step until bracketed, then call Zoom().
   *
   * To reach dx_max in log time, double the stepsize each time (phase_one)
   * until within factor of 2 of dx_max, then approach it by bisection.
   **/
  dx_prev = 0.0;
  f = f0;
  vcopy(g0, g, ndv);

  *f_final = f0; /* Initialize these just in case we break early */
  vcopy(g0, g_final, ndv);

  for(k=0;k<=MAX_LINE_STEPS;k++){ /* k must start at 0 for zoom test */
    assert(dx >= EPS_A);
    assert(dx <= dx_max);

    /* Apply step */
    for (i=0;i<ndv;i++)
      x[i] = x0[i] + dx*sdir[i];

    f_prev = f;
    vcopy(g, g_prev, ndv);

    /* Compute functional and gradient */
    f = FGeval(ndv, x, g);
    nfev++;
    g_sdir = vdot(g,sdir,ndv);
    printf("  LS %d: dx = %g : f=%.6g, g_sdir=%.6g\n",k,dx,f,g_sdir);fflush(stdout);

    /* Check various conditions to determine how to update interval */
#if (STRONGWOLFE)
    int wolfe_curv_satisfied = (fabs(g_sdir) <= -MU_2*g0_sdir);
#else
    int wolfe_curv_satisfied = (    -g_sdir  <= -MU_2*g0_sdir);
#endif

    int sufficient_decrease = (f < f0 + MU_1*dx*g0_sdir); /* rel to x0 */
    int f_increase = (f > f_prev); /* relative to current best */

    /* If this step satisfies the Wolfe conditions and is lower than
     * the previous best one, we're done. */
    if (sufficient_decrease && wolfe_curv_satisfied && (k==0 || !f_increase))
      break;

    /* (i) Insufficient decrease in f, relative to f[0] and g[0]
       (ii) f increased from last trial step size
       (iii) Gradient has flipped signs to positive.
       Any of these implies that we have bracketed step sizes
       that satisfy the strong Wolfe conditions  --> Zoom() */

    int zoom = (!sufficient_decrease || (k>0 && f_increase) || (g_sdir >= 0));

    if (zoom){
      double dxlo, dxhi;
      double flo, fhi;
      double *glo, *ghi;
      if (!sufficient_decrease || (k>0 && f_increase)){ /* Cases (i), (ii) */
        dxlo = dx_prev;
        flo  = f_prev;
        glo  = g_prev;

        dxhi = dx;
        fhi  = f;
        ghi  = g;
      }else{  /* Case (iii) */
        dxlo = dx;
        flo  = f;
        glo  = g;

        dxhi = dx_prev;
        fhi  = f_prev;
        ghi  = g_prev;
      }
      dx = Zoom(ndv, x0, sdir, dxlo, dxhi, f0, g0, flo, glo, fhi, ghi, f_final, g_final, FGeval);
      return dx;
    }

    /* None of the conditions apply. We have neither found a strong Wolfe
     * step, nor have we bracketed one. Expand the range and continue */
    dx_prev = dx;

    if (phase_one){
      dx = dx_prev*FACTOR; /* Exponential expansion */
      if (dx > dx_max)/* New step violates bounds */
        phase_one = 0; /* Switch to exponential approach to dx_max */
    }

    if (!phase_one)
      dx = 0.5*(dx_prev + dx_max); /* Bisect to exponentially approach dx_max */

    /* Numerical safety fuse -- dx didn't change sufficiently */
    if (fabs(dx_prev-dx) < EPS_A || dx < EPS_A){
      dx = dx_prev; /* Revert to the one corresponding to f and g */
      break;
    }
  }

  if (k >= MAX_LINE_STEPS-1){
    printf("Line search failed to satisfy strong Wolfe conditions in %d iterations\n",MAX_LINE_STEPS);
  }

  *f_final = f;
  vcopy(g, g_final, ndv);

  return dx;
}

/* BFGS: Update approximate Hessian inverse
 * gk = current gradients
 * gkm1 = gradients at previous major iterate
 * sk = Step vector (dx)
 * Vk = Old Hessian inverse approximation (modified inplace)
 */
int UpdateHessianInverse(const int ndv, const double *gk, const double *gkm1, const double *sk, double *Vk)
{
  int i, j;
  double yk[ndv];
  double I[ndv*ndv];
  double M1[ndv*ndv];
  double M2[ndv*ndv];
  double M3[ndv*ndv];
  double Ak[ndv*ndv];
  double Bk[ndv*ndv];
  double skT_yk;

  /* Make identity matrix */
  for (i=0;i<ndv;i++)
    for (j=0;j<ndv;j++)
      I[ndv*i+j] = (i==j);

  /* 2) Compute  yk = gk - gkm1 (change in gradient during last step) */

  vadd(gk,gkm1,yk,ndv,-1);

  /* Compute denominator */
  skT_yk = vdot(sk,yk,ndv);

  if (skT_yk == 0){
    /* This happens if either the gradient does not change
       (2nd deriv == 0) or if the step size was 0. */
    printf("BFGS Hessian updated failed: denominator s*y=0.\n");
    // printf("yk =");
    // PrintVec(yk,ndv);
    // printf("sk =");
    // PrintVec(sk,ndv);
    return 1;
  }

  /* First term: Ak = I - sk*yk'/(sk'*yk) */
  vouter(sk,yk,M1,ndv);            /* M1 = sk*yk' */
  mmultconst(M1,1./skT_yk,M2,ndv); /* M2 = M1 / denom */
  madd(I,M2,Ak,ndv,-1);            /* Ak = I - M2 */

  /* Second term: Bk = I - (yk*sk')/(sk'*yk) */
  vouter(yk,sk,M1,ndv);            /* M1 = yk*sk' */
  mmultconst(M1,1./skT_yk,M2,ndv); /* M2 = M1/denom */
  madd(I,M2,Bk,ndv,-1);            /* Bk = I - M2 */

  /* Ak*Vk*Bk */
  mmult(Ak,Vk,M1,ndv); /* M1 = Ak*Vk  -- Vk is still at its old value */
  mmult(M1,Bk,M2,ndv); /* M2 = M1*Bk */

  /* Final term */
  vouter(sk,sk,M1,ndv); /* M1 = sk*sk' */
  mmultconst(M1,1./skT_yk,M3,ndv); /* M3 = M1/denom */

  /* Compute new Vk = M2 + M3 */
  madd(M3,M2,Vk,ndv,1); /* Vk = M2 + M3 */

#if 0
  printf("New grad: \n");
  PrintVec(g,ndv);
  printf("yk = \n");
  PrintVec(yk,ndv);
  printf("sk = \n");
  PrintVec(sk,ndv);
  printf("Ak = Bk':\n");
  PrintMat(Ak,ndv);
  // printf("Vk_prev:\n");
  // PrintMat(Vk_prev,ndv);
  printf("Ak*Vk_prev*Bk:\n");
  PrintMat(M2,ndv);
  printf("Last term:\n");
  PrintMat(M3,ndv);
  printf("Vk:\n");
  PrintMat(Vk,ndv);
#endif

  /* Verify that Inverse Hessian is symmetric */
  int fixed = 0;
  for (i=0;i<ndv;i++){
    for (j=i+1;j<ndv;j++){
      if (fabs(Vk[ndv*i+j] - Vk[ndv*j+i])/fabs(Vk[ndv*i+j]) > 1.e-8){
        printf("Error: Asymmetric Inv. Hessian:\n");
        printf("diff = %16.14f\n",Vk[ndv*i+j] - Vk[ndv*j+i]);
        PrintMat(Vk,ndv);
        return 1;
      }else if (Vk[ndv*i+j] != Vk[ndv*j+i]) {
          fixed++;
          /* Force exact symmetry to fix small numerical errors */
          Vk[ndv*i+j] = Vk[ndv*j+i];
      }
    }
  }
  if (fixed > 0){
    printf("Fixed %d tiny asymmetries in Inv. Hessian\n",fixed);
  }

  /* Verify that Inverse Hessian is positive definite
   * [simple test that is only exact for 2D] */
  if (Vk[0] <= 0 || Vk[ndv+1]*Vk[0]-Vk[1]*Vk[ndv] <= 0){
    printf("Error: Inv. Hessian not Positive definite:\n");
    PrintMat(Vk,ndv);
    return 1;
  }

  return 0;
}

/* Optimize using the BFGS algorithm
 *
 * On call:
 * x, lb, ub = initial values, lower bounds, upper bounds
 * H = Fully populated (NxN) initial Hessian
 * FGeval = Pointer to function evaluating both function and gradient
 *
 * On return:
 * x = Optimal values of x
 * H = Final Hessian
 *
 * Returns final function value
 */
double OptimizeBFGS(const int ndv, double *x, const double *lb, const double *ub, double *H, double (*FGeval)(const int, const double *, double *)){
  FILE *fp=NULL;

  int i,k;
  double dx, init_dx; /* Step lengths */
  double f, fnew; /* Function value */

  double g[ndv]; /* Gradients */
  double gnew[ndv];
  double mag_g; /* Magnitude of gradients */
  double g_sdir; /* Gradient projected into search direction */
  double df_exp;  /* Expected improvement in f */
  double sdir[ndv]; /* Search direction */
  double sk[ndv]; /* Step vector = sdir*dx */

  double Vk[ndv*ndv]; /* BFGS Hessian inverse approximation */

  /* Start optimization history file */
  fp = fopen("sdir.his", "w");
  fprintf(fp,"# CBFGS optimization history\n");
  fprintf(fp,"# Major nfev designXXX     f   ||g||    df_exp \n");fflush(fp);

  minv(H, Vk, ndv); /* Invert to form Vk */

  dx = 1.0;
  nfev = 0;
  printf("Evaluating functional at baseline design\n");fflush(stdout);
  /* Compute initial function value and gradient
   * Remainder of iterates are computed from inside the Linesearch function */
  f = FGeval(ndv, x, g);
  nfev++;
  mag_g = vmag(g, ndv);  /* Monitor gradient magnitude */

  printf("x0= ");
  PrintVec(x, ndv);
  printf("f0=%.6g, ||g0||=%.6e\n",f,mag_g);


  init_dx = STP0;
  /* If hessian not specified, scale first search direction to unity */
  int prescale = 1;
  for (i=0;i<ndv;i++){
    if (H[i*ndv+i] != 1.0) {
      prescale = 0;
      break;
    }
  }
  if (prescale){
    init_dx = STP0/mag_g;
    printf("Set initial dx to %e based on gradient magnitude\n",init_dx);
    fflush(stdout);
  }


  int stop = 0;
  for (k=0;k<MAX_MAJOR_ITER;k++){

    /* ---- Compute search direction s = -Vk*g ---- */
    mvmult(Vk,g,sdir,ndv);
    for (i=0;i<ndv;i++)
      sdir[i] *= -1; // Move in negative gradient direction

    /* Gradient projected into search direction must be negative */
    g_sdir = vdot(g,sdir,ndv);
    if (g_sdir >= 0){
      printf(" --> Invalid search direction -- Non-negative gradient: %e\n",g_sdir);
      exit(1);
    }

    df_exp = -0.5*g_sdir; /* Expected improvement 0.5*(g)(H^-1 * g) */

    /* Write out data */
    printf("Major %i (nfev=%i): f=%.6g, ||g||=%.3e, df_exp=%.3e\n",k,nfev,f,mag_g,df_exp);
    printf("        at x=");
    PrintVec(x, ndv);
    fflush(stdout);
    fprintf(fp,"%6d %6d %03d %.15e %.15e %.15e\n",k,nfev,nfev-1,f,mag_g,df_exp);
    fflush(fp);

    if (stop) break;

    /* Usually this is just 1, for the full BFGS step */
    if (k>0) init_dx = INIT_DX;

    /* Pass in the function and gradient at the line search start. On return,
     * fnew and gnew contain the value and gradients at the chosen dx. */
#if CONVEX_NOISY
    dx = GradLineSearch(ndv, x, lb, ub, sdir, init_dx, f, g, &fnew,gnew,FGeval);
#else
    dx = LineSearch(ndv, x, lb, ub, sdir, init_dx, f, g, &fnew, gnew, FGeval);
#endif
    if (dx <= 0){
      printf("Line search failed to make improvement\n");
      break; /* Break to avoid using this step for anything */
    }

    mag_g = vmag(gnew, ndv);  /* Monitor gradient magnitude */

    /* Compute step vector sk */
    for (i=0;i<ndv;i++)
      sk[i] = dx*sdir[i];

    /* Set x to correspond to where f and g were evaluated */
    for (i=0;i<ndv;i++)
      x[i] += sk[i];

    /* Convergence criterion */
    if (mag_g < EPS_G){
      printf("Terminating: Gradient magnitude below tolerance\n");
      stop = 1;
    }

    /* Numerical precision safety fuses */
    if (k > 0){
      /* 1) Change in functional is too small for numerical precision */

      /* This can be falsely triggered by noisy function evaluations, so
       * we may want to augment it with gradient information, or make it
       * over a window. May want to disable it. */
#if (! CONVEX_NOISY)
      if (fabs(fnew-f) < MIN(EPS_A, EPS_R*fabs(f))){
        printf("Exiting because of numerical precision in f\n");
        stop = 1;
      }
#endif

      /* 2) Change in *all* DV values are too small for numerical precision */
      int allsmall = 1;
      for (i=0;i<ndv;i++){
        if (fabs(sk[i]) > EPS_A){
          allsmall = 0;
          break;
        }
      }
      if (allsmall){
        printf("Exiting because of numerical precision in x\n");
        stop = 1;
      }
    }

    /* Update the Hessian approximation */
    int rc = UpdateHessianInverse(ndv, gnew, g, sk, Vk);
    if (rc){
      if (stop){
        printf("Already triggered stop. Final BFGS update failed. Hessian is for penultimate x.\n");
      }else{
        printf("Hessian update failed. Terminating...\n");
        stop = 1;
      }
    }else{
#if SAVE_ALL_HESSIANS
      /* Save updated Hessian in hessian.*.dat */
      char filename[32];
      FILE *hfp;
      int  i,j;

      minv(Vk, H, ndv); /* Store back in H, to return by pointer */

      sprintf(filename, "hessian.%04d.dat",k);
      hfp = fopen(filename,"w");
      for(i=0;i<ndv;i++){
        for(j=0;j<ndv;j++)
          fprintf(hfp, "%16.14e ",H[ndv*i+j]);
        fprintf(hfp, "\n");
      }
      fclose(hfp);
#endif
    }

    /* Handshake */
    f = fnew;
    vcopy(gnew, g, ndv);

  } /* End major BFGS loop */

  fclose(fp);

  printf("Major iterations: %i, Function evals = %i\n",k,nfev);
  for (i=0;i<ndv;i++)
    printf("x[%d]=%.12e\n",i,x[i]);
  printf("f=%.12e\n",f);
  printf("||g|| = %.4e\n",mag_g);

  /* Invert Vk to get final Hessian */
  minv(Vk, H, ndv); /* Store back in H, to return by pointer */



  return f;
}
