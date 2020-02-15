#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "matrix.h" // Clearly I don't understand includes...
#define MIN(A,B) (((A)>(B))?(B):(A))
#define MAX(A,B) (((A)>(B))?(A):(B))
#include "interp.h"

/* Given coefficients (a,b) for a quadratic polynomial ax^2+bx+c,
 * return the x-location of the minimum within the range [xlo, xhi]
 * Note: coefficient c does not impact the x-location
 */
double clamped_quad_min(const double a, const double b, const double xlo, const double xhi)
{
  assert (xlo < xhi);
  double xmin;
  if (a == 0){
    // Degenerate: No curvature -- return midpoint
    xmin = 0.5*(xlo+xhi);
  }else if (a < 0){
    /* Negative curvature -- minimum must be at one of the bounds */
    double flo = a*xlo*xlo + b*xlo;
    double fhi = a*xhi*xhi + b*xhi;
    xmin = (flo < fhi) ? xlo : xhi; /* Return smaller of the two */
  }else{
    /* Positive curvature -- minimum may be interior */
    xmin = -b/(2*a);
    xmin = MIN(xmin, xhi); /* If above upper bound, upper bound is optimal */
    xmin = MAX(xmin, xlo); /* If below lower bound, lower bound is optimal */
  }
  printf("   y=%.6g*x^2%+.6g*x   [min = %.6g]\n",a,b,xmin);
  return xmin;
}

/* Given coefficients (a,b,c) for a cubic polynomial ax^3+bx^2+cx+d,
 * return the x-location of the minimum within the range [xlo, xhi]
 * Note: coefficient d does not impact the x-location
 */
double clamped_cubic_min(const double a, const double b, const double c, const double xlo, const double xhi)
{
  assert (xlo < xhi);
  double xmin;
  if (a == 0){
    // Degenerate: Not cubic -- return quadratic minimizer
    xmin = clamped_quad_min(b, c, xlo, xhi);
  }else{
    /* Minima of cubic are at x= (-b(+/-)sqrt(b^2-3ac)/(3a) */

    double r = b*b-3*a*c; /* Compute argument of square root */

    /* Values at endpoints -- if the optimum is not interior, it will be one
       of these. */
    double flo = ((a*xlo + b)*xlo + c)*xlo;
    double fhi = ((a*xhi + b)*xhi + c)*xhi;
    xmin = (flo < fhi) ? xlo : xhi; /* Default if not interior minimum */

    if (r>0){
      /* There is an inflection -- the positive root (out of +/-) is a
      local minimum because the 2nd derivative is positive, so optimum is
      either there or at a boundary. Unfortunately, we can't really tell
      which of the three is best without computing all three and comparing. */
      double flocalmin = (-b+sqrt(r))/(3*a);
      xmin = MIN(xmin, flocalmin);
    }
  }
  printf("   y=%.6g*x^3%+.6g*x^2+%.6g*x   [min = %.6g]\n",a,b,c,xmin);
  return xmin;
}

/* Quadratic through three values */
double quad_f3(const double x0, const double f0, const double x1, const double f1, const double x2, const double f2, const double xlo, const double xhi)
{
  double A[9];
  double Ainv[9];
  double rhs[3];
  double coeffs[3];
  assert (x0 != x1 || x0 != x2);
  if (x0 == x1){
    return 0.5*(x0+x2); // Bisect in degenerate cases
  }else if (x1 == x2 || x0 == x2){
    return 0.5*(x0+x1);
  }

  A[0] = x0*x0;  A[1] = x0; A[2] = 1;
  A[3] = x1*x1;  A[4] = x1; A[5] = 1;
  A[6] = x2*x2;  A[7] = x2; A[8] = 1;
  rhs[0] = f0; rhs[1] = f1; rhs[2] = f2;

  // minv(A, Ainv, 3);
  int rc = invert_3x3_determinant(A, Ainv);
  if (rc) exit(1);

  mvmult(Ainv, rhs, coeffs, 3);
  return clamped_quad_min(coeffs[0], coeffs[1], xlo, xhi);
}

/* Quadratic through two values and one gradient */
double quad_f2g1(const double x0, const double f0, const double x1, const double f1, const double x2, const double g2, const double xlo, const double xhi)
{
  double A[9];
  double Ainv[9];
  double rhs[3];
  double coeffs[3];

  assert (x0 != x1);
  A[0] = x0*x0;  A[1] = x0; A[2] = 1;
  A[3] = x1*x1;  A[4] = x1; A[5] = 1;
  A[6] = 2*x2;   A[7] = 1;  A[8] = 0;
  rhs[0] = f0; rhs[1] = f1; rhs[2] = g2;

  // minv(A, Ainv, 3);
  int rc = invert_3x3_determinant(A, Ainv);
  if (rc) exit(1);

  // PrintMat(A,3);
  // PrintVec(rhs,3);
  // PrintMat(Ainv,3);

  mvmult(Ainv, rhs, coeffs, 3);
  return clamped_quad_min(coeffs[0], coeffs[1], xlo, xhi);
}

/* Quadratic through two gradients [vertical offset irrelevant] */
double quad_g2(const double x0, const double g0, const double x1, const double g1, const double xlo, const double xhi)
{
  assert (x0 != x1);
  const double a = (g0-g1)/(2*(x0-x1));
  const double b = g0 - 2*a*x0;
  if (a < 0){
    printf("You promised I'd only see convexity: a=%.4e...\n",a);
    exit(1);
  }
  return clamped_quad_min(a, b, xlo, xhi);
}

/* Cubic through three gradients [vertical offset irrelevant] */
double cubic_g3(const double x0, const double g0, const double x1, const double g1, const double x2, const double g2, const double xlo, const double xhi)
{
  double A[9];
  double Ainv[9];
  double rhs[3];
  double coeffs[3];

  assert (x0 != x1 || x0 != x2);
  if (x0 == x1){
    return quad_g2(x0,g0,x2,g2,xlo,xhi); // Quadratic in degenerate cases
  }else if (x1 == x2 || x0 == x2){
    return quad_g2(x0,g0,x1,g1,xlo,xhi);
  }

  A[0] = 3*x0*x0;  A[1] = 2*x0; A[2] = 1;
  A[3] = 3*x1*x1;  A[4] = 2*x1; A[5] = 1;
  A[6] = 3*x2*x2;  A[7] = 2*x2; A[8] = 1;
  rhs[0] = g0; rhs[1] = g1; rhs[2] = g2;

  int rc = invert_3x3_determinant(A, Ainv);
  if (rc) exit(1);
  mvmult(Ainv, rhs, coeffs, 3);

  return clamped_cubic_min(coeffs[0],coeffs[1],coeffs[2],xlo,xhi);
}
