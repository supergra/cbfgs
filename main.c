#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "bfgs.h"
#include "matrix.h"

/* Evaluate function and gradient simultaneously -- this will be a listener */
double Rosenbrock(const int ndv, const double *x, double *g){
  assert(ndv == 2);
  double f = (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
  g[0] = -2*(1-x[0]) - 400*(x[1]-x[0]*x[0])*x[0];
  g[1] = 200*(x[1]-x[0]*x[0]);
  return f;
}

double Quadratic(const int ndv, const double *x, double *g){
  int i;
  double f = 0;
  for(i=0;i<ndv;i++) {
    f += x[i]*x[i];
    g[i] = 2*x[i];
  }
  return f;
}

double Quartic(const int ndv, const double *x, double *g){
  int i;
  double f = 0;
  for(i=0;i<ndv;i++) {
    f += x[i]*x[i]*x[i]*x[i];
    g[i] = 4*x[i]*x[i]*x[i];
  }
  return f;
}

double Yanai(const int ndv, const double *x, double *g){
  assert(ndv==1);
  double b1 = 0.002;
  double b2 = 0.001; /* Steep on left -- make b2 smaller */
  double gb1 = sqrt(1+b1*b1) - b1;
  double gb2 = sqrt(1+b2*b2) - b2;
  double f = gb1*sqrt((1-x[0])*(1-x[0]) + b2*b2) + gb2*sqrt(x[0]*x[0]+b1*b1);

  g[0] = gb1/sqrt((1-x[0])*(1-x[0]) + b2*b2)*2*(1-x[0])*(-1) + gb2/sqrt(x[0]*x[0]+b1*b1)*2*x[0];

  f *= 1000;
  g[0]*= 1000;
  return f;
}

// Digital noise function:
// (1) high frequency, random offsets
// (2) Zero derivative everywhere but at discontinuities
#define NBINS 100
double random_deltas[NBINS];

double DigitalNoise(const int ndv, const double *x, const double freq, const double amplitude)
{
  srand(1234567); /* Repeatability */
  int i;
  for(i=0;i<NBINS;i++){
    random_deltas[i] = ((double)rand()/(double)(RAND_MAX));
  }

  int j;
  double noise = 0;
  for (i=0;i<ndv;i++){
    j = (int) (x[i]*freq+10000) % NBINS; // offset shunts symmetry away
    noise += amplitude*random_deltas[j];
  }
  return noise;
}

double NoisyFunc(const int ndv, const double *x, double *g){
  double f = Rosenbrock(ndv, x, g);
  // double f = Quartic(ndv, x, g);
  /* Add in noise */
  double freq = 100;
  double amp = 1.;
  f += DigitalNoise(ndv, x, freq, amp);
  return f;
}


int main(void){
  int i,j,k;
  FILE *fp;
  void *FandG;

  printf("Starting CBFGS in standalone mode.\n");
  const int ndv = 2;

  double x[ndv];
  double lb[ndv];
  double ub[ndv];
  double H[ndv*ndv];

  /* Set initial Hessian approximation */
  for (i=0;i<ndv;i++){
    for (j=0;j<ndv;j++){
      H[ndv*i+j] = 1*(i==j);
    }
  }

  // FandG = Rosenbrock;
  FandG = NoisyFunc;
  x[0] = -1.2;
  x[1] = 1;
  for(i=0;i<ndv;i++){
    lb[i] = -10000.;
    ub[i] =  10000.;
  }
  // FandG = Yanai;
  // x[0] = 0.02;
  // lb[0] = 0;
  // ub[0] = 1;

  printf("Launching BFGS optimization\n");
  OptimizeBFGS(ndv, x, lb, ub, H, FandG);

  printf("Optimal x:\n");
  for (i=0;i<ndv;i++)
    printf("x%d = %.12e\n",i,x[i]);
  printf("Final Hessian (saved in hessian.dat):\n");
  PrintMat(H, ndv);

  /* Store final Hessian in hessian.dat */
  fp = fopen("hessian.dat","w");
  for(j=0;j<ndv;j++){
    for(k=0;k<ndv;k++)
      fprintf(fp, "%16.14e ",H[ndv*j+k]);
    fprintf(fp, "\n");
  }
  fclose(fp);

  exit(0);
}
