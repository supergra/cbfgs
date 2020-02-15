#include <stdio.h>
#include <math.h>
#include <matrix.h>

#define MIN(A,B) (((A)>(B))?(B):(A))
#define MAX(A,B) (((A)>(B))?(A):(B))

/* Multiply a square matrix by a vector */
void mvmult(const double *M, const double *v, double *Mv, const int n)
{
  int i,k;
  for (i=0;i<n;i++){
    Mv[i]=0;
    for (k=0;k<n;k++)
      Mv[i] += M[n*i+k]*v[k];
  }
}

/* Multiply two square matrices */
void mmult(const double *A, const double *B, double *AB, const int n)
{
  int i,j,k;
  for (i=0;i<n;i++)
    for (j=0;j<n;j++){
      AB[n*i+j]=0;
      for (k=0;k<n;k++)
        AB[n*i+j]+=A[n*i+k]*B[n*k+j];
    }
}

/* Multiply square matrix by constant */
void mmultconst(const double *A, const double c, double *cA, const int n)
{
  int i;
  for (i=0;i<n*n;i++)
    cA[i] = A[i]*c;
}

/* Add two square matrices */
void madd(const double *A, const double *B, double *AplusB, const int n, const double sign)
{
  int i,j;
  for(i=0;i<n;i++)
    for(j=0;j<n;j++)
      AplusB[n*i+j]=A[n*i+j]+sign*B[n*i+j];
}

/* Add two vectors */
void vadd(const double *va, const double *vb, double *vaplusvb, const int n, const double sign)
{
  int i;
  for(i=0;i<n;i++)
    vaplusvb[i]=va[i]+sign*vb[i];
}

/* Outer product of two vectors */
void vouter(const double *va, const double *vb, double *M, const int n)
{
  int i,j;
  for(i=0;i<n;i++)
    for(j=0;j<n;j++)
      M[n*i+j] = va[i]*vb[j];   /* Eh? */
}

/* Dot product of two vectors */
double vdot(const double *va, const double *vb, const int n)
{
  int i;
  double dot = 0;
  for(i=0;i<n;i++)
    dot += va[i]*vb[i];
  return dot;
}

/* Copy vector */
void vcopy(const double *source, double *target, const int n){
  int i;
  for(i=0;i<n;i++)
    target[i] = source[i];
}

/* Copy square matrix */
void mcopy(const double *source, double *target, const int n){
  int i,j;
  for(i=0;i<n;i++)
    for(j=0;j<n;j++)
      target[i*n+j] = source[i*n+j];
}

/* Magnitude of vector */
double vmag(const double *v, const int n)
{
  int i;
  double mag=0;
  for(i=0;i<n;i++)
    mag += v[i]*v[i];
  return sqrt(mag);
}

/* Normalize vector (in-place) */
void vnorm(double *v, const int n)
{
  int i;
  double mag = vmag(v, n);
  if (mag == 0){
    printf("Can't normalize zero vector.\n");
    return;
  }
  for(i=0;i<n;i++)
    v[i] /= mag;
}

/* Invert nxn matrix */
/* http://programming-technique.blogspot.com/2011/09/numerical-methods-inverse-of-nxn-matrix.html */
void minv(const double *A, double *Ainv, const int n){
  int i,j,k;
  double B[n*n];
  /* Copy A to scratch matrix B, since we need to modify it in place */
  mcopy(A,B,n);

  /* Start with Identity matrix */
  for(i = 0; i < n; i++)
    for(j = 0; j < n; j++)
      Ainv[n*i+j] = (i==j);

  /* Do stuff -- O(n^3) operations */
  double ratio;
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(i!=j){
        if (B[n*i+i] == 0){
          printf("You hit the dumb inverse function\n");
          exit(1);
        }
        ratio = B[n*j+i]/B[n*i+i]; /* Dumbass shit divide by zero */
        for(k = 0; k < n; k++){
          B[n*j+k]    -= ratio * B[n*i+k];
          Ainv[n*j+k] -= ratio * Ainv[n*i+k];
        }
      }
    }
  }

  /* Do more stuff */
  double a;
  for(i = 0; i < n; i++){
    a = B[n*i+i];
    for(j = 0; j < n; j++){
      B[n*i+j] /= a;
      Ainv[n*i+j] /= a;
    }
  }
}

/* Invert 3x3 using determinant formula [For checking Cholesky, but Cholesky
 * is way better behaved when near 0 determinant]. */
int invert_3x3_determinant(const double *A, double *Ainv){
  int i, j;

  double det = 0.0;
  for (i=0;i<3;i++){ /* Loop over diagonal start column */
    double tmp = 1.0;
    for (j=0;j<3;j++){ /* Product along this forward diagonal */
      tmp *= A[j*3+((i+j)%3)];
    }
    det += tmp;
    tmp = 1.0;
    for (j=0;j<3;j++){ /* Product along this backward diagonal */
      tmp *= A[j*3+((i-j+3)%3)];
    }
    det -= tmp;
  }

  /* Alternate code that is equivalent */
  // double comp_det = 0.0;
  // for(i=0;i<3;i++){
  //   comp_det += A[0*3+i]*A[1*3+((i+1)%3)]*A[2*3+((i+2)%3)];
  //   comp_det -= A[0*3+i]*A[1*3+((i+2)%3)]*A[2*3+((i+1)%3)];
  // }
  // printf("COMP_DET = %e\n",comp_det);

  if (det == 0){
    printf("Determinant = 0\n");
    return 1;
  }

  // printf("Determinant = %e\n",det);

  for (i=0;i<3;i++){ /* Row */
    for (j=0;j<3;j++){ /* Col */
      int ip   = (i+1)%3;
      int ipp  = (i+2)%3;
      int jp   = (j+1)%3;
      int jpp  = (j+2)%3;
      /* These seem to have been transposed... */
      Ainv[j*3+i] = ((A[ip*3+jp]*A[ipp*3+jpp]) - (A[ip*3+jpp]*A[ipp*3+jp]))/det;
    }
  }
  return 0;
}

/* Print matrix */
void PrintMat(const double *M, const int n)
{
  int j,k;
  for(j=0;j<n;j++){
    printf("   ");
    for(k=0;k<n;k++)
      printf("%16.14g ",M[n*j+k]);
    printf("\n");
  }
}

/* Print vector */
void PrintVec(const double *vec, const int n)
{
  int j;
  printf("   ");
  for(j=0;j<n;j++)
    printf("%17.16g ",vec[j]);
  printf("\n");
}
