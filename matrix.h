/* Multiply a square matrix by a vector */
void mvmult(const double *M, const double *v, double *Mv, const int n);
/* Multiply two square matrices */
void mmult(const double *A, const double *B, double *AB, const int n);
/* Multiply square matrix by constant */
void mmultconst(const double *A, const double c, double *cA, const int n);
/* Add two square matrices */
void madd(const double *A, const double *B, double *AplusB, const int n, const double sign);
/* Add two vectors */
void vadd(const double *va, const double *vb, double *vaplusvb, const int n, const double sign);
/* Outer product of two vectors */
void vouter(const double *va, const double *vb, double *M, const int n);
/* Dot product of two vectors */
double vdot(const double *va, const double *vb, const int n);
/* Copy vector */
void vcopy(const double *source, double *target, const int n);
/* Copy square matrix */
void mcopy(const double *source, double *target, const int n);
/* Magnitude of vector */
double vmag(const double *v, const int n);
/* Normalize vector (in-place) */
void vnorm(double *v, const int n);
/* Invert nxn matrix */
void minv(const double *A, double *Ainv, const int n);
/* Invert 3x3 using determinant formula [For checking Cholesky, but Cholesky
 * is way better behaved when near 0 determinant]. */
int invert_3x3_determinant(const double *A, double *Ainv);
/* Print matrix */
void PrintMat(const double *M, const int n);
/* Print vector */
void PrintVec(const double *vec, const int n);
