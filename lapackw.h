/*
 * lapackw.h
 *
 *  Created on: 2011/01/09
 *      Author: ueno
 */

#ifndef LAPACKW_H_
#define LAPACKW_H_

#include "cblas.h"

// LAPACK Wrapper

extern "C" {

void sstegr_(char* jobz, char* range, int* n, float* d, float* e, float* vl, float* vu, int* il, int* iu,
		float* abstol, int* m, float* w, float* z, int* ldz, int* isuppz, float* work, int* lwork,
		int* iwork, int* liwork, int* info);

void dstegr_(char* jobz, char* range, int* n, double* d, double* e, double* vl, double* vu, int* il, int* iu,
		double* abstol, int* m, double* w, double* z, int* ldz, int* isuppz, double* work, int* lwork,
		int* iwork, int* liwork, int* info);

void sstev_(char* jobz, int* n, float* d, float* e, float* z, int* ldz, float* work, int* info);
void dstev_(char* jobz, int* n, double* d, double* e, double* z, int* ldz, double* work, int* info);

void sgesdd_(char* jobz, int* m, int* n, float* a, int* lda, float* s,
		float* u, int* ldu, float* vt, int* ldvt, float* work, int* lwork,
		int* iwork, int* info);

void dgesdd_(char* jobz, int* m, int* n, double* a, int* lda, double* s,
		double* u, int* ldu, double* vt, int* ldvt, double* work, int* lwork,
		int* iwork, int* info);

}

template<typename REAL>
int inline clapack_stegr(char jobz, char range, int n, REAL* d, REAL* e, REAL vl, REAL vu, int il, int iu,
		REAL abstol, int* m, REAL* w, REAL* z, int ldz, int* isuppz, REAL* work, int lwork,
		int* iwork, int liwork);

template<>
int inline clapack_stegr(char jobz, char range, int n, float* d, float* e, float vl, float vu, int il, int iu,
		float abstol, int* m, float* w, float* z, int ldz, int* isuppz, float* work, int lwork,
		int* iwork, int liwork)
{
	int info = 0;
	sstegr_(&jobz,&range,&n,d,e,&vl,&vu,&il,&iu,&abstol,m,w,z,&ldz,isuppz,work,&lwork,iwork,&liwork,&info);
	return info;
}

template<typename REAL>
int inline clapack_stev(char jobz, int n, REAL* d, REAL* e, REAL* z, int ldz, REAL* work);
template<> int inline clapack_stev(char jobz, int n, float* d, float* e, float* z, int ldz, float* work)
{
	int info = 0;
	sstev_(&jobz,&n,d,e,z,&ldz,work,&info);
	return info;
}
template<> int inline clapack_stev(char jobz, int n, double* d, double* e, double* z, int ldz, double* work)
{
	int info = 0;
	dstev_(&jobz,&n,d,e,z,&ldz,work,&info);
	return info;
}

template<>
int inline clapack_stegr(char jobz, char range, int n, double* d, double* e, double vl, double vu, int il, int iu,
		double abstol, int* m, double* w, double* z, int ldz, int* isuppz, double* work, int lwork,
		int* iwork, int liwork)
{
	int info = 0;
	dstegr_(&jobz,&range,&n,d,e,&vl,&vu,&il,&iu,&abstol,m,w,z,&ldz,isuppz,work,&lwork,iwork,&liwork,&info);
	return info;
}

// sizeof(iwork) == 8*min(m,n)
template<typename REAL>
int inline clapack_gesdd(char jobz, int m, int n, REAL* a, int lda, REAL* s,
		REAL* u, int ldu, REAL* vt, int ldvt, REAL* work, int lwork, int* iwork);
template<>
int inline clapack_gesdd(char jobz, int m, int n, float* a, int lda, float* s,
		float* u, int ldu, float* vt, int ldvt, float* work, int lwork, int* iwork)
{
	int info = 0;
	sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
	return info;
}
template<>
int inline clapack_gesdd(char jobz, int m, int n, double* a, int lda, double* s,
		double* u, int ldu, double* vt, int ldvt, double* work, int lwork, int* iwork)
{
	int info = 0;
	dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
	return info;
}

// BLAS Wrapper

extern "C" {

void sgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const float* alpha,
		const float* a, const int* lda, const float* b, const int* ldb, const float* beta, float* c, const int* ldc);
void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k, const double* alpha,
		const double* a, const int* lda, const double* b, const int* ldb, const double* beta, double* c, const int* ldc);

float snrm2_(const int* n, const float* x, const int* incx);
double dnrm2_(const int* n, const double* x, const int* incx);

float sdot_(const int* n, const float* sx, const int* incx, const float* sy, const int* incy);
double ddot_(const int* n, const double* sx, const int* incx, const double* sy, const int* incy);

float sscal_(const int* n, const float* alpha, float* x, const int* incx);
double dscal_(const int* n, const double* alpha, double* x, const int* incx);

void saxpy_(const int* n, const float* alpha, const float *x, const int* incx, float *y, const int* incy);
void daxpy_(const int* n, const double* alpha, const double *x, const int* incx, double *y, const int* incy);

void scopy_(const int* n, const float *x, const int* incx, float *y, const int* incy);
void dcopy_(const int* n, const double *x, const int* incx, double *y, const int* incy);

}

template<typename REAL>
void inline cblas_gemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, REAL alpha, const REAL *A,
                 int lda, const REAL *B, int ldb,
                 REAL beta, REAL *C, int ldc);
template<>
void inline cblas_gemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, float alpha, const float *A,
                 int lda, const float *B, int ldb,
                 float beta, float *C, int ldc)
{
	char transa, transb;
	if( Order == CblasColMajor ) {
		transa = (TransA == CblasNoTrans) ? 'N' : 'T';
		transb = (TransB == CblasNoTrans) ? 'N' : 'T';
	}
	else {
		transa = (TransB == CblasNoTrans) ? 'N' : 'T';
		transb = (TransA == CblasNoTrans) ? 'N' : 'T';
	}
	sgemm_(&transa,&transb,&M,&N,&K,&alpha,A,&lda,B,&ldb,&beta,C,&ldc);
}
template<>
void inline cblas_gemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, double alpha, const double *A,
                 int lda, const double *B, int ldb,
                 double beta, double *C, int ldc)
{
	char transa, transb;
	if( Order == CblasColMajor ) {
		transa = (TransA == CblasNoTrans) ? 'N' : 'T';
		transb = (TransB == CblasNoTrans) ? 'N' : 'T';
	}
	else {
		transa = (TransB == CblasNoTrans) ? 'N' : 'T';
		transb = (TransA == CblasNoTrans) ? 'N' : 'T';
	}
	dgemm_(&transa,&transb,&M,&N,&K,&alpha,A,&lda,B,&ldb,&beta,C,&ldc);
}

template<typename REAL> REAL  inline cblas_nrm2(int N, const REAL *X, int incX);
template<> float inline cblas_nrm2(int N, const float *X, int incX)
{
	return snrm2_(&N, X, &incX);
}
template<> double inline cblas_nrm2(int N, const double *X, int incX)
{
	return dnrm2_(&N, X, &incX);
}

template<typename REAL> REAL  inline cblas_dot(int N, const REAL  *X, int incX,
        const REAL  *Y, int incY);
template<> float  inline cblas_dot(int N, const float  *X, int incX,
        const float  *Y, int incY)
{
	return sdot_(&N,X,&incX,Y,&incY);
}
template<> double  inline cblas_dot(int N, const double  *X, int incX,
        const double  *Y, int incY)
{
	return ddot_(&N,X,&incX,Y,&incY);
}

template<typename REAL> void inline cblas_scal(int N, REAL alpha, REAL *X, int incX);
template<> void inline cblas_scal(int N, float alpha, float *X, int incX)
{
	sscal_(&N,&alpha,X,&incX);
}
template<> void inline cblas_scal(int N, double alpha, double *X, int incX)
{
	dscal_(&N,&alpha,X,&incX);
}

template<typename REAL>
void inline cblas_axpy(int N, REAL alpha, const REAL *X,
                 int incX, REAL *Y, int incY);
template<> void inline cblas_axpy(int N, float alpha, const float *X,
                 int incX, float *Y, int incY)
{
	saxpy_(&N,&alpha,X,&incX,Y,&incY);
}
template<> void inline cblas_axpy(int N, double alpha, const double *X,
                 int incX, double *Y, int incY)
{
	daxpy_(&N,&alpha,X,&incX,Y,&incY);
}

template<typename REAL>
void inline cblas_copy(int N, const REAL *X, int incX, REAL *Y, int incY);
template<> void inline cblas_copy(int N, const float *X, int incX, float *Y, int incY)
{
	scopy_(&N,X,&incX,Y,&incY);
}
template<> void inline cblas_copy(int N, const double *X, int incX, double *Y, int incY)
{
	dcopy_(&N,X,&incX,Y,&incY);
}

#endif /* LAPACKW_H_ */
