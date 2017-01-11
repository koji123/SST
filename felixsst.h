/*
 * felixsst.h
 *
 *  Created on: 2011/02/01
 *      Author: ueno
 */

#ifndef FELIXSST_H_
#define FELIXSST_H_

#include <sys/time.h>
#include <math.h>
#include "lapackw.h"
#if __INTEL_COMPILER
#include <xmmintrin.h>
#endif

// "workspace" has working memory and SST algorithm and does not have state.
template <typename REAL>
class CpuSSTWorkspace
{
public:
  virtual ~CpuSSTWorkspace() { }

  // wnd : timeline data
  // w : window size
  // gap : gap between now and past window
  // r : # of eigen vector to be used
  // a : REAL array length w, to hold state
  // first : input true for the first call to the timeline
  // returns SST score
	virtual REAL computeScore(const REAL* wnd, int w, int gap, int r, REAL* a, bool first) = 0;
};

template <typename REAL>
class FelixSSTWorkspace : public CpuSSTWorkspace<REAL>
{
protected:
	REAL* work;
	int posw;
	int lwork;
	int _w;
	int _r;
	REAL* a0;
#if FELIX_SST_EVR
	int optWorkgr, optWorkigr;
#endif

	void getOptParam(int w, int r, int k)
	{
#if FELIX_SST_EVR
		REAL optWork;
		int optWorki;

		{
			REAL dum1,dum2,dum4,dum5;
			int dum3,dum6;
			if( clapack_stegr<REAL>('V','I',k,&dum1,&dum2,0,0,1,r,0,&dum3,&dum4,&dum5,k,&dum6,&optWork,-1,&optWorki,-1) != 0 ){
				throw "ERROR:clapack_sstegr";
			}
			optWorkgr = (int)optWork;
			optWorkigr = optWorki;
		}

		lwork = (2*w + (r+2)*k) + std::max(3*w, 3*k + optWorkgr + optWorkigr);
#else
		lwork = (2*w + (k+2)*k) + 3*w;
#endif
	}

	void setA0()
	{
		a0 = &work[posw];
		posw += _w;

		// set random normalized vector
		srand(2536);
		for(int i = 0; i < _w; i++){
			a0[i] = (float)(rand()/3000 + 100);
		}
		REAL nrm = cblas_nrm2<REAL>(_w, a0, 1);
		cblas_scal<REAL>(_w, 1.0/nrm, a0, 1);
	}

public:
  FelixSSTWorkspace()
		: work(NULL)
		, _w(0)
		, _r(0)
	{
	}

  virtual ~FelixSSTWorkspace()
	{
		if( work ) free(work);
		work = NULL;
		_w = _r = 0;
	}

	virtual REAL computeScore(const REAL* wnd, int w, int gap, int r, REAL* a, bool first)
	{
		int k = 2*r - (r&1);

		// allocate work memory
		if( _w != w || _r != r ){
			if( work ) free(work);
			_w = w; _r = r;
			getOptParam(w, r, k);
			posw = 0;
			work = (REAL*)malloc(lwork*sizeof(REAL));
			if( work == NULL ) throw "OutOfMemory";
			setA0();
		}

		const REAL *w1 = wnd;
		const REAL *w2 = &w1[2*w-1+gap];

		// required work size : w + (r+2)*k
		int oldposw = posw;
		REAL* u = &work[posw]; posw += w;
		REAL* alpha = &work[posw]; posw += k;
		REAL* beta = &work[posw]; posw += k;
#if FELIX_SST_EVR
		REAL* x = &work[posw]; posw += k*r;
#else
		REAL* x = &work[posw]; posw += k*k;
#endif

		if( first ) memcpy(a, a0, w*sizeof(REAL));

		computeMu(w, w2, u, a, a0);
		lanczosIteration(w, k, w1, u, alpha, beta);
		int m = computeEigenVectors(r, k, alpha, beta, x);
#if FELIX_SST_EVR
		REAL score = getScore(m, k, x);
#else
		REAL score = getScore(m, k, &x[(k-r)*k]);
#endif
		posw = oldposw;
		return score;
	}

protected:
#if FELIX_SST_PRODUCT_H_OPT
	void productH(int w, const REAL* wn, const REAL* x, REAL* tmp, REAL* y)
	{
		// Because H is a symmetric matrix, H == H'.
		// H*H'*x = H*H*x = H*(H*q)
		//  tmp = H*x
	// this code results an error
	//	cblas_sgemv(CblasColMajor, CblasNoTrans, w, w, 1.0f, wn, 1, x, 1, 0.0f, tmp, 1);
		memset(tmp, 0x00, w*sizeof(REAL));
	//	for(int i = 0; i < w; i++) cblas_axpy<REAL>(w, x[i], &wn[i], 1, tmp, 1);
		productHInner(w,wn,x,tmp);
		//  y = H*tmp (=H*H*x=H*H'*q)
	//	cblas_sgemv(CblasColMajor, CblasNoTrans, w, w, 1.0f, wn, 1, tmp, 1, 0.0f, y, 1);
		memset(y, 0x00, w*sizeof(REAL));
	//	for(int i = 0; i < w; i++) cblas_axpy<REAL>(w, tmp[i], &wn[i], 1, y, 1);
		productHInner(w,wn,tmp,y);
	}
#endif // #if FELIX_SST_PRODUCT_H_OPT

#if FELIX_SST_PRODUCT_H_OPT == 2

#ifdef __INTEL_COMPILER

#define v4sf __m128
//#define __M128_ZEROINIT _mm_setzero_ps()
#define __M128_ZEROINIT {{ 0.0f }}
#define _m128_loadups(p) _mm_loadu_ps(p)
#define _m128_mulps(op1,op2) _mm_mul_ps(op1,op2)
#define _m128_addps(op1,op2) _mm_add_ps(op1,op2)

#else // #ifdef __INTEL_COMPILER

	typedef float v4sf __attribute__ ((vector_size (16)));
#define __M128_ZEROINIT {{ 0.0f }}
#define _m128_loadups(p) __builtin_ia32_loadups(p)
#define _m128_mulps(op1,op2) ((op1)*(op2))
#define _m128_addps(op1,op2) ((op1)+(op2))

#endif // #ifdef __INTEL_COMPILER

	typedef union {
	        v4sf v;
	        float f[4];
	} f4vec;
	static void productHInner(int w, const REAL* a, const REAL* x, REAL* y)
	{
		int r = 0;
		for( ; r < (w&~3); r+=4){
			int i = 0;
			const float* ra = a+r;
			f4vec d0 = __M128_ZEROINIT, d1 = __M128_ZEROINIT, d2 = __M128_ZEROINIT, d3 = __M128_ZEROINIT;
			for( ; i < (w&~3); i+=4){
				v4sf xi = _m128_loadups(x+i);
				d0.v = _m128_addps(d0.v, _m128_mulps(xi, _m128_loadups(ra+i+0)));
				d1.v = _m128_addps(d1.v, _m128_mulps(xi, _m128_loadups(ra+i+1)));
				d2.v = _m128_addps(d2.v, _m128_mulps(xi, _m128_loadups(ra+i+2)));
				d3.v = _m128_addps(d3.v, _m128_mulps(xi, _m128_loadups(ra+i+3)));
			}
			float sum0 = d0.f[0] + d0.f[1] + d0.f[2] + d0.f[3];
			float sum1 = d1.f[0] + d1.f[1] + d1.f[2] + d1.f[3];
			float sum2 = d2.f[0] + d2.f[1] + d2.f[2] + d2.f[3];
			float sum3 = d3.f[0] + d3.f[1] + d3.f[2] + d3.f[3];
			for( ; i < w; i++){
				sum0 += x[i] * a[r+i+0];
				sum1 += x[i] * a[r+i+1];
				sum2 += x[i] * a[r+i+2];
				sum3 += x[i] * a[r+i+3];
			}
			y[r+0] = sum0;
			y[r+1] = sum1;
			y[r+2] = sum2;
			y[r+3] = sum3;
		}
		for( ; r < w; r++){
			int i = 0;
			const float* ra = a+r;
			f4vec d = __M128_ZEROINIT;
			for( ; i < (w&~3); i+=4){
				d.v = _m128_addps(d.v, _m128_mulps(_m128_loadups(x+i), _m128_loadups(ra+i+0)));
			}
			float sum = d.f[0] + d.f[1] + d.f[2] + d.f[3];
			for( ; i < w; i++){
				sum += x[i] * a[r+i+0];
			}
			y[r] = sum;
		}
	}
#elif FELIX_SST_PRODUCT_H_OPT
	static void productHInner8(int w, const REAL* a, const REAL* x, REAL* y)
	{
		REAL f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0, f6 = 0, f7 = 0;

		for(int i = 0; i < w; i++){
			REAL xi = x[i];
			f0 += a[i+0] * xi;
			f1 += a[i+1] * xi;
			f2 += a[i+2] * xi;
			f3 += a[i+3] * xi;
			f4 += a[i+4] * xi;
			f5 += a[i+5] * xi;
			f6 += a[i+6] * xi;
			f7 += a[i+7] * xi;
		}

		y[0] = f0;
		y[1] = f1;
		y[2] = f2;
		y[3] = f3;
		y[4] = f4;
		y[5] = f5;
		y[6] = f6;
		y[7] = f7;
	}
	static void productHInner4(int w, const REAL* a, const REAL* x, REAL* y)
	{
		REAL f0 = 0, f1 = 0, f2 = 0, f3 = 0;

		for(int i = 0; i < w; i++){
			REAL xi = x[i];
			f0 += a[i+0] * xi;
			f1 += a[i+1] * xi;
			f2 += a[i+2] * xi;
			f3 += a[i+3] * xi;
		}

		y[0] = f0;
		y[1] = f1;
		y[2] = f2;
		y[3] = f3;
	}
	static void productHInner2(int w, const REAL* a, const REAL* x, REAL* y)
	{
		REAL f0 = 0, f1 = 0;

		for(int i = 0; i < w; i++){
			REAL xi = x[i];
			f0 += a[i+0] * xi;
			f1 += a[i+1] * xi;
		}

		y[0] = f0;
		y[1] = f1;
	}
	static void productHInner1(int w, const REAL* a, const REAL* x, REAL* y)
	{
		REAL f0 = 0;

		for(int i = 0; i < w; i++){
			REAL xi = x[i];
			f0 += a[i+0] * xi;
		}

		y[0] = f0;
	}
	static void productHInner(int w, const REAL* a, const REAL* x, REAL* y)
	{
		int i = 0;
		for( ; i < (w&~7); i += 8){
			productHInner8(w,a+i,x,y+i);
		}
		if( w&4 ){
			productHInner4(w,a+i,x,y+i); i+= 4;
		}
		if( w&2 ){
			productHInner2(w,a+i,x,y+i); i+= 2;
		}
		if( w&1 ){
			productHInner1(w,a+i,x,y+i); i+= 1;
		}
	}
#else // #elif FELIX_SST_PRODUCT_H_OPT
	// y = H * H' * x
	// H is a matrix made from the window, wn.
	// parameters
	// w : [in] window size
	// wn : [in] window
	// x : [in] dim(w)
	// tmp : [in] dim(w) work space
	// y : [in] dim(w)
	void productH(int w, const REAL* wn, const REAL* x, REAL* tmp, REAL* y)
	{
		// Because H is a symmetric matrix, H == H'.
		// H*H'*x = H*H*x = H*(H*q)
		//  tmp = H*x
	// this code results an error
	//	cblas_sgemv(CblasColMajor, CblasNoTrans, w, w, 1.0f, wn, 1, x, 1, 0.0f, tmp, 1);
		memset(tmp, 0x00, w*sizeof(REAL));
		for(int i = 0; i < w; i++) cblas_axpy<REAL>(w, x[i], &wn[i], 1, tmp, 1);
		//  y = H*tmp (=H*H*x=H*H'*q)
	//	cblas_sgemv(CblasColMajor, CblasNoTrans, w, w, 1.0f, wn, 1, tmp, 1, 0.0f, y, 1);
		memset(y, 0x00, w*sizeof(REAL));
		for(int i = 0; i < w; i++) cblas_axpy<REAL>(w, tmp[i], &wn[i], 1, y, 1);
	}
#endif // #elif FELIX_SST_PRODUCT_H_OPT

	// w : [in] window size
	// w2 : [in] current time line
	// u : [out] mu, the max eigenvalue vector
	// a : [in/out] feedback vector. Please input previous result of a. At first, input random normalized vector.
	// work : the work space.
	// lwork : the length of work space. The enough size of lwork is w
	void computeMu(int w, const REAL* w2, REAL* u, REAL* a, const REAL* a0)
	{
		int oldposw = posw;

		REAL* vec = &work[posw]; posw += w;
		REAL* x = a;
		REAL* y = u;

		int maxitr = 100;
		REAL threshold = 1e-6;
		REAL epsiron = 0.1f; // I'm not sure how to determine this parameter. It may be related to Zha-Simon's online algorithm.
/*
		// make H2
		for(int i = 0; i < w; i++){
			cblas_scopy(w, w2+i, 1, &h2[i*w], 1);
		}
		// A = H2' * H2
		cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, w,w,w, 1.0f, h2, w, h2, w, 0.0f, A, w);
*/
		// power
		REAL lk = 0.0f;
		for(int i = 0; i < maxitr; i++){
			// y = Ax;
			productH(w, w2, x, vec, y);
		//	cblas_sgemv(CblasColMajor, CblasNoTrans, w, w, 1.0f, A, w, x, 1, 0.0f, y, 1);
			REAL lk1 = 0.0f, s = 0.0f, c = 0.0f;
			// s = y*y; c = y*x; lk1 = s/c
			s = cblas_dot<REAL>(w, y, 1, y, 1);
			c = cblas_dot<REAL>(w, y, 1, x, 1);
			lk1 = s/c;
			// normalize y ( y = y/sqrt(s) )
			cblas_scal<REAL>(w, 1.0/sqrt(s), y, 1);
			// x = y
			cblas_copy<REAL>(w, y, 1, x, 1);
			if(fabs(lk-lk1)<(threshold*lk)){
				// finish iteration
				break;
			}
			lk = lk1;
		}

		// a = u + e*a0
		cblas_axpy<REAL>(w, epsiron, a0, 1, a, 1);

		posw = oldposw;
	}

	// w: [in] windows size
	// k : [in] r/2 (r is integer)
	// w1 : [in] past time line
	// u : [in] the singular vector correspond with the max singular value.
	// a : [out] diagonal values of the compressed matrix
	// b : [out] bidiagonal values of the compressed matrix
	// work : work space
	// lwork : the length of work space. The enough size of lwork is 3*w
	void lanczosIteration(int w, int k, const REAL* w1, REAL* u, REAL* a, REAL* b)
	{
		int oldposw = posw;

		REAL* q = u; // q1 = r
		REAL* prod = &work[posw]; posw += w;
	//	REAL* r = &work[posw]; posw += w;
		REAL* p = &work[posw]; posw += w;
		REAL* vec = &work[posw]; posw += w;

		// p1 = 0
		memset(p, 0x00, w*sizeof(REAL));

		for(int i = 0; i < k; i++){ // s = i+1
			// prod = ro * q (=H*H'*q)
			productH(w, w1, q, vec, prod);
			/*
			memcpy(prod, q, w*sizeof(REAL));
			// Because H is symmetric matrix, H == H'.
			// H*H'*q = H*H*q = H*(H*q)
			//  vec = H*prod
			cblas_sgemv(CblasColMajor, CblasNoTrans, w, w, 1.0f, w1, 1, prod, 1, 0.0f, vec, 1);
			//  prod = H*vec (=H*H*prod=H*H*q=H*H'*q=ro*q)
			cblas_sgemv(CblasColMajor, CblasNoTrans, w, w, 1.0f, w1, 1, vec, 1, 0.0f, prod, 1);
*/
			// a(s) = q' * prod
			a[i] = cblas_dot<REAL>(w, q, 1, prod, 1);

			// r = prod - a(s)*q + p
			//  prod = prod - a(s)*q
			cblas_axpy<REAL>(w, -a[i], q, 1, prod, 1);
			//  prod += p
			cblas_axpy<REAL>(w, 1.0f, p, 1, prod, 1);
			// now prod is r.

			// b(s) = norm(r)
			b[i] = cblas_nrm2<REAL>(w, prod, 1);

			// When it is the last loop, below code is not required.

			// p = - b(s) * q
			memcpy(p, q, w*sizeof(REAL));
			cblas_scal<REAL>(w, -b[i], p, 1);

			// q = r / b(s)
			memcpy(q, prod, w*sizeof(REAL));
			cblas_scal<REAL>(w, 1.0f/b[i], q, 1);
		}

		posw = oldposw;
	}

	// lwork >= 3*k + optWorkgr + optWorkigr
	// return : min(the number of found eigenvalues, r)
	int computeEigenVectors(int r, int k, REAL* a, REAL* b, REAL* x)
	{
		int oldposw = posw;

#if FELIX_SST_EVR
		int m; // the total number of eigenvalues
		 // eigenvalues
		REAL* w = &work[posw]; posw += k;
		int* isuppz = (int*)&work[posw]; posw += 2*k;
		REAL* optWork = &work[posw]; posw += optWorkgr;
		int* optWorki = (int*)&work[posw]; posw += optWorkigr;

		if( clapack_stegr<REAL>('V','I',k,a,b,0,0,k-r+1,k,0,&m,w,x,k,isuppz,optWork,optWorkgr,optWorki,optWorkigr) != 0 ){
			posw = oldposw;
			throw "ERROR:clapack_sstegr";
		}

		posw = oldposw;
		return m < r ? m : r;
#else
		REAL* optWork = &work[posw]; posw += 2*k;

		if( clapack_stev<REAL>('V', k, a, b, x, k, optWork) != 0 ){
			posw = oldposw;
			throw "ERROR:clapack_stev";
		}
		posw = oldposw;
		return r;
#endif
	}

	REAL getScore(int m, int k, REAL* x)
	{
		return 1.0f - cblas_dot<REAL>(m, x, k, x, k);
	}
};

template <typename REAL>
class NaiveSSTWorkspace : public CpuSSTWorkspace<REAL>
{
protected:
	REAL* work;
	int posw;
	int lwork;
	int _w;
	int _r;
	int optWorkdd, optWorkidd;

	void getOptParam(int w, int r)
	{
		REAL optWork;
		{
			REAL dum1, dum2, dum3, dum4;
			int dum5;
			if( clapack_gesdd<REAL>('A',w,w,&dum1,w,&dum2,&dum3,w,&dum4,w,&optWork,-1,&dum5) != 0 ){
				throw "Error:clapack_sgesdd";
			}
		}

		optWorkdd = (int)optWork;
		optWorkidd = 8*w;

		lwork = 3*w*w + 2*w + optWorkdd + optWorkidd;
	}

public:
  NaiveSSTWorkspace()
		: work(NULL)
		, _w(0)
		, _r(0)
	{
		//
	}

  virtual ~NaiveSSTWorkspace()
	{
		if( work ) free(work);
		work = NULL;
		_w = _r = 0;
	}

	virtual REAL computeScore(const REAL* wnd, int w, int gap, int r, REAL* a, bool first)
	{
		// allocate work memory
		if( _w != w || _r != r ){
			if( work ) free(work);
			_w = w; _r = r;
			getOptParam(w, r);
			posw = 0;
			work = (REAL*)malloc(lwork*sizeof(REAL));
			if( work == NULL ) throw "OutOfMemory";
		}

		const REAL *w1 = wnd;
		const REAL *w2 = &w1[2*w-1+gap];

		// required work size : 3*w*w + 2*w + optWorkdd + optWorkidd
		int oldposw = posw;
		REAL* h2u = &work[posw]; posw += w;
		REAL* u = &work[posw]; posw += w*w;
		REAL* vt = &work[posw]; posw += w*w;
		REAL* h = &work[posw]; posw += w*w;
		REAL* s = &work[posw]; posw += w;
		REAL* ddwork = &work[posw]; posw += optWorkdd;
		int* ddworki = (int*)&work[posw]; posw += optWorkidd;

		for(int t = 0; t < w; t++){
			memcpy(&h[t*w], &w2[t], w*sizeof(REAL));
		}
		if( clapack_gesdd<REAL>('A',w,w,h,w,s,u,w,vt,w,ddwork,optWorkdd,ddworki) != 0 ){
			throw "Error:clapack_sgesdd";
		}
		// copy the first u;
		memcpy(h2u, u, sizeof(REAL)*w);

		for(int t = 0; t < w; t++){
			memcpy(&h[t*w], &w1[t], w*sizeof(REAL));
		}
		if( clapack_gesdd<REAL>('A',w,w,h,w,s,u,w,vt,w,ddwork,optWorkdd,ddworki) != 0 ){
			throw "Error:clapack_sgesdd";
		}

		cblas_gemm<REAL>(CblasColMajor, CblasTrans, CblasNoTrans, 1, r, w, 1.0, h2u, w, u, w, 0.0, s, 1);
		REAL score = 1 - cblas_dot<REAL>(r, s, 1, s, 1);

		posw = oldposw;
		return score;
	}
};

#endif /* FELIXSST_H_ */
