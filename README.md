## SST
This is a fast Singular Spectrum Transformation (SST) implementation. SST can be used for anomaly detection over temporal sequence data.
## Compiling and Running

	# g++ -O2 sst.cpp -llapack -lblas
	# ./a.out

NOTE: LAPACK and BLAS library is required for compiling.

## How to use
Please, see `sst_test()` in sst.cpp and `CpuSstWorkspace` class in felixsst.h. `FelixSstWorkspace` implements a fast version SST. `NaiveSstWorkspace` implements a SVD-based SST.

## Lisence
MIT License.

## References
- [データストリーム処理における GPU タスク並列を用いたスケーラブルな異常検知](https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_action_common_download&item_id=82198&item_no=1&attribute_id=1&file_no=1)
- [Change-Point Detection using Krylov Subspace Learning](http://epubs.siam.org/doi/pdf/10.1137/1.9781611972771.54)
