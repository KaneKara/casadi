cmake_minimum_required(VERSION 2.8.6)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../blasfeo/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dummy_include)
casadi_external_library(casadi_hpmpc_X64_AVX_GENERIC src/auxiliary/d_aux_lib4.c
src/auxiliary/d_aux_extern_depend_lib4.c
src/auxiliary/i_aux.c
src/kernel/avx/kernel_dgemm_avx_lib4.c
src/kernel/avx/kernel_dtrmm_avx_lib4.c
src/kernel/avx/kernel_dtrsm_avx_lib4.c
src/kernel/avx/kernel_dsyrk_avx_lib4.c
src/kernel/avx/kernel_dpotrf_avx_lib4.c
src/kernel/avx/kernel_dgemv_avx_lib4.c
src/kernel/avx/kernel_dtrmv_avx_lib4.c
src/kernel/avx/kernel_dtrsv_avx_lib4.c
src/kernel/avx/kernel_dsymv_avx_lib4.c
src/kernel/avx/kernel_dtran_avx_lib4.c
src/kernel/avx/kernel_dttmm_avx_lib4.c
src/kernel/avx/kernel_dtrinv_avx_lib4.c
src/kernel/avx/kernel_dcopy_avx_lib4.c
src/kernel/avx/kernel_dgetrf_avx_lib4.c
src/blas/blas_d_lib4.c
src/lqcp_solvers/d_back_ric_rec.c
src/lqcp_solvers/d_for_schur_rec.c
src/lqcp_solvers/d_res.c
src/lqcp_solvers/d_part_cond.c
src/mpc_solvers/d_ip2_hard.c
src/mpc_solvers/d_res_ip_hard.c
src/mpc_solvers/d_ip2_res_hard.c
src/mpc_solvers/avx/d_aux_ip_hard_lib4.c
src/mpc_solvers/avx/d_res_ip_res_hard.c
src/interfaces/c/c_interface_work_space.c
src/interfaces/c/c_order_interface.c
src/interfaces/c/fortran_order_interface.c)
SET_TARGET_PROPERTIES(casadi_hpmpc_X64_AVX_GENERIC PROPERTIES COMPILE_FLAGS "-DLA_BLASFEO -DTARGET_GENERIC -O2 -fPIC -DBLASFEO -m64 -mavx -DTARGET_X64_AVX")

casadi_external_library(casadi_hpmpc_X64_AVX2_X64_INTEL_HASWELL src/auxiliary/d_aux_lib4.c
src/auxiliary/d_aux_extern_depend_lib4.c
src/auxiliary/i_aux.c
src/kernel/avx2/kernel_dgemm_avx2_lib4.c
src/kernel/avx2/kernel_dtrmm_avx2_lib4.c
src/kernel/avx2/kernel_dtrsm_avx2_lib4.c
src/kernel/avx2/kernel_dsyrk_avx2_lib4.c
src/kernel/avx2/kernel_dpotrf_avx2_lib4.c
src/kernel/avx2/kernel_dgemv_avx2_lib4.c
src/kernel/avx2/kernel_dtrmv_avx2_lib4.c
src/kernel/avx2/kernel_dtrsv_avx2_lib4.c
src/kernel/avx2/kernel_dsymv_avx2_lib4.c
src/kernel/avx2/kernel_dtran_avx2_lib4.c
src/kernel/avx2/kernel_dttmm_avx2_lib4.c
src/kernel/avx2/kernel_dtrinv_avx2_lib4.c
src/kernel/avx2/kernel_dgetrf_avx2_lib4.c
src/kernel/avx/kernel_dcopy_avx_lib4.c
src/blas/blas_d_lib4.c
src/lqcp_solvers/d_back_ric_rec.c
src/lqcp_solvers/d_for_schur_rec.c
src/lqcp_solvers/d_res.c
src/lqcp_solvers/d_part_cond.c
src/mpc_solvers/d_ip2_hard.c
src/mpc_solvers/d_res_ip_hard.c
src/mpc_solvers/d_ip2_res_hard.c
src/mpc_solvers/avx/d_aux_ip_hard_lib4.c
src/mpc_solvers/avx/d_res_ip_res_hard.c
src/interfaces/c/c_interface_work_space.c
src/interfaces/c/c_order_interface.c
src/interfaces/c/fortran_order_interface.c)
SET_TARGET_PROPERTIES(casadi_hpmpc_X64_AVX2_X64_INTEL_HASWELL PROPERTIES COMPILE_FLAGS "-DLA_BLASFEO -DTARGET_X64_INTEL_HASWELL -O2 -fPIC -DBLASFEO -m64 -mavx2 -mfma -DTARGET_X64_AVX2")

casadi_external_library(casadi_hpmpc_X64_AVX_X64_INTEL_SANDY_BRIDGE src/auxiliary/d_aux_lib4.c
src/auxiliary/d_aux_extern_depend_lib4.c
src/auxiliary/i_aux.c
src/kernel/avx/kernel_dgemm_avx_lib4.c
src/kernel/avx/kernel_dtrmm_avx_lib4.c
src/kernel/avx/kernel_dtrsm_avx_lib4.c
src/kernel/avx/kernel_dsyrk_avx_lib4.c
src/kernel/avx/kernel_dpotrf_avx_lib4.c
src/kernel/avx/kernel_dgemv_avx_lib4.c
src/kernel/avx/kernel_dtrmv_avx_lib4.c
src/kernel/avx/kernel_dtrsv_avx_lib4.c
src/kernel/avx/kernel_dsymv_avx_lib4.c
src/kernel/avx/kernel_dtran_avx_lib4.c
src/kernel/avx/kernel_dttmm_avx_lib4.c
src/kernel/avx/kernel_dtrinv_avx_lib4.c
src/kernel/avx/kernel_dcopy_avx_lib4.c
src/kernel/avx/kernel_dgetrf_avx_lib4.c
src/blas/blas_d_lib4.c
src/lqcp_solvers/d_back_ric_rec.c
src/lqcp_solvers/d_for_schur_rec.c
src/lqcp_solvers/d_res.c
src/lqcp_solvers/d_part_cond.c
src/mpc_solvers/d_ip2_hard.c
src/mpc_solvers/d_res_ip_hard.c
src/mpc_solvers/d_ip2_res_hard.c
src/mpc_solvers/avx/d_aux_ip_hard_lib4.c
src/mpc_solvers/avx/d_res_ip_res_hard.c
src/interfaces/c/c_interface_work_space.c
src/interfaces/c/c_order_interface.c
src/interfaces/c/fortran_order_interface.c)
SET_TARGET_PROPERTIES(casadi_hpmpc_X64_AVX_X64_INTEL_SANDY_BRIDGE PROPERTIES COMPILE_FLAGS "-DLA_BLASFEO -DTARGET_X64_INTEL_SANDY_BRIDGE -O2 -fPIC -DBLASFEO -m64 -mavx -DTARGET_X64_AVX")

casadi_external_library(casadi_hpmpc_X64_SSE3_X64_INTEL_CORE src/auxiliary/d_aux_lib4.c
src/auxiliary/d_aux_extern_depend_lib4.c
src/auxiliary/i_aux.c
src/kernel/sse3/kernel_dgemm_sse3_lib4.c
src/kernel/sse3/kernel_dtrmm_sse3_lib4.c
src/kernel/sse3/kernel_dtrsm_sse3_lib4.c
src/kernel/sse3/kernel_dsyrk_sse3_lib4.c
src/kernel/sse3/kernel_dpotrf_sse3_lib4.c
src/kernel/c99/kernel_dgemv_c99_lib4.c
src/kernel/c99/kernel_dtrmv_c99_lib4.c
src/kernel/c99/kernel_dtrsv_c99_lib4.c
src/kernel/c99/kernel_dsymv_c99_lib4.c
src/kernel/c99/kernel_dtran_c99_lib4.c
src/kernel/c99/kernel_dttmm_c99_lib4.c
src/kernel/c99/kernel_dtrinv_c99_lib4.c
src/kernel/c99/kernel_dcopy_c99_lib4.c
src/kernel/c99/kernel_dgetrf_c99_lib4.c
src/blas/blas_d_lib4.c
src/lqcp_solvers/d_back_ric_rec.c
src/lqcp_solvers/d_for_schur_rec.c
src/lqcp_solvers/d_res.c
src/lqcp_solvers/d_part_cond.c
src/mpc_solvers/d_ip2_hard.c
src/mpc_solvers/d_res_ip_hard.c
src/mpc_solvers/d_ip2_res_hard.c
src/mpc_solvers/c99/d_aux_ip_hard_lib4.c
src/mpc_solvers/c99/d_res_ip_res_hard.c
src/interfaces/c/c_interface_work_space.c
src/interfaces/c/c_order_interface.c
src/interfaces/c/fortran_order_interface.c)
SET_TARGET_PROPERTIES(casadi_hpmpc_X64_SSE3_X64_INTEL_CORE PROPERTIES COMPILE_FLAGS "-DLA_BLASFEO -DTARGET_X64_INTEL_CORE -O2 -fPIC -DBLASFEO -m64 -msse3 -DTARGET_X64_SSE3")

casadi_external_library(casadi_hpmpc_X64_SSE3_X64_AMD_BULLDOZER src/auxiliary/d_aux_lib4.c
src/auxiliary/d_aux_extern_depend_lib4.c
src/auxiliary/i_aux.c
src/kernel/sse3/kernel_dgemm_sse3_lib4.c
src/kernel/sse3/kernel_dtrmm_sse3_lib4.c
src/kernel/sse3/kernel_dtrsm_sse3_lib4.c
src/kernel/sse3/kernel_dsyrk_sse3_lib4.c
src/kernel/sse3/kernel_dpotrf_sse3_lib4.c
src/kernel/c99/kernel_dgemv_c99_lib4.c
src/kernel/c99/kernel_dtrmv_c99_lib4.c
src/kernel/c99/kernel_dtrsv_c99_lib4.c
src/kernel/c99/kernel_dsymv_c99_lib4.c
src/kernel/c99/kernel_dtran_c99_lib4.c
src/kernel/c99/kernel_dttmm_c99_lib4.c
src/kernel/c99/kernel_dtrinv_c99_lib4.c
src/kernel/c99/kernel_dcopy_c99_lib4.c
src/kernel/c99/kernel_dgetrf_c99_lib4.c
src/blas/blas_d_lib4.c
src/lqcp_solvers/d_back_ric_rec.c
src/lqcp_solvers/d_for_schur_rec.c
src/lqcp_solvers/d_res.c
src/lqcp_solvers/d_part_cond.c
src/mpc_solvers/d_ip2_hard.c
src/mpc_solvers/d_res_ip_hard.c
src/mpc_solvers/d_ip2_res_hard.c
src/mpc_solvers/c99/d_aux_ip_hard_lib4.c
src/mpc_solvers/c99/d_res_ip_res_hard.c
src/interfaces/c/c_interface_work_space.c
src/interfaces/c/c_order_interface.c
src/interfaces/c/fortran_order_interface.c)
SET_TARGET_PROPERTIES(casadi_hpmpc_X64_SSE3_X64_AMD_BULLDOZER PROPERTIES COMPILE_FLAGS "-DLA_BLASFEO -DTARGET_X64_AMD_BULLDOZER -O2 -fPIC -DBLASFEO -m64 -msse3 -DTARGET_X64_SSE3")

casadi_external_library(casadi_hpmpc_C99_4X4_GENERIC src/auxiliary/d_aux_lib4.c
src/auxiliary/d_aux_extern_depend_lib4.c
src/auxiliary/i_aux.c
src/kernel/c99/kernel_dgemm_c99_lib4.c
src/kernel/c99/kernel_dtrmm_c99_lib4.c
src/kernel/c99/kernel_dtrsm_c99_lib4.c
src/kernel/c99/kernel_dsyrk_c99_lib4.c
src/kernel/c99/kernel_dpotrf_c99_lib4.c
src/kernel/c99/kernel_dgemv_c99_lib4.c
src/kernel/c99/kernel_dtrmv_c99_lib4.c
src/kernel/c99/kernel_dtrsv_c99_lib4.c
src/kernel/c99/kernel_dsymv_c99_lib4.c
src/kernel/c99/kernel_dtran_c99_lib4.c
src/kernel/c99/kernel_dttmm_c99_lib4.c
src/kernel/c99/kernel_dtrinv_c99_lib4.c
src/kernel/c99/kernel_dcopy_c99_lib4.c
src/kernel/c99/kernel_dgetrf_c99_lib4.c
src/blas/blas_d_lib4.c
src/lqcp_solvers/d_back_ric_rec.c
src/lqcp_solvers/d_for_schur_rec.c
src/lqcp_solvers/d_res.c
src/lqcp_solvers/d_part_cond.c
src/mpc_solvers/d_ip2_hard.c
src/mpc_solvers/d_res_ip_hard.c
src/mpc_solvers/d_ip2_res_hard.c
src/mpc_solvers/c99/d_aux_ip_hard_lib4.c
src/mpc_solvers/c99/d_res_ip_res_hard.c
src/interfaces/c/c_interface_work_space.c
src/interfaces/c/c_order_interface.c
src/interfaces/c/fortran_order_interface.c)
SET_TARGET_PROPERTIES(casadi_hpmpc_C99_4X4_GENERIC PROPERTIES COMPILE_FLAGS "-DLA_BLASFEO -DTARGET_GENERIC -O2 -fPIC -DBLASFEO -march=native -DTARGET_C99_4X4")

casadi_external_library(casadi_hpmpc_C99_4X4_PREFETCH_GENERIC src/auxiliary/d_aux_lib4.c
src/auxiliary/d_aux_extern_depend_lib4.c
src/auxiliary/i_aux.c
src/kernel/c99/kernel_dgemm_c99_prefetch_lib4.c
src/kernel/c99/kernel_dtrmm_c99_prefetch_lib4.c
src/kernel/c99/kernel_dtrsm_c99_prefetch_lib4.c
src/kernel/c99/kernel_dsyrk_c99_lib4.c
src/kernel/c99/kernel_dpotrf_c99_prefetch_lib4.c
src/kernel/c99/kernel_dgemv_c99_lib4.c
src/kernel/c99/kernel_dtrmv_c99_lib4.c
src/kernel/c99/kernel_dtrsv_c99_lib4.c
src/kernel/c99/kernel_dsymv_c99_lib4.c
src/kernel/c99/kernel_dtran_c99_lib4.c
src/kernel/c99/kernel_dttmm_c99_lib4.c
src/kernel/c99/kernel_dtrinv_c99_lib4.c
src/kernel/c99/kernel_dcopy_c99_lib4.c
src/kernel/c99/kernel_dgetrf_c99_lib4.c
src/blas/blas_d_lib4.c
src/lqcp_solvers/d_back_ric_rec.c
src/lqcp_solvers/d_for_schur_rec.c
src/lqcp_solvers/d_res.c
src/lqcp_solvers/d_part_cond.c
src/mpc_solvers/d_ip2_hard.c
src/mpc_solvers/d_res_ip_hard.c
src/mpc_solvers/d_ip2_res_hard.c
src/mpc_solvers/c99/d_aux_ip_hard_lib4.c
src/mpc_solvers/c99/d_res_ip_res_hard.c
src/interfaces/c/c_interface_work_space.c
src/interfaces/c/c_order_interface.c
src/interfaces/c/fortran_order_interface.c)
SET_TARGET_PROPERTIES(casadi_hpmpc_C99_4X4_PREFETCH_GENERIC PROPERTIES COMPILE_FLAGS "-DLA_BLASFEO -DTARGET_GENERIC -O2 -fPIC -DBLASFEO -march=native -DTARGET_C99_4X4_PREFETCH")

