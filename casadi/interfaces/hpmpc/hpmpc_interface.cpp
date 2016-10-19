/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            K.U. Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */


#include "hpmpc_interface.hpp"

using namespace std;
namespace casadi {


  extern "C"
  int CASADI_CONIC_HPMPC_EXPORT
  casadi_register_conic_hpmpc(Conic::Plugin* plugin) {
    plugin->creator = HpmpcInterface::creator;
    plugin->name = "hpmpc";
    plugin->doc = HpmpcInterface::meta_doc.c_str();
    plugin->version = 30;
    return 0;
  }

  extern "C"
  void CASADI_CONIC_HPMPC_EXPORT casadi_load_conic_hpmpc() {
    Conic::registerPlugin(casadi_register_conic_hpmpc);
  }

  HpmpcInterface::HpmpcInterface(const std::string& name,
                                     const std::map<std::string, Sparsity>& st)
    : Conic(name, st) {
  }

  HpmpcInterface::~HpmpcInterface() {
    clear_memory();
  }

  Options HpmpcInterface::options_
  = {{&Conic::options_},
     {{"N",
       {OT_INT,
        "OCP horizon"}},
      {"nx",
       {OT_INTVECTOR,
        "Number of states"}},
      {"nu",
       {OT_INTVECTOR,
        "Number of controls"}},
      {"ng",
       {OT_INTVECTOR,
        "Number of non-dynamic constraints"}},
      {"mu0",
       {OT_DOUBLE,
        "Max element in cost function as estimate of max multiplier"}},
      {"max_iter",
       {OT_INT,
        "Max number of iterations"}},
      {"tol",
       {OT_DOUBLE,
        "Tolerance in the duality measure"}},
      {"warm_start",
       {OT_BOOL,
        "Tolerance in the duality measure"}},
      {"target",
       {OT_STRING,
        "hpmpc target"}},
      {"blasfeo_target",
       {OT_STRING,
        "hpmpc target"}}}
  };

  void HpmpcInterface::init(const Dict& opts) {
    Conic::init(opts);

    // Default options
    mu0_ = 1;
    max_iter_ = 1000;
    tol_ = 1e-8;
    warm_start_ = false;
    target_ = "C99_4X4";
    blasfeo_target_ = "GENERIC";
    target_ = "X64_AVX";
    blasfeo_target_ = "X64_INTEL_SANDY_BRIDGE";

    // Read options
    for (auto&& op : opts) {
      if (op.first=="N") {
        N_ = op.second;
      } else if (op.first=="nx") {
        nxs_ = op.second;
      } else if (op.first=="nu") {
        nus_ = op.second;
      } else if (op.first=="ng") {
        ngs_ = op.second;
      } else if (op.first=="mu0") {
        mu0_ = op.second;
      } else if (op.first=="max_iter") {
        max_iter_ = op.second;
      } else if (op.first=="tol") {
        tol_ = op.second;
      } else if (op.first=="warm_start") {
        warm_start_ = op.second;
      } else if (op.first=="target") {
        target_ = static_cast<std::string>(op.second);
      } else if (op.first=="blasfeo_target") {
        blasfeo_target_ = static_cast<std::string>(op.second);
      }
    }


    // Load libraries HPMPC and BLASFEO, when applicable
    std::string searchpath;

#ifdef BLASFEO_DLOPEN
    DL_HANDLE_TYPE handle_blasfeo = load_library("casadi_blasfeo_" + blasfeo_target_, searchpath,
      true);
#endif

#ifdef HPMPC_DLOPEN

    std::string libname = "casadi_hpmpc_" + target_ + "_" + blasfeo_target_;
    DL_HANDLE_TYPE handle = load_library(libname, searchpath, true);

    std::string work_size_name = "hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes";

#ifdef _WIN32
    hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes =
      (Work_size)GetProcAddress(handle, TEXT(work_size_name.c_str()));
#else // _WIN32
    // Reset error
    dlerror();

    // Load creator
    hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes = (Work_size)dlsym(handle, work_size_name.c_str());
#endif // _WIN32

    casadi_assert_message(hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes!=0,
      "HPMPC interface: symbol \"" + work_size_name + "\" found in " + searchpath + ".");

    std::string ocp_solve_name = "fortran_order_d_ip_ocp_hard_tv";

#ifdef _WIN32
    fortran_order_d_ip_ocp_hard_tv =
      (Ocp_solve)GetProcAddress(handle, TEXT(ocp_solve_name.c_str()));
#else // _WIN32
    // Reset error
    dlerror();

    // Load creator
    fortran_order_d_ip_ocp_hard_tv = (Ocp_solve)dlsym(handle, ocp_solve_name.c_str());
#endif // _WIN32

    casadi_assert_message(fortran_order_d_ip_ocp_hard_tv!=0,
      "HPMPC interface: symbol \"" + ocp_solve_name + "\" found in " + searchpath + ".");
#endif

    A0sp_  = Sparsity::dense(nxs_[0], nxs_[0]);
    S0sp_  = Sparsity::dense(nus_[0], nus_[0]);
    C0sp_  = Sparsity::dense(ngs_[0], nxs_[0]);

    const std::vector<int>& nx = nxs_;
    const std::vector<int>& ng = ngs_;
    const std::vector<int>& nu = nus_;

    /* Disassemble A input into:
       B A   I
       D C
           B A  I
           D C
    */
    std::vector< Block > A_blocks, B_blocks, C_blocks, D_blocks, I_blocks;
    int offset_r = 0, offset_c = 0;
    for (int k=0;k<N_;++k) { // Loop over blocks
      A_blocks.push_back(std::make_tuple(offset_r,       offset_c+nu[k],      nx[k], nx[k]));
      B_blocks.push_back(std::make_tuple(offset_r,       offset_c,            nx[k], nu[k]));
      C_blocks.push_back(std::make_tuple(offset_r+nx[k], offset_c+nu[k],      ng[k], nx[k]));
      D_blocks.push_back(std::make_tuple(offset_r+nx[k], offset_c,            ng[k], nu[k]));

      offset_c+= nx[k]+nu[k];
      if (k+1<N_)
        I_blocks.push_back(std::make_tuple(offset_r, offset_c+nu[k+1], nx[k+1], nx[k+1]));
      else
        I_blocks.push_back(std::make_tuple(offset_r, offset_c,         nx[k+1], nx[k+1]));
      offset_r+= nx[k]+ng[k];
    }

    C_blocks.push_back(std::make_tuple(offset_r, offset_c,      ng[N_], nx[N_]));

    Asp_ = blocksparsity(na_, nx_, A_blocks);
    Bsp_ = blocksparsity(na_, nx_, B_blocks);
    Csp_ = blocksparsity(na_, nx_, C_blocks);
    Dsp_ = blocksparsity(na_, nx_, D_blocks);
    Isp_ = blocksparsity(na_, nx_, I_blocks, true);

    Sparsity total = Asp_ + Bsp_ + Csp_ + Dsp_ + Isp_;
    casadi_assert_message((sparsity_in(CONIC_A) + total).nnz() == total.nnz(),
      "HPMPC: specified structure of A does not correspond to what the interface can handle.");
    casadi_assert(total.nnz() == Asp_.nnz() + Bsp_.nnz() + Csp_.nnz() + Dsp_.nnz() + Isp_.nnz());

    /* Disassemble H input into:
       R S
       S'Q'
           R S
           S'Q'

       Multiply by 2
    */
    std::vector< Block > R_blocks, S_blocks, Q_blocks;
    int offset = 0;
    for (int k=0;k<N_;++k) { // Loop over blocks
      R_blocks.push_back(std::make_tuple(offset,       offset,       nu[k], nu[k]));
      S_blocks.push_back(std::make_tuple(offset,       offset+nu[k], nu[k], nx[k]));
      Q_blocks.push_back(std::make_tuple(offset+nu[k], offset+nu[k], nx[k], nx[k]));
      offset+= nx[k]+nu[k];
    }
    Q_blocks.push_back(std::make_tuple(offset,         offset,       nx[N_], nx[N_]));

    Rsp_ = blocksparsity(nx_, nx_, R_blocks);
    Ssp_ = blocksparsity(nx_, nx_, S_blocks);
    Qsp_ = blocksparsity(nx_, nx_, Q_blocks);

    total = Rsp_ + Ssp_ + Qsp_ + Ssp_.T();
    casadi_assert_message((sparsity_in(CONIC_H) + total).nnz() == total.nnz(),
      "HPMPC: specified structure of H does not correspond to what the interface can handle.");
    casadi_assert(total.nnz() == Rsp_.nnz() + 2*Ssp_.nnz() + Qsp_.nnz());

    /* Disassemble LBA/UBA input into:
       b
       lg/ug

       b
       lg/ug
    */
    offset = 0;
    std::vector< Block > b_blocks, lug_blocks;
    for (int k=0;k<N_;++k) {
      b_blocks.push_back(std::make_tuple(offset,         0, nx[k], 1));
      lug_blocks.push_back(std::make_tuple(offset+nx[k], 0, ng[k], 1));
      offset+= ng[k] + nx[k];
    }
    lug_blocks.push_back(std::make_tuple(offset, 0, ng[N_], 1));

    bsp_ = blocksparsity(na_, 1, b_blocks);
    lugsp_ = blocksparsity(na_, 1, lug_blocks);
    total = bsp_ + lugsp_;
    casadi_assert(total.nnz() == bsp_.nnz() + lugsp_.nnz());
    casadi_assert(total.nnz() == na_);

    /* Disassemble G/X0 input into:
       r/u
       q/x

       r/u
       q/x
    */
    offset = 0;
    std::vector< Block > u_blocks, x_blocks;
    for (int k=0;k<N_;++k) {
      u_blocks.push_back(std::make_tuple(offset,       0, nu[k], 1));
      x_blocks.push_back(std::make_tuple(offset+nu[k], 0, nx[k], 1));
      offset+= nx[k] + nu[k];
    }
    x_blocks.push_back(std::make_tuple(offset, 0, nx[N_], 1));

    usp_ = blocksparsity(nx_, 1, u_blocks);
    xsp_ = blocksparsity(nx_, 1, x_blocks);
    total = usp_ + xsp_;
    casadi_assert(total.nnz() == usp_.nnz() + xsp_.nnz());
    casadi_assert(total.nnz() == nx_);

  }


  void HpmpcInterface::init_memory(void* mem) const {
    auto m = static_cast<HpmpcMemory*>(mem);

    m->nx = nxs_; m->nx.insert(m->nx.begin(), 0);
    m->nu = nus_; m->nu.push_back(0);
    m->ng = ngs_;

    const int* nx = get_ptr(m->nx) + 1;
    const std::vector<int>& nu = m->nu;
    const std::vector<int>& ng = m->ng;
    const std::vector<int>& nb = m->nb;

    m->nb.resize(N_+1);
    int offset = 0;
    m->nb[0] = nu[0];
    for (int k=1;k<N_;++k) m->nb[k] = nx[k]+nu[k];
    m->nb[N_] = nx[N_];

    m->A.resize(Asp_.nnz());
    m->B.resize(Bsp_.nnz());
    m->C.resize(Csp_.nnz());
    m->D.resize(Dsp_.nnz());
    m->R.resize(Rsp_.nnz());
    m->I.resize(Isp_.nnz());
    m->S.resize(Ssp_.nnz());
    m->Q.resize(Qsp_.nnz());
    m->b.resize(bsp_.nnz());
    m->b2.resize(bsp_.nnz());
    m->x.resize(xsp_.nnz());m->q.resize(xsp_.nnz());
    m->u.resize(usp_.nnz());m->r.resize(usp_.nnz());
    m->lg.resize(lugsp_.nnz());
    m->ug.resize(lugsp_.nnz());

    //m->lb.reserve(nx_);m->ub.reserve(nx_);

    offset = nu[0]+nx[0];
    for (int k=1;k<N_;++k) offset+=nx[k]+nu[k];
    offset+=nx[N_];
    m->lb.resize(offset);m->lb.reserve(offset+1);
    m->ub.resize(offset);m->ub.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nx[k];
    m->pi.resize(offset);m->pi.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_+1;++k) offset+=m->nx[k]+nu[k];
    m->hidxb.resize(offset+10);m->hidxb.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=ng[k]+nx[k]+nu[k];
    offset+=ng[N_]+nx[N_];
    m->lam.resize(2*offset);m->lam.reserve(2*offset+1);

    // Allocate double* work vectors

    m->As.resize(N_);
    m->Qs.resize(N_+1);
    offset = 0;
    for (int k=0;k<N_;++k) {
      m->As[k] = get_ptr(m->A)+offset;
      m->Qs[k] = get_ptr(m->Q)+offset;
      offset+=nx[k]*nx[k];

    }
    m->Qs[N_] = get_ptr(m->Q)+offset;

    m->Bs.resize(N_);
    m->Ss.resize(N_);
    offset = 0;
    for (int k=0;k<N_;++k) {
      m->Bs[k] = get_ptr(m->B)+offset;
      m->Ss[k] = get_ptr(m->S)+offset;
      offset+=nx[k]*nu[k];
    }

    m->Cs.resize(N_+1);
    offset = 0;
    for (int k=0;k<N_+1;++k) {
      m->Cs[k] = get_ptr(m->C)+offset;
      offset+=ng[k]*nx[k];
    }
    m->Ds.resize(N_);
    offset = 0;
    for (int k=0;k<N_;++k) {
      m->Ds[k] = get_ptr(m->D)+offset;
      offset+=ng[k]*nu[k];
    }

    m->Rs.resize(N_);
    offset = 0;
    for (int k=0;k<N_;++k) {
      m->Rs[k] = get_ptr(m->R)+offset;
      offset+=nu[k]*nu[k];
    }

    m->qs.resize(N_+1);
    m->bs.resize(N_);
    m->xs.resize(N_+1);
    m->pis.resize(N_);
    offset = 0;
    for (int k=0;k<N_;++k) {
      m->qs[k] = get_ptr(m->q)+offset;
      m->bs[k] = get_ptr(m->b)+offset;
      m->xs[k] = get_ptr(m->x)+offset;
      m->pis[k] = get_ptr(m->pi)+offset;
      offset+=nx[k];
    }
    m->qs[N_] = get_ptr(m->q)+offset;
    m->xs[N_] = get_ptr(m->x)+offset;

    m->rs.resize(N_);
    m->us.resize(N_+1);
    offset = 0;
    for (int k=0;k<N_;++k) {
      m->rs[k] = get_ptr(m->r)+offset;
      m->us[k] = get_ptr(m->u)+offset;
      offset+=nu[k];
    }
    m->us[N_] = get_ptr(m->u)+offset;

    m->lbs.resize(N_+1);
    m->ubs.resize(N_+1);
    offset = 0;
    m->lbs[0] = get_ptr(m->lb);
    m->ubs[0] = get_ptr(m->ub);
    offset = nu[0]+nx[0];
    for (int k=1;k<N_;++k) {
      m->lbs[k] = get_ptr(m->lb)+offset;
      m->ubs[k] = get_ptr(m->ub)+offset;
      offset+=nu[k]+nx[k];
    }
    m->lbs[N_] = get_ptr(m->lb)+offset;
    m->ubs[N_] = get_ptr(m->ub)+offset;

    m->lgs.resize(N_+1);
    m->ugs.resize(N_+1);
    offset = 0;
    for (int k=0;k<N_+1;++k) {
      m->lgs[k] = get_ptr(m->lg)+offset;
      m->ugs[k] = get_ptr(m->ug)+offset;
      offset+=ng[k];
    }

    m->lams.resize(N_+1);
    offset = 0;
    for (int k=0;k<N_+1;++k) {
      m->lams[k] = get_ptr(m->lam)+offset;
      offset+=2*(ng[k]+nb[k]);
    }

    m->hidxbs.resize(N_+1);
    offset = 0;
    for (int k=0;k<N_+1;++k) {
      m->hidxbs[k] = get_ptr(m->hidxb)+offset;
      for (int i=0;i<m->nb[k];++i) m->hidxbs[k][i] = i;
      offset+=3;
    }

    // Allocate extra workspace as per HPMPC request
    int workspace_size = hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(
      N_, get_ptr(m->nx), get_ptr(m->nu), get_ptr(m->nb), get_ptr(m->hidxbs), get_ptr(m->ng), N_);
    m->workspace.resize(workspace_size);
    m->stats.resize(max_iter_*5);

    m->projvec.resize(nx_+na_);

  }

  void HpmpcInterface::
  eval(void* mem, const double** arg, double** res, int* iw, double* w) const {
    auto m = static_cast<HpmpcMemory*>(mem);

    const int* nx = get_ptr(m->nx) + 1;
    const std::vector<int>& nu = m->nu;
    const std::vector<int>& ng = m->ng;
    const std::vector<int>& nb = m->nb;

    // Dissect A matrix
    casadi_project(arg[CONIC_A], sparsity_in(CONIC_A), get_ptr(m->A), Asp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_A], sparsity_in(CONIC_A), get_ptr(m->B), Bsp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_A], sparsity_in(CONIC_A), get_ptr(m->C), Csp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_A], sparsity_in(CONIC_A), get_ptr(m->D), Dsp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_A], sparsity_in(CONIC_A), get_ptr(m->I), Isp_, get_ptr(m->projvec));

    for (auto && i : m->I) {
      casadi_assert_message(i==-1,
        "HPMPC error: gap constraint must depend on negative xk+1");
    }

    // Dissect H matrix
    casadi_project(arg[CONIC_H], sparsity_in(CONIC_H), get_ptr(m->R), Rsp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_H], sparsity_in(CONIC_H), get_ptr(m->S), Ssp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_H], sparsity_in(CONIC_H), get_ptr(m->Q), Qsp_, get_ptr(m->projvec));

    // The definition of HPMPC lacks a factor 2
    for (int i=0;i<m->R.size();++i) m->R[i] = 0.5*m->R[i];
    for (int i=0;i<m->S.size();++i) m->S[i] = 0.5*m->S[i];
    for (int i=0;i<m->Q.size();++i) m->Q[i] = 0.5*m->Q[i];


    // Dissect LBA/UBA
    casadi_project(arg[CONIC_LBA], sparsity_in(CONIC_LBA), get_ptr(m->b), bsp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_UBA], sparsity_in(CONIC_UBA), get_ptr(m->b2), bsp_, get_ptr(m->projvec));
    casadi_assert(std::equal(m->b.begin(), m->b.end(), m->b2.begin()));
    casadi_project(arg[CONIC_LBA], sparsity_in(CONIC_LBA), get_ptr(m->lg), lugsp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_UBA], sparsity_in(CONIC_UBA), get_ptr(m->ug), lugsp_, get_ptr(m->projvec));

    // Flip sign of b
    for (int i=0;i<m->b.size();++i) m->b[i] = -m->b[i];

    /* Disassemble LBX/UBX input
    */
    std::copy(arg[CONIC_LBX], arg[CONIC_LBX]+m->lb.size(), get_ptr(m->lb));
    std::copy(arg[CONIC_UBX], arg[CONIC_UBX]+m->ub.size(), get_ptr(m->ub));

    bool checkbounds = std::equal(get_ptr(m->lb)+nu[0], get_ptr(m->lb)+nu[0]+nx[0],
                                  get_ptr(m->ub)+nu[0]);
    casadi_assert_message(checkbounds,
        "HPMPC solver requires equality constraints on the first state.");

    // Dissect G
    casadi_project(arg[CONIC_G], sparsity_in(CONIC_G), get_ptr(m->r), usp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_G], sparsity_in(CONIC_G), get_ptr(m->q), xsp_, get_ptr(m->projvec));
    // The definition of HPMPC lacks a factor 2
    for (int i=0;i<m->r.size();++i) m->r[i] = 0.5*m->r[i];
    for (int i=0;i<m->q.size();++i) m->q[i] = 0.5*m->q[i];

    // Dissect X0
    casadi_project(arg[CONIC_X0], sparsity_in(CONIC_X0), get_ptr(m->u), usp_, get_ptr(m->projvec));
    casadi_project(arg[CONIC_X0], sparsity_in(CONIC_X0), get_ptr(m->x), xsp_, get_ptr(m->projvec));

    // Instead of expecting an initial state constraint,
    // HPMPC reads from x.
    std::copy(get_ptr(m->lb)+nu[0], get_ptr(m->lb)+nu[0]+nx[0], m->xs[0]);

    /* Disassemble LAM_X0 into
      pi
      lam

      pi
      lam

    */
    int offset = 0;
    for (int k=0;k<N_;++k) {
      std::copy(arg[CONIC_LAM_X0]+offset, arg[CONIC_LAM_X0]+offset+nx[k], m->pis[k]);
      offset+= nx[k];
      std::copy(arg[CONIC_LAM_X0]+offset, arg[CONIC_LAM_X0]+offset+ng[k], m->lams[k]);
      offset+= ng[k];
    }

    int kk = -1;

    std::vector<double> inf_norm_res(4);

    int N = N_;

    // First entry of b, augment with A*x
    casadi_mv(m->As[0], A0sp_, get_ptr(m->x), m->bs[0], 0);
    casadi_mv(m->Ss[0], S0sp_, get_ptr(m->x), m->rs[0], 0);

    for (int i=0;i<nx[0]*ng[0];++i) *(m->Cs[0]+i) = - *(m->Cs[0]+i);
    casadi_mv(m->Cs[0], C0sp_, get_ptr(m->x), m->lgs[0], 0);
    casadi_mv(m->Cs[0], C0sp_, get_ptr(m->x), m->ugs[0], 0);
    for (int i=0;i<nx[0]*ng[0];++i) *(m->Cs[0]+i) = - *(m->Cs[0]+i);

    int ret = fortran_order_d_ip_ocp_hard_tv(&kk, max_iter_, mu0_, tol_, N_, get_ptr(m->nx),
      get_ptr(m->nu), get_ptr(m->nb), get_ptr(m->hidxbs), get_ptr(m->ng), N_, warm_start_,
      get_ptr(m->As), get_ptr(m->Bs), get_ptr(m->bs), get_ptr(m->Qs), get_ptr(m->Ss),
      get_ptr(m->Rs), get_ptr(m->qs), get_ptr(m->rs), get_ptr(m->lbs), get_ptr(m->ubs),
      get_ptr(m->Cs), get_ptr(m->Ds), get_ptr(m->lgs), get_ptr(m->ugs), get_ptr(m->xs),
      get_ptr(m->us), get_ptr(m->pis), get_ptr(m->lams), get_ptr(inf_norm_res),
      get_ptr(m->workspace), get_ptr(m->stats));

    // Read in the results
    offset = 0;
    for (int k=0;k<N_;++k) {
      std::copy(m->us[k], m->us[k]+nu[k], res[CONIC_X]+offset);
      offset+= nu[k];
      std::copy(m->xs[k], m->xs[k]+nx[k], res[CONIC_X]+offset);
      offset+= nx[k];
    }
    std::copy(m->xs[N_], m->xs[N_]+nx[N_], res[CONIC_X]+offset);

    // Not implemented yet: multipliers
    std::fill(res[CONIC_LAM_X], res[CONIC_LAM_X]+nx_, nan);
    std::fill(res[CONIC_LAM_A], res[CONIC_LAM_A]+na_, nan);
  }

  HpmpcMemory::HpmpcMemory() {
  }

  HpmpcMemory::~HpmpcMemory() {

  }

  Sparsity HpmpcInterface::blocksparsity(int rows, int cols, const std::vector<Block>& b,
      bool eye) {
    DM r(rows, cols);
    for (auto && block : b) {
      int offset_r, offset_c, rows, cols;
      std::tie(offset_r, offset_c, rows, cols) = block;
      if (eye) {
        r(range(offset_r, offset_r+rows), range(offset_c, offset_c+cols)) = DM::eye(rows);
        casadi_assert(rows==cols);
      } else {
        r(range(offset_r, offset_r+rows), range(offset_c, offset_c+cols)) = DM::zeros(rows, cols);
      }
    }
    return r.sparsity();
  }

} // namespace casadi
