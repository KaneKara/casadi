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

    // Check sparsity pattern of A, G

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

    // Allocate double work vectors
    for (int k=0;k<N_;++k) offset+=nx[k]*nx[k];
    m->A.resize(offset);m->A.reserve(offset+1);
    offset+=nx[N_]*nx[N_];
    m->Q.resize(offset);m->Q.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nx[k]*nu[k];
    m->B.resize(offset);m->B.reserve(offset+1);
    m->S.resize(offset);m->S.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nu[k]*nu[k];
    m->R.resize(offset);m->R.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nu[k]*ng[k];
    m->D.resize(offset);m->D.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nx[k]*ng[k];
    m->C.resize(offset);m->C.reserve(offset+1);

    offset = nu[0]+nx[0];
    for (int k=1;k<N_;++k) offset+=nx[k]+nu[k];
    offset+=nx[N_];
    m->lb.resize(offset);m->lb.reserve(offset+1);
    m->ub.resize(offset);m->ub.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nx[k];
    m->b.resize(offset);m->b.reserve(offset+1);
    m->pi.resize(offset);m->pi.reserve(offset+1);
    offset+=nx[N_];
    m->q.resize(offset);m->q.reserve(offset+1);
    m->x.resize(offset);m->x.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nu[k];
    m->r.resize(offset);m->r.reserve(offset+1);
    m->u.resize(offset);m->u.reserve(offset+1);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=ng[k];
    m->lg.resize(offset);m->lg.reserve(offset+1);
    m->ug.resize(offset);m->ug.reserve(offset+1);

    if (offset) casadi_warning("Untested: non-gap constraints.");

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



  }

  void HpmpcInterface::
  eval(void* mem, const double** arg, double** res, int* iw, double* w) const {
    auto m = static_cast<HpmpcMemory*>(mem);

    const int* nx = get_ptr(m->nx) + 1;
    const std::vector<int>& nu = m->nu;
    const std::vector<int>& ng = m->ng;
    const std::vector<int>& nb = m->nb;

    const Sparsity& asp = sparsity_in(CONIC_A);
    const int* colind = asp.colind();
    const int* row = asp.row();

    /* Disassemble A input into:
       B A   I
       D C
           B A  I
           D C
    */
    {
      int offset_col = 0;
      int offset_row = 0;
      bool flag_identity = true;
      for (int k=0;k<N_;++k) {
        for (int cc=0; cc<nu[k]; ++cc) {
          for (int el=colind[offset_col+cc]; el<colind[offset_col+cc+1]; ++el) {
            int r = row[el];
            if (r<offset_row+nx[k])
              m->Bs[k][nx[k]*cc+r-offset_row] = arg[CONIC_A][el];
            else
              m->Ds[k][ng[k]*cc+r-offset_row-nx[k]] = arg[CONIC_A][el];
          }
        }
        for (int cc=0; cc<nx[k]; ++cc) {
          for (int el=colind[offset_col+cc+nu[k]]; el<colind[offset_col+cc+1+nu[k]]; ++el) {
            int r = row[el];
            if (r<offset_row)
              flag_identity &= arg[CONIC_A][el]==-1;
            else if (row[el]<offset_row+nx[k])
              m->As[k][nx[k]*cc+r-offset_row] = arg[CONIC_A][el];
            else
              m->Cs[k][ng[k]*cc+r-offset_row-nx[k]] = arg[CONIC_A][el];
          }
        }

        offset_col += nx[k] + nu[k];
        offset_row += nx[k] + ng[k];
      }
      casadi_assert_message(flag_identity,
        "HPMPC error: gap constraint must depend on negative xk+1");
    }

    /* Disassemble H input into:
       R S
       S'Q'
           R S
           S'Q'

       Multiply by 2
    */
    colind = sparsity_in(CONIC_H).colind();
    row = sparsity_in(CONIC_H).row();
    int offset = 0;
    for (int k=0;k<N_;++k) {
      for (int cc=0; cc<nu[k]; ++cc) {
        for (int el=colind[offset+cc]; el<colind[offset+cc+1]; ++el) {
          int r = row[el];
          if (r<offset+nu[k])
            m->Rs[k][nu[k]*cc+r-offset] = 0.5*arg[CONIC_H][el];
        }
      }
      for (int cc=0; cc<nx[k]; ++cc) {
        for (int el=colind[offset+cc+nu[k]]; el<colind[offset+cc+1+nu[k]]; ++el) {
          int r = row[el];
          if (row[el]<offset+nu[k])
            m->Ss[k][nu[k]*cc+r-offset] = 0.5*arg[CONIC_H][el];
          else
            m->Qs[k][nx[k]*cc+r-offset-nu[k]] = 0.5*arg[CONIC_H][el];
        }
      }

      offset += nx[k] + nu[k];
    }

    for (int cc=0; cc<nx[N_]; ++cc) {
      for (int el=colind[offset+cc]; el<colind[offset+cc+1]; ++el) {
        int r = row[el];
        m->Qs[N_][nx[N_]*cc+r-offset] = 0.5*arg[CONIC_H][el];
      }
    }


    /* Disassemble LBA/UBA input into:
       b
       lg/ug

       b
       lg/ug
    */
    offset = 0;
    for (int k=0;k<N_;++k) {
      std::copy(arg[CONIC_LBA]+offset, arg[CONIC_LBA]+offset+nx[k], m->bs[k]);
      casadi_assert(std::equal(arg[CONIC_LBA]+offset, arg[CONIC_LBA]+offset+nx[k],
        arg[CONIC_UBA]+offset));
      offset+= nx[k];
      std::copy(arg[CONIC_LBA]+offset, arg[CONIC_LBA]+offset+ng[k], m->lgs[k]);
      std::copy(arg[CONIC_UBA]+offset, arg[CONIC_UBA]+offset+ng[k], m->ugs[k]);
      offset+= ng[k];
    }

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

    /* Disassemble G input into:
       r
       q

       r
       q
    */
    offset = 0;
    for (int k=0;k<N_;++k) {
      std::copy(arg[CONIC_G]+offset, arg[CONIC_G]+offset+nu[k], m->rs[k]);
      for (int i=0;i<nu[k];++i) m->rs[k][i] = 0.5*m->rs[k][i];
      offset+= nu[k];
      std::copy(arg[CONIC_G]+offset, arg[CONIC_G]+offset+nx[k], m->qs[k]);
      for (int i=0;i<nx[k];++i) m->qs[k][i] = 0.5*m->qs[k][i];
      offset+= nx[k];
    }

    /* Disassemble X0 into
      u
      x

      u
      x

    */
    offset = 0;
    for (int k=0;k<N_;++k) {
      std::copy(arg[CONIC_X0]+offset, arg[CONIC_X0]+offset+nu[k], m->us[k]);
      offset+= nu[k];
      std::copy(arg[CONIC_X0]+offset, arg[CONIC_X0]+offset+nx[k], m->xs[k]);
      offset+= nx[k];
    }
    std::copy(arg[CONIC_X0]+offset, arg[CONIC_X0]+offset+nx[N_], m->xs[N_]);

    // Instead of expecting an initial state constraint,
    // HPMPC reads from x.
    std::copy(get_ptr(m->lb)+nu[0], get_ptr(m->lb)+nu[0]+nx[0], m->xs[0]);

    /* Disassemble LAM_X0 into
      pi
      lam

      pi
      lam

    */
    offset = 0;
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

} // namespace casadi
