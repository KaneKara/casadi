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
        "Tolerance in the duality measure"}}}
  };

  void HpmpcInterface::init(const Dict& opts) {
    Conic::init(opts);

    mu0_ = 1;  
    max_iter_ = 1000;
    tol_ = 1e-8;
    warm_start_ = false;

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
      }
    }

  }

  void HpmpcInterface::init_memory(void* mem) const {
    auto m = static_cast<HpmpcMemory*>(mem);
    
    m->nx = nxs_;
    m->nu = nus_; m->nu.push_back(0);
    m->ng = ngs_;
    
    const std::vector<int>& nx = m->nx;
    const std::vector<int>& nu = m->nu;
    const std::vector<int>& ng = m->ng;
    
    m->nb.resize(N_+1);
    int offset = 0;
    for (int k=0;k<N_;++k) m->nb[k] = nx[k]+nu[k];
    m->nb[N_] = nx[N_];
    
    for (int k=0;k<N_;++k) offset+=nx[k]*nx[k];
    m->A.resize(offset);
    offset+=nx[N_]*nx[N_];
    m->Q.resize(offset);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nx[k]*nu[k];
    m->B.resize(offset);
    m->S.resize(offset);
    
    offset = 0;
    for (int k=0;k<N_;++k) offset+=nu[k]*nu[k];
    m->R.resize(offset);
    
    offset = 0;
    for (int k=0;k<N_;++k) offset+=nu[k]*ng[k];
    m->D.resize(offset);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nx[k]*ng[k];
    m->C.resize(offset);
    
    offset = nu[0];
    for (int k=1;k<N_;++k) offset+=nx[k]+nu[k];
    offset+=nx[N_];
    m->lb.resize(offset);
    m->ub.resize(offset);
    
    offset = 0;
    for (int k=0;k<N_;++k) offset+=nx[k];
    m->b.resize(offset);
    m->pi.resize(offset);
    offset+=nx[N_];
    m->q.resize(offset);
    m->x.resize(offset);

    offset = 0;
    for (int k=0;k<N_;++k) offset+=nu[k];
    m->r.resize(offset);
    m->u.resize(offset);
    
    offset = 0;
    for (int k=0;k<N_;++k) offset+=ng[k];
    m->lg.resize(offset);
    m->ug.resize(offset);
    m->lam.resize(offset);
    
    offset = 0;
    for (int k=0;k<N_+1;++k) offset+=m->nb[k];
    m->hidxb.resize(offset);
    
    offset = 0;
    for (int k=0;k<N_;++k) offset+=ng[k]+nx[k]+nu[k];
    m->lam.resize(offset);
    
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
    offset = nu[0];
    for (int k=1;k<N_;++k) {
      m->lbs[k] = get_ptr(m->lb)+offset;
      m->ubs[k] = get_ptr(m->ub)+offset;
      offset+=nu[k]+nx[k];
    }
    m->lbs[N_] = get_ptr(m->lb)+offset;
    m->ubs[N_] = get_ptr(m->ub)+offset;

    m->lgs.resize(N_+1);
    m->ugs.resize(N_+1);
    m->lams.resize(N_+1);
    offset = 0;
    for (int k=0;k<N_+1;++k) {
      m->lgs[k] = get_ptr(m->lg)+offset;
      m->ugs[k] = get_ptr(m->ug)+offset;
      m->lams[k] = get_ptr(m->lam)+offset;
      offset+=ng[k];
    }

    m->hidxbs.resize(N_+1);
    offset = 0;
    for (int k=0;k<N_+1;++k) {
      m->hidxbs[k] = get_ptr(m->hidxb)+offset;
      for (int i=0;i<m->nb[k];++i) m->hidxbs[k][i] = i;
      offset+=m->nb[k];
    }

    int workspace_size = hpmpc_d_ip_ocp_hard_tv_work_space_size_bytes(N_, get_ptr(m->nx), get_ptr(m->nu), get_ptr(m->nb), get_ptr(m->hidxbs), get_ptr(m->ng), N_);
	  m->workspace.resize(workspace_size);
	  
	  m->stats.resize(max_iter_*5);


  }

  void HpmpcInterface::
  eval(void* mem, const double** arg, double** res, int* iw, double* w) const {
    auto m = static_cast<HpmpcMemory*>(mem);

    const std::vector<int>& nx = m->nx;
    const std::vector<int>& nu = m->nu;
    const std::vector<int>& ng = m->ng;
    const std::vector<int>& nb = m->nb;
    
    const Sparsity& asp = sparsity_in(CONIC_A);
    const int* colind = asp.colind();
    const int* row = asp.row();
    
    DM debug(asp);
    std::copy(arg[CONIC_A],arg[CONIC_A]+debug.nnz(),debug.nonzeros().begin());
    debug.print_dense();
    
    debug = DM(sparsity_in(CONIC_LBA));
    std::copy(arg[CONIC_LBA],arg[CONIC_LBA]+debug.nnz(),debug.nonzeros().begin());
    debug.print_dense();
    
    debug = DM(sparsity_in(CONIC_LBA));
    std::copy(arg[CONIC_UBA],arg[CONIC_UBA]+debug.nnz(),debug.nonzeros().begin());
    debug.print_dense();
    
    debug = DM(sparsity_in(CONIC_H));
    std::copy(arg[CONIC_H],arg[CONIC_H]+debug.nnz(),debug.nonzeros().begin());
    debug.print_dense();
    
    int offset_col = 0;
    int offset_row = 0;
    int cntA = 0;
    int cntB = 0;
    int cntC = 0;
    int cntD = 0;
    bool flag_identity = true;
    
    /* Disassemble A input into:
       B A   I
       D C
           B A  I
           D C
    */
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
    
    /* Disassemble H input into:
       R S
       S'Q'
           R S
           S'Q'
    */
    colind = sparsity_in(CONIC_H).colind();
    row = sparsity_in(CONIC_H).row();
    int offset = 0;
    for (int k=0;k<N_;++k) {
      for (int cc=0; cc<nu[k]; ++cc) {
        for (int el=colind[offset+cc]; el<colind[offset+cc+1]; ++el) {
          int r = row[el];
          if (r<offset+nu[k])
            m->Rs[k][nu[k]*cc+r-offset] = arg[CONIC_H][el];
        }
      }
      for (int cc=0; cc<nx[k]; ++cc) {
        for (int el=colind[offset+cc+nu[k]]; el<colind[offset+cc+1+nu[k]]; ++el) {
          int r = row[el];
          if (row[el]<offset+nu[k])
            m->Ss[k][nu[k]*cc+r-offset] = arg[CONIC_H][el];
          else 
            m->Qs[k][nx[k]*cc+r-offset-nu[k]] = arg[CONIC_H][el];
        }
      }

      offset += nx[k] + nu[k];
    }
    
    for (int cc=0; cc<nx[N_]; ++cc) {
      for (int el=colind[offset+cc]; el<colind[offset+cc+1]; ++el) {
        int r = row[el];
        m->Qs[N_][nx[N_]*cc+r-offset] = arg[CONIC_H][el];
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
      casadi_assert(std::equal(arg[CONIC_LBA]+offset, arg[CONIC_LBA]+offset+nx[k], arg[CONIC_UBA]+offset));
      offset+= nx[k];
      std::copy(arg[CONIC_LBA]+offset, arg[CONIC_LBA]+offset+ng[k], m->lgs[k]);
      std::copy(arg[CONIC_UBA]+offset, arg[CONIC_UBA]+offset+ng[k], m->ugs[k]);
      offset+= ng[k];
    }
    
    /* Disassemble LBX/UBX input
    */
    std::copy(arg[CONIC_LBX],arg[CONIC_LBX]+m->lb.size(), get_ptr(m->lb));
    std::copy(arg[CONIC_UBX],arg[CONIC_UBX]+m->ub.size(), get_ptr(m->ub));
    
    /* Disassemble G input into:
       r
       q
       
       r
       q
    */
    offset = 0;
    for (int k=0;k<N_;++k) {
      std::copy(arg[CONIC_G]+offset, arg[CONIC_G]+offset+nu[k], m->rs[k]);
      offset+= nu[k];
      std::copy(arg[CONIC_G]+offset, arg[CONIC_G]+offset+nx[k], m->qs[k]);
      offset+= nx[k];
    }
    std::copy(arg[CONIC_G]+offset, arg[CONIC_G]+offset+nx[N_], m->qs[N_]);
    
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

    
    /* Disassemble LAM_X0 into
      u
      x
      
      u
      x
      
    */
    offset = 0;
    for (int k=0;k<N_;++k) {
      std::copy(arg[CONIC_LAM_X0]+offset, arg[CONIC_LAM_X0]+offset+nx[k], m->pis[k]);
      offset+= nx[k];
      std::copy(arg[CONIC_LAM_X0]+offset, arg[CONIC_LAM_X0]+offset+ng[k], m->lams[k]);
      offset+= ng[k];
    }
    





    std::cout << m->A << std::endl;
    std::cout << m->B << std::endl;
    std::cout << m->C << std::endl;
    std::cout << m->D << std::endl;
    

    std::cout << "R" << m->R << std::endl;
    std::cout << "S" << m->S << std::endl;
    std::cout << "Q" << m->Q << std::endl;


    std::cout << "b" << m->b << std::endl;

    std::cout << "r" << m->r << std::endl;
    std::cout << "q" << m->q << std::endl;


    std::cout << "lb" << m->lb << std::endl;
    std::cout << "ub" << m->ub << std::endl;

    std::cout << "lg" << m->lg << std::endl;
    std::cout << "ug" << m->ug << std::endl;

    std::cout << "x" << m->x << std::endl;
    std::cout << "u" << m->u << std::endl;

    std::cout << "pi" << m->pi << std::endl;
    std::cout << "lam" << m->lam << std::endl;

    std::cout << "hidxb" << m->hidxb << std::endl;
      
    int kk = -1;
    
    std::vector<double> inf_norm_res(4);
    
    std::cout << "tol_: " << tol_ << std::endl;
    
    int ret = fortran_order_d_ip_ocp_hard_tv(&kk, max_iter_, mu0_, tol_,	N_, get_ptr(m->nx), get_ptr(m->nu), get_ptr(m->nb), get_ptr(m->hidxbs), get_ptr(m->ng), N_, warm_start_, get_ptr(m->As), get_ptr(m->Bs), get_ptr(m->bs), get_ptr(m->Qs), get_ptr(m->Ss), get_ptr(m->Rs), get_ptr(m->qs), get_ptr(m->rs), get_ptr(m->lbs), get_ptr(m->ubs), get_ptr(m->Cs), get_ptr(m->Ds), get_ptr(m->lgs), get_ptr(m->ugs), get_ptr(m->xs), get_ptr(m->us), get_ptr(m->pis), get_ptr(m->lams), get_ptr(inf_norm_res), get_ptr(m->workspace), get_ptr(m->stats));

    std::cout << "status: " << ret << std::endl;
    
    std::cout << "tol_: " << inf_norm_res << std::endl;
    offset = 0;
    for (int k=0;k<N_;++k) {
      std::copy(m->us[k], m->us[k]+nu[k], res[CONIC_X]+offset);
      offset+= nu[k];
      std::copy(m->xs[k], m->xs[k]+nx[k], res[CONIC_X]+offset);
      offset+= nx[k];
    }
    std::copy(m->xs[N_], m->xs[N_]+nx[N_], res[CONIC_X]+offset);

    
    casadi_assert(flag_identity);
    
  }

  HpmpcMemory::HpmpcMemory() {
    userOut() << "init" << std::endl;
  }

  HpmpcMemory::~HpmpcMemory() {

  }

} // namespace casadi
