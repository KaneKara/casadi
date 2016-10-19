#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
from casadi import *
import casadi as c
import numpy
import unittest
from types import *
from helpers import *

conics = []
if has_nlpsol("ipopt"):
  ipopt_options = {"fixed_variable_treatment":"relax_bounds",
                   "jac_c_constant":"yes",
                   "jac_d_constant":"yes",
                   "hessian_constant":"yes",
                   "tol":1e-12}
  conics.append(("nlpsol",{"nlpsol":"ipopt", "nlpsol_options.ipopt": ipopt_options},{}))

if has_conic("ooqp"):
  conics.append(("ooqp",{},{"less_digits":1}))

if has_conic("qpoases"):
  conics.append(("qpoases",{},{}))

if has_conic("cplex"):
  conics.append(("cplex",{},{}))

# if has_conic("sqic"):
#   conics.append(("sqic",{},{}))

print(conics)

class ConicTests(casadiTestCase):

  @requires_conic("hpmpc")
  @requires_conic("qpoases")
  def test_hpmc(self):

    inf = 100
    T = 10. # Time horizon
    N = 4 # number of control intervals

    # Declare model variables
    x1 = MX.sym('x1')
    x2 = MX.sym('x2')
    x = vertcat(x1, x2)
    u = MX.sym('u')

    # Model equations
    xdot = vertcat(0.6*x1 - 1.11*x2 + 0.3*u-0.03, 0.7*x1+0.01)

    # Objective term
    L = x1**2 + 3*x2**2 + 7*u**2 -0.4*x1*x2-0.3*x1*u+u -x1-2*x2

    # Fixed step Runge-Kutta 4 integrator
    F = Function('F', [x, u], [x+xdot, L])

    J = F.jacobian()
    print J()
    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    Xs = SX.sym('X', 2, 1, N+1)
    Us = SX.sym('U', 1, 1, N+1)

    for k in range(N):
        w += [Us[k]]
        lbw += [-inf]
        ubw += [inf]
        w0  += [0]
        w += [Xs[k]]
          
        if k==0:
          lbw += [0, 1]
          ubw += [0, 1]
          w0 += [0.3, 0.7]
        elif k==3:
          lbw += [0.5, 0.3]
          ubw += [0.5, 0.3]
          w0 += [0.5, 0.3]
        else:
          lbw += [-inf, -inf]
          ubw += [  inf,  inf]
          w0  += [0, 1]

        xplus, l = F(Xs[k],Us[k])
        J+= l
        # Add equality constraint
        g   += [xplus-Xs[k+1]]
        lbg += [0, 0]
        ubg += [0, 0]

    J+= mtimes(Xs[-1].T,Xs[-1])

    w += [Xs[-1]]
    lbw += [-inf, -inf]
    ubw += [  inf,  inf]
    w0  += [0, 1]
          
    # Create an NLP solver
    prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}


    J = Function("J",[prob["x"]],[jacobian(prob["g"],prob["x"])])
    J(w0).print_dense()
    
        
    solver_ref = qpsol('solver', 'qpoases', prob)
    solver = qpsol('solver', 'hpmpc', prob,{"N":N,"nx":[2]*(N+1),"nu":[1]*N,"ng":[0]*(N+1),"tol":1e-12,"mu0":2,"max_iter":20})

    sol_ref = solver_ref(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    self.checkarray(sol_ref["x"], sol["x"])
    print sol["x"]
    assert False

    
if __name__ == '__main__':
    unittest.main()
