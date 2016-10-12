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
L = x1**2 + 3*x2**2 + 7*u**2 -0.4*x1*x2-0.3*x1*u+u-x1-2*x2

# Fixed step Runge-Kutta 4 integrator
F = Function('F', [x, u], [x + xdot, L])

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
    lbw += [-5]
    ubw += [5]
    w0  += [0]
    w += [Xs[k]]
      
    if k==0:
      #lbw += [0, 1]
      #ubw += [0, 1]
      lbw += [-inf, -inf]
      ubw += [inf, inf]
      w0 += [0, 1]
    else:
      lbw += [-inf, -inf]
      ubw += [  inf,  inf]
      w0  += [4, 1]

    xplus, l = F(Xs[k],Us[k])
    J+= l
    # Add equality constraint
    g   += [xplus-Xs[k+1]]
    lbg += [0, 0]
    ubg += [0, 0]

J+= mtimes(Xs[-1].T,Xs[-1])+0.1*Xs[-1][0]*Xs[-1][1]+3+0.03*Xs[-1][0]

w += [Xs[-1]]
lbw += [-0.25, -inf]
ubw += [  inf,  inf]
w0  += [4, 1]
      
# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}


J = Function("J",[prob["x"]],[jacobian(prob["g"],prob["x"])])
J(w0).print_dense()
#solver = qpsol('solver', 'qpoases', prob)
solver = qpsol('solver', 'hpmpc', prob,{"N":N,"nx":[2]*(N+1),"nu":[1]*N,"ng":[0]*(N+1)})

#[2.12347, 0, 1, 4.68448, -0.50296, 1.01, 4.1949, -0.550493, 0.667928, 2.60331, -0.393719, 0.292583, 1.0789, -0.203723, 0.0269795, 0.083636, -0.0622339, -0.105627, -0.365252, 0.0127622, -0.13919, -0.44035, 0.0353452, -0.120257, -0.336076, 0.0279325, -0.0855153, -0.191927, 0.0087912, -0.0559625, -0.0787218, -0.0113937, -0.0398086, -0.015449, -0.0276589, -0.0377843, 0.00674384, -0.0369485, -0.0471455, 0.00541802, -0.0347629, -0.0630095, -0.00411427, -0.0140548, -0.0773435, -0.0129435, 0.0321294, -0.0771818, -0.0186908, 0.103196, -0.0446912, -0.023901, 0.179113, 0.0375458, -0.0337928, 0.207735, 0.172925, -0.0533645, 0.100292, 0.32834, -0.25, 0.408544]


# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
print sol['x']



