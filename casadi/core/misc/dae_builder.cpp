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


#include "dae_builder.hpp"

#include <map>
#include <string>
#include <sstream>
#include <ctime>

#include "../std_vector_tools.hpp"
#include "../casadi_exception.hpp"
#include "../matrix/matrix_tools.hpp"
#include "../mx/mx_tools.hpp"
#include "../function/mx_function.hpp"
#include "../function/integrator.hpp"
#include "../function/code_generator.hpp"
#include "../casadi_calculus.hpp"
#include "xml_file.hpp"

using namespace std;
namespace casadi {

  inline string str(const vector<MX>& v) {
    // TODO(@jaeandersson) Move to std_vector_tools.hpp
    stringstream s;
    s << "[";
    for (int i=0; i<v.size(); ++i) {
      if (i>0) s << ", ";
      s << str(v[i]);
    }
    s << "]";
    return s.str();
  }

  DaeBuilder::DaeBuilder() {
    this->t = MX::sym("t");
  }

  void DaeBuilder::parseFMI(const std::string& filename) {

    // Load
    XmlFile xml_file("tinyxml");
    XmlNode document = xml_file.parse(filename);

    // **** Add model variables ****
    {
      //if (verbose) cout << "Adding model variables." << endl;

      // Get a reference to the ModelVariables node
      const XmlNode& modvars = document[0]["ModelVariables"];

      // Add variables
      for (int i=0; i<modvars.size(); ++i) {

        // Get a reference to the variable
        const XmlNode& vnode = modvars[i];

        // Get the attributes
        string name        = vnode.getAttribute("name");
        int valueReference;
        vnode.readAttribute("valueReference", valueReference);
        string variability = vnode.getAttribute("variability");
        string causality   = vnode.getAttribute("causality");
        string alias       = vnode.getAttribute("alias");

        // Skip to the next variable if its an alias
        if (alias.compare("alias") == 0 || alias.compare("negatedAlias") == 0)
          continue;

        // Get the name
        const XmlNode& nn = vnode["QualifiedName"];
        string qn = qualifiedName(nn);

        // Add variable, if not already added
        if (varmap_.find(qn)==varmap_.end()) {

          // Create variable
          Variable var(name);

          // Value reference
          var.valueReference = valueReference;

          // Variability
          if (variability.compare("constant")==0)
            var.variability = CONSTANT;
          else if (variability.compare("parameter")==0)
            var.variability = PARAMETER;
          else if (variability.compare("discrete")==0)
            var.variability = DISCRETE;
          else if (variability.compare("continuous")==0)
            var.variability = CONTINUOUS;
          else
            throw CasadiException("Unknown variability");

          // Causality
          if (causality.compare("input")==0)
            var.causality = INPUT;
          else if (causality.compare("output")==0)
            var.causality = OUTPUT;
          else if (causality.compare("internal")==0)
            var.causality = INTERNAL;
          else
            throw CasadiException("Unknown causality");

          // Alias
          if (alias.compare("noAlias")==0)
            var.alias = NO_ALIAS;
          else if (alias.compare("alias")==0)
            var.alias = ALIAS;
          else if (alias.compare("negatedAlias")==0)
            var.alias = NEGATED_ALIAS;
          else
            throw CasadiException("Unknown alias");

          // Other properties
          if (vnode.hasChild("Real")) {
            const XmlNode& props = vnode["Real"];
            props.readAttribute("unit", var.unit, false);
            props.readAttribute("displayUnit", var.displayUnit, false);
            props.readAttribute("min", var.min, false);
            props.readAttribute("max", var.max, false);
            props.readAttribute("initialGuess", var.initialGuess, false);
            props.readAttribute("start", var.start, false);
            props.readAttribute("nominal", var.nominal, false);
            props.readAttribute("free", var.free, false);
          }

          // Variable category
          if (vnode.hasChild("VariableCategory")) {
            string cat = vnode["VariableCategory"].getText();
            if (cat.compare("derivative")==0)
              var.category = CAT_DERIVATIVE;
            else if (cat.compare("state")==0)
              var.category = CAT_STATE;
            else if (cat.compare("dependentConstant")==0)
              var.category = CAT_DEPENDENT_CONSTANT;
            else if (cat.compare("independentConstant")==0)
              var.category = CAT_INDEPENDENT_CONSTANT;
            else if (cat.compare("dependentParameter")==0)
              var.category = CAT_DEPENDENT_PARAMETER;
            else if (cat.compare("independentParameter")==0)
              var.category = CAT_INDEPENDENT_PARAMETER;
            else if (cat.compare("algebraic")==0)
              var.category = CAT_ALGEBRAIC;
            else
              throw CasadiException("Unknown variable category: " + cat);
          }

          // Add to list of variables
          addVariable(qn, var);

          // Sort expression
          switch (var.category) {
          case CAT_DERIVATIVE:
            // Skip - meta information about time derivatives is
            //        kept together with its parent variable
            break;
          case CAT_STATE:
            this->s.push_back(var.v);
            this->sdot.push_back(var.d);
            break;
          case CAT_DEPENDENT_CONSTANT:
            // Skip
            break;
          case CAT_INDEPENDENT_CONSTANT:
            // Skip
            break;
          case CAT_DEPENDENT_PARAMETER:
            // Skip
            break;
          case CAT_INDEPENDENT_PARAMETER:
            if (var.free) {
              this->p.push_back(var.v);
            } else {
              // Skip
            }
            break;
          case CAT_ALGEBRAIC:
            if (var.causality == INTERNAL) {
              this->s.push_back(var.v);
              this->sdot.push_back(var.d);
            } else if (var.causality == INPUT) {
              this->u.push_back(var.v);
            }
            break;
          default:
            casadi_error("Unknown category");
          }
        }
      }
    }

    // **** Add binding equations ****
    {
      //if (verbose) cout << "Adding binding equations." << endl;

      // Get a reference to the BindingEquations node
      const XmlNode& bindeqs = document[0]["equ:BindingEquations"];

      for (int i=0; i<bindeqs.size(); ++i) {
        const XmlNode& beq = bindeqs[i];

        // Get the variable and binding expression
        Variable& var = readVariable(beq[0]);
        MX bexpr = readExpr(beq[1][0]);
        this->i.push_back(var.v);
        this->idef.push_back(bexpr);
      }
    }

    // **** Add dynamic equations ****
    {
      // Get a reference to the DynamicEquations node
      const XmlNode& dyneqs = document[0]["equ:DynamicEquations"];

      // Add equations
      for (int i=0; i<dyneqs.size(); ++i) {

        // Get a reference to the variable
        const XmlNode& dnode = dyneqs[i];

        // Add the differential equation
        MX de_new = readExpr(dnode[0]);
        this->dae.push_back(de_new);
      }
    }

    // **** Add initial equations ****
    {
      // Get a reference to the DynamicEquations node
      const XmlNode& initeqs = document[0]["equ:InitialEquations"];

      // Add equations
      for (int i=0; i<initeqs.size(); ++i) {

        // Get a reference to the node
        const XmlNode& inode = initeqs[i];

        // Add the differential equations
        for (int i=0; i<inode.size(); ++i) {
          this->init.push_back(readExpr(inode[i]));
        }
      }
    }

    // **** Add optimization ****
    if (document[0].hasChild("opt:Optimization")) {

      // Get a reference to the DynamicEquations node
      const XmlNode& opts = document[0]["opt:Optimization"];
      for (int i=0; i<opts.size(); ++i) {

        // Get a reference to the node
        const XmlNode& onode = opts[i];

        // Get the type
        if (onode.checkName("opt:ObjectiveFunction")) { // mayer term
          try {
            // Add components
            for (int i=0; i<onode.size(); ++i) {
              const XmlNode& var = onode[i];

              // If string literal, ignore
              if (var.checkName("exp:StringLiteral"))
                continue;

              // Read expression
              MX v = readExpr(var);

              // Treat as an output
              add_y("mterm");
              add_ydef(v, "mterm_rhs");
            }
          } catch(exception& ex) {
            throw CasadiException(std::string("addObjectiveFunction failed: ") + ex.what());
          }
        } else if (onode.checkName("opt:IntegrandObjectiveFunction")) {
          try {
            for (int i=0; i<onode.size(); ++i) {
              const XmlNode& var = onode[i];

              // If string literal, ignore
              if (var.checkName("exp:StringLiteral")) continue;

              // Read expression
              MX v = readExpr(var);

              // Treat as a quadrature state
              add_q("lterm");
              add_quad(v, "lterm_rhs");
            }
          } catch(exception& ex) {
            throw CasadiException(std::string("addIntegrandObjectiveFunction failed: ")
                                  + ex.what());
          }
        } else if (onode.checkName("opt:IntervalStartTime")) {
          // Ignore, treated above
        } else if (onode.checkName("opt:IntervalFinalTime")) {
          // Ignore, treated above
        } else if (onode.checkName("opt:TimePoints")) {
          // Ignore, treated above
        } else if (onode.checkName("opt:PointConstraints")) {
          casadi_warning("opt:PointConstraints not supported, ignored");
        } else if (onode.checkName("opt:Constraints")) {
          casadi_warning("opt:Constraints not supported, ignored");
        } else if (onode.checkName("opt:PathConstraints")) {
          casadi_warning("opt:PointConstraints not supported, ignored");
        } else {
          casadi_warning("DaeBuilder::addOptimization: Unknown node " << onode.getName());
        }
      }
    }

    // Make sure that the dimensions are consistent at this point
    casadi_assert_warning(this->s.size()==this->dae.size(),
                          "The number of differential-algebraic equations does not match "
                          "the number of implicitly defined states.");
    casadi_assert_warning(this->z.size()==this->alg.size(),
                          "The number of algebraic equations (equations not involving "
                          "differentiated variables) does not match the number of "
                          "algebraic variables.");
  }

  Variable& DaeBuilder::readVariable(const XmlNode& node) {
    // Qualified name
    string qn = qualifiedName(node);

    // Find and return the variable
    return variable(qn);
  }

  MX DaeBuilder::readExpr(const XmlNode& node) {
    const string& fullname = node.getName();
    if (fullname.find("exp:")== string::npos) {
      casadi_error("DaeBuilder::readExpr: unknown - expression is supposed to "
                   "start with 'exp:' , got " << fullname);
    }

    // Chop the 'exp:'
    string name = fullname.substr(4);

    // The switch below is alphabetical, and can be thus made more efficient,
    // for example by using a switch statement of the first three letters,
    // if it would ever become a bottleneck
    if (name.compare("Add")==0) {
      return readExpr(node[0]) + readExpr(node[1]);
    } else if (name.compare("Acos")==0) {
      return acos(readExpr(node[0]));
    } else if (name.compare("Asin")==0) {
      return asin(readExpr(node[0]));
    } else if (name.compare("Atan")==0) {
      return atan(readExpr(node[0]));
    } else if (name.compare("Cos")==0) {
      return cos(readExpr(node[0]));
    } else if (name.compare("Der")==0) {
      const Variable& v = readVariable(node[0]);
      return v.d;
    } else if (name.compare("Div")==0) {
      return readExpr(node[0]) / readExpr(node[1]);
    } else if (name.compare("Exp")==0) {
      return exp(readExpr(node[0]));
    } else if (name.compare("Identifier")==0) {
      return readVariable(node).v;
    } else if (name.compare("IntegerLiteral")==0) {
      int val;
      node.getText(val);
      return val;
    } else if (name.compare("Instant")==0) {
      double val;
      node.getText(val);
      return val;
    } else if (name.compare("Log")==0) {
      return log(readExpr(node[0]));
    } else if (name.compare("LogLt")==0) { // Logical less than
      return readExpr(node[0]) < readExpr(node[1]);
    } else if (name.compare("LogGt")==0) { // Logical less than
      return readExpr(node[0]) > readExpr(node[1]);
    } else if (name.compare("Mul")==0) { // Multiplication
      return readExpr(node[0]) * readExpr(node[1]);
    } else if (name.compare("Neg")==0) {
      return -readExpr(node[0]);
    } else if (name.compare("NoEvent")==0) {
      // NOTE: This is a workaround, we assume that whenever NoEvent occurs,
      // what is meant is a switch
      int n = node.size();

      // Default-expression
      MX ex = readExpr(node[n-1]);

      // Evaluate ifs
      for (int i=n-3; i>=0; i -= 2) ex = if_else(readExpr(node[i]), readExpr(node[i+1]), ex);

      return ex;
    } else if (name.compare("Pow")==0) {
      return pow(readExpr(node[0]), readExpr(node[1]));
    } else if (name.compare("RealLiteral")==0) {
      double val;
      node.getText(val);
      return val;
    } else if (name.compare("Sin")==0) {
      return sin(readExpr(node[0]));
    } else if (name.compare("Sqrt")==0) {
      return sqrt(readExpr(node[0]));
    } else if (name.compare("StringLiteral")==0) {
      throw CasadiException(node.getText());
    } else if (name.compare("Sub")==0) {
      return readExpr(node[0]) - readExpr(node[1]);
    } else if (name.compare("Tan")==0) {
      return tan(readExpr(node[0]));
    } else if (name.compare("Time")==0) {
      return t;
    } else if (name.compare("TimedVariable")==0) {
      return readVariable(node[0]).v;
    }

    // throw error if reached this point
    throw CasadiException(string("DaeBuilder::readExpr: Unknown node: ") + name);

  }

  void DaeBuilder::repr(std::ostream &stream, bool trailing_newline) const {
    stream << "DAE("
           << "#s = " << this->s.size() << ", "
           << "#x = " << this->x.size() << ", "
           << "#z = " << this->z.size() << ", "
           << "#q = " << this->q.size() << ", "
           << "#i = " << this->i.size() << ", "
           << "#y = " << this->y.size() << ", "
           << "#u = " << this->u.size() << ", "
           << "#p = " << this->p.size() << ")";
    if (trailing_newline) stream << endl;
  }

  void DaeBuilder::print(ostream &stream, bool trailing_newline) const {
    // Assert correctness
    sanityCheck();

    // Print dimensions
    repr(stream);

    // Print the variables
    stream << "Variables" << endl;
    stream << "{" << endl;
    stream << "  t = " << str(this->t) << endl;
    stream << "  s = " << str(this->s) << endl;
    stream << "  x = " << str(this->x) << endl;
    stream << "  z =  " << str(this->z) << endl;
    stream << "  q =  " << str(this->q) << endl;
    stream << "  i =  " << str(this->i) << endl;
    stream << "  y =  " << str(this->y) << endl;
    stream << "  u =  " << str(this->u) << endl;
    stream << "  p =  " << str(this->p) << endl;
    stream << "}" << endl;

    if (!this->dae.empty()) {
      stream << "Fully-implicit differential-algebraic equations" << endl;
      for (int k=0; k<this->dae.size(); ++k) {
        stream << "0 == " << this->dae[k] << endl;
      }
      stream << endl;
    }

    if (!this->x.empty()) {
      stream << "Differential equations" << endl;
      for (int k=0; k<this->x.size(); ++k) {
        stream << str(der(this->x[k])) << " == " << str(this->ode[k]) << endl;
      }
      stream << endl;
    }

    if (!this->alg.empty()) {
      stream << "Algebraic equations" << endl;
      for (int k=0; k<this->z.size(); ++k) {
        stream << "0 == " << str(this->alg[k]) << endl;
      }
      stream << endl;
    }

    if (!this->q.empty()) {
      stream << "Quadrature equations" << endl;
      for (int k=0; k<this->q.size(); ++k) {
        stream << str(der(this->q[k])) << " == " << str(this->quad[k]) << endl;
      }
      stream << endl;
    }

    if (!this->init.empty()) {
      stream << "Initial equations" << endl;
      for (int k=0; k<this->init.size(); ++k) {
        stream << "0 == " << str(this->init[k]) << endl;
      }
      stream << endl;
    }

    if (!this->i.empty()) {
      stream << "Intermediate variables" << endl;
      for (int i=0; i<this->i.size(); ++i)
        stream << str(this->i[i]) << " == " << str(this->idef[i]) << endl;
      stream << endl;
    }

    if (!this->y.empty()) {
      stream << "Output variables" << endl;
      for (int i=0; i<this->y.size(); ++i)
        stream << str(this->y[i]) << " == " << str(this->ydef[i]) << endl;
      stream << endl;
    }
    if (trailing_newline) stream << endl;
  }

  void DaeBuilder::eliminate_quad() {
    // Move all the quadratures to the list of differential states
    this->x.insert(this->x.end(), this->q.begin(), this->q.end());
    this->q.clear();
  }

  void DaeBuilder::scaleVariables() {
    // Assert correctness
    sanityCheck();

    // Gather variables and expressions to replace
    vector<MX> v_id, v_rep;
    for (VarMap::iterator it=varmap_.begin(); it!=varmap_.end(); ++it) {
      if (it->second.nominal!=1) {
        Variable& v=it->second;
        casadi_assert(v.nominal!=0);
        v.min /= v.nominal;
        v.max /= v.nominal;
        v.start /= v.nominal;
        v.derivativeStart /= v.nominal;
        v.initialGuess /= v.nominal;
        v_id.push_back(v.v);
        v_id.push_back(v.d);
        v_rep.push_back(v.v * v.nominal);
        v_rep.push_back(v.d * v.nominal);
      }
    }

    // Quick return if no expressions to substitute
    if (v_id.empty()) return;

    // Collect all expressions to be replaced
    vector<MX> ex;
    ex.insert(ex.end(), this->ode.begin(), this->ode.end());
    ex.insert(ex.end(), this->dae.begin(), this->dae.end());
    ex.insert(ex.end(), this->alg.begin(), this->alg.end());
    ex.insert(ex.end(), this->quad.begin(), this->quad.end());
    ex.insert(ex.end(), this->idef.begin(), this->idef.end());
    ex.insert(ex.end(), this->ydef.begin(), this->ydef.end());
    ex.insert(ex.end(), this->init.begin(), this->init.end());

    // Substitute all at once (more efficient since they may have common subexpressions)
    ex = substitute(ex, v_id, v_rep);

    // Get the modified expressions
    vector<MX>::const_iterator it=ex.begin();
    for (int i=0; i<this->x.size(); ++i) this->ode[i] = *it++ / nominal(this->x[i]);
    for (int i=0; i<this->s.size(); ++i) this->dae[i] = *it++;
    for (int i=0; i<this->z.size(); ++i) this->alg[i] = *it++;
    for (int i=0; i<this->q.size(); ++i) this->quad[i] = *it++ / nominal(this->q[i]);
    for (int i=0; i<this->i.size(); ++i) this->idef[i] = *it++ / nominal(this->i[i]);
    for (int i=0; i<this->y.size(); ++i) this->ydef[i] = *it++ / nominal(this->y[i]);
    for (int i=0; i<this->init.size(); ++i) this->init[i] = *it++;
    casadi_assert(it==ex.end());

    // Nominal value is 1 after scaling
    for (VarMap::iterator it=varmap_.begin(); it!=varmap_.end(); ++it) {
      it->second.nominal=1;
    }
  }

  void DaeBuilder::sort_i() {
    // Quick return if no intermediates
    if (this->i.empty()) return;

    // Find out which intermediates depends on which other
    MXFunction f(vertcat(this->i), vertcat(this->i) - vertcat(this->idef));
    f.init();
    Sparsity sp = f.jacSparsity();
    casadi_assert(sp.isSquare());

    // BLT transformation
    vector<int> rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock;
    sp.dulmageMendelsohn(rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock);

    // Resort equations and variables
    vector<MX> idefnew(this->i.size()), inew(this->i.size());
    for (int i=0; i<colperm.size(); ++i) {
      // Permute equations
      idefnew[i] = this->idef[colperm[i]];

      // Permute variables
      inew[i] = this->i[colperm[i]];
    }
    this->idef = idefnew;
    this->i = inew;
  }

  void DaeBuilder::split_i() {
    // Quick return if no intermediates
    if (this->i.empty()) return;

    // Begin by sorting the intermediate
    sort_i();

    // Sort the equations by causality
    vector<MX> ex;
    substituteInPlace(this->i, this->idef, ex);

    // Make sure that the interdependencies have been properly eliminated
    casadi_assert(!dependsOn(vertcat(this->idef), vertcat(this->i)));
  }

  void DaeBuilder::eliminate_i() {
    // Quick return if possible
    if (this->i.empty()) return;

    // Begin by sorting the intermediate
    sort_i();

    // Collect all expressions to be replaced
    vector<MX> ex;
    ex.insert(ex.end(), this->ode.begin(), this->ode.end());
    ex.insert(ex.end(), this->dae.begin(), this->dae.end());
    ex.insert(ex.end(), this->alg.begin(), this->alg.end());
    ex.insert(ex.end(), this->quad.begin(), this->quad.end());
    ex.insert(ex.end(), this->ydef.begin(), this->ydef.end());
    ex.insert(ex.end(), this->init.begin(), this->init.end());

    // Substitute all at once (since they may have common subexpressions)
    substituteInPlace(this->i, this->idef, ex);

    // Get the modified expressions
    vector<MX>::const_iterator it=ex.begin();
    for (int i=0; i<this->x.size(); ++i) this->ode[i] = *it++;
    for (int i=0; i<this->s.size(); ++i) this->dae[i] = *it++;
    for (int i=0; i<this->z.size(); ++i) this->alg[i] = *it++;
    for (int i=0; i<this->q.size(); ++i) this->quad[i] = *it++;
    for (int i=0; i<this->y.size(); ++i) this->ydef[i] = *it++;
    for (int i=0; i<this->init.size(); ++i) this->init[i] = *it++;
    casadi_assert(it==ex.end());
  }

  void DaeBuilder::scaleEquations() {
    casadi_error("DaeBuilder::scaleEquations broken");
#if 0
    cout << "Scaling equations ..." << endl;
    double time1 = clock();

    // Variables
    enum Variables {T, X, XDOT, Z, P, U, NUM_VAR};
    vector<MX > v(NUM_VAR); // all variables
    v[T] = this->t;
    v[X] = this->x;
    v[XDOT] = der(this->x); // BUG!!!
    v[Z] = this->z;
    v[P] = this->p;
    v[U] = this->u;

    // Create the jacobian of the implicit equations with respect to [x, z, p, u]
    MX xz;
    xz.append(v[X]);
    xz.append(v[Z]);
    xz.append(v[P]);
    xz.append(v[U]);
    MXFunction fcn = MXFunction(xz, this->ode);
    MXFunction J(v, fcn.jac());

    // Evaluate the Jacobian in the starting point
    J.init();
    J.setInput(0.0, T);
    J.setInput(start(this->x, true), X);
    J.input(XDOT).setAll(0.0);
    J.setInput(start(this->z, true), Z);
    J.setInput(start(this->p, true), P);
    J.setInput(start(this->u, true), U);
    J.evaluate();

    // Get the maximum of every row
    Matrix<double> &J0 = J.output();
    vector<double> scale(J0.size1(), 0.0); // scaling factors
    for (int cc=0; cc<J0.size2(); ++cc) {
      // Loop over non-zero entries of the column
      for (int el=J0.colind(cc); el<J0.colind(cc+1); ++el) {
        // Row
        int rr=J0.row(el);

        // The scaling factor is the maximum norm, ignoring not-a-number entries
        if (!isnan(J0.at(el))) {
          scale[rr] = std::max(scale[rr], fabs(J0.at(el)));
        }
      }
    }

    // Make sure nonzero factor found
    for (int rr=0; rr<J0.size1(); ++rr) {
      if (scale[rr]==0) {
        cout << "Warning: Could not generate a scaling factor for equation " << rr;
        scale[rr]=1.;
      }
    }

    // Scale the equations
    this->ode /= scale;

    double time2 = clock();
    double dt = (time2-time1)/CLOCKS_PER_SEC;
    cout << "... equation scaling complete after " << dt << " seconds." << endl;
#endif
  }

  void DaeBuilder::sort_dae() {
    // Quick return if no differential states
    if (this->x.empty()) return;

    // Find out which differential equation depends on which differential state
    MXFunction f(vertcat(this->sdot), vertcat(this->dae));
    f.init();
    Sparsity sp = f.jacSparsity();
    casadi_assert(sp.isSquare());

    // BLT transformation
    vector<int> rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock;
    sp.dulmageMendelsohn(rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock);

    // Resort equations and variables
    vector<MX> daenew(this->s.size()), snew(this->s.size()), sdotnew(this->s.size());
    for (int i=0; i<rowperm.size(); ++i) {
      // Permute equations
      daenew[i] = this->dae[rowperm[i]];

      // Permute variables
      snew[i] = this->s[colperm[i]];
      sdotnew[i] = this->sdot[colperm[i]];
    }
    this->dae = daenew;
    this->s = snew;
    this->sdot = sdotnew;
  }

  void DaeBuilder::sort_alg() {
    // Quick return if no algebraic states
    if (this->z.empty()) return;

    // Find out which algebraic equation depends on which algebraic state
    MXFunction f(vertcat(this->z), vertcat(this->alg));
    f.init();
    Sparsity sp = f.jacSparsity();
    casadi_assert(sp.isSquare());

    // BLT transformation
    vector<int> rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock;
    sp.dulmageMendelsohn(rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock);

    // Resort equations and variables
    vector<MX> algnew(this->z.size()), znew(this->z.size());
    for (int i=0; i<rowperm.size(); ++i) {
      // Permute equations
      algnew[i] = this->alg[rowperm[i]];

      // Permute variables
      znew[i] = this->z[colperm[i]];
    }
    this->alg = algnew;
    this->z = znew;
  }

  void DaeBuilder::makeSemiExplicit() {
    // Only works if there are no i
    eliminate_i();

    // Separate the algebraic variables and equations
    split_dae();

    // Quick return if there are no implicitly defined states
    if (this->s.empty()) return;

    // Write the ODE as a function of the state derivatives
    MXFunction f(vertcat(this->sdot), vertcat(this->dae));
    f.init();

    // Get the sparsity of the Jacobian which can be used to determine which
    // variable can be calculated from which other
    Sparsity sp = f.jacSparsity();
    casadi_assert(sp.isSquare());

    // BLT transformation
    vector<int> rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock;
    int nb = sp.dulmageMendelsohn(rowperm, colperm, rowblock, colblock,
                                  coarse_rowblock, coarse_colblock);

    // Resort equations and variables
    vector<MX> daenew(this->s.size()), snew(this->s.size()), sdotnew(this->s.size());
    for (int i=0; i<rowperm.size(); ++i) {
      // Permute equations
      daenew[i] = this->dae[rowperm[i]];

      // Permute variables
      snew[i] = this->s[colperm[i]];
      sdotnew[i] = this->sdot[colperm[i]];
    }
    this->dae = daenew;
    this->s = snew;
    this->sdot = sdotnew;

    // Now write the sorted ODE as a function of the state derivatives
    f = MXFunction(vertcat(this->sdot), vertcat(this->dae));
    f.init();

    // Get the Jacobian
    MX J = f.jac();

    // Explicit ODE
    vector<MX> new_ode;

    // Loop over blocks
    for (int b=0; b<nb; ++b) {

      // Get variables in the block
      vector<MX> xb(this->s.begin()+colblock[b], this->s.begin()+colblock[b+1]);
      vector<MX> xdotb(this->sdot.begin()+colblock[b], this->sdot.begin()+colblock[b+1]);

      // Get equations in the block
      vector<MX> fb(this->dae.begin()+rowblock[b], this->dae.begin()+rowblock[b+1]);

      // Get local Jacobian
      MX Jb = J(Slice(rowblock[b], rowblock[b+1]), Slice(colblock[b], colblock[b+1]));

      // If Jb depends on xb, then the state derivative does not enter linearly
      // in the ODE and we cannot solve for the state derivative
      casadi_assert_message(!dependsOn(Jb, vertcat(xdotb)),
                            "Cannot find an explicit expression for variable(s) " << xb);

      // Divide fb into a part which depends on vb and a part which doesn't according to
      // "fb == mul(Jb, vb) + fb_res"
      vector<MX> fb_res = substitute(fb, xdotb, vector<MX>(xdotb.size(), 0));

      // Solve for vb
      vector<MX> fb_exp = vertsplit(solve(Jb, -vertcat(fb_res)));

      // Add to explicitly determined equations and variables
      new_ode.insert(new_ode.end(), fb_exp.begin(), fb_exp.end());
    }

    // Eliminate inter-dependencies
    vector<MX> ex;
    substituteInPlace(this->sdot, new_ode, ex, false);

    // Add to explicit differential states and ODE
    this->x.insert(this->x.end(), this->s.begin(), this->s.end());
    this->ode.insert(this->ode.end(), new_ode.begin(), new_ode.end());
    this->dae.clear();
    this->s.clear();
    this->sdot.clear();
  }

  void DaeBuilder::eliminate_alg() {
    // Only works if there are no i
    eliminate_i();

    // Quick return if there are no algebraic states
    if (this->z.empty()) return;

    // Write the algebraic equations as a function of the algebraic states
    MXFunction f(vertcat(this->z), vertcat(this->alg));
    f.init();

    // Get the sparsity of the Jacobian which can be used to determine which
    // variable can be calculated from which other
    Sparsity sp = f.jacSparsity();
    casadi_assert(sp.isSquare());

    // BLT transformation
    vector<int> rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock;
    int nb = sp.dulmageMendelsohn(rowperm, colperm, rowblock, colblock,
                                  coarse_rowblock, coarse_colblock);

    // Resort equations and variables
    vector<MX> algnew(this->z.size()), znew(this->z.size());
    for (int i=0; i<rowperm.size(); ++i) {
      // Permute equations
      algnew[i] = this->alg[rowperm[i]];

      // Permute variables
      znew[i] = this->z[colperm[i]];
    }
    this->alg = algnew;
    this->z = znew;

    // Rewrite the sorted algebraic equations as a function of the algebraic states
    f = MXFunction(vertcat(this->z), vertcat(this->alg));
    f.init();

    // Variables where we have found an explicit expression and where we haven't
    vector<MX> z_exp, z_imp;

    // Explicit and implicit equations
    vector<MX> f_exp, f_imp;

    // Loop over blocks
    for (int b=0; b<nb; ++b) {

      // Get local variables
      vector<MX> zb(this->z.begin()+colblock[b], this->z.begin()+colblock[b+1]);

      // Get local equations
      vector<MX> fb(this->alg.begin()+rowblock[b], this->alg.begin()+rowblock[b+1]);

      // Get local Jacobian
      MX Jb = jacobian(vertcat(fb), vertcat(zb));

      // If Jb depends on zb, then we cannot (currently) solve for it explicitly
      if (dependsOn(Jb, vertcat(zb))) {

        // Add the equations to the new list of algebraic equations
        f_imp.insert(f_imp.end(), fb.begin(), fb.end());

        // ... and the variables accordingly
        z_imp.insert(z_imp.end(), zb.begin(), zb.end());

      } else { // The variables that we wish to determine enter linearly

        // Divide fb into a part which depends on vb and a part which doesn't
        // according to "fb == mul(Jb, vb) + fb_res"
        vector<MX> fb_res = substitute(fb, zb, vector<MX>(zb.size(), 0));

        // Solve for vb
        vector<MX> fb_exp = vertsplit(solve(Jb, -vertcat(fb_res)));

        // Add to explicitly determined equations and variables
        z_exp.insert(z_exp.end(), zb.begin(), zb.end());
        f_exp.insert(f_exp.end(), fb_exp.begin(), fb_exp.end());
      }
    }

    // Eliminate inter-dependencies in fb_exp
    vector<MX> ex;
    substituteInPlace(z_exp, f_exp, ex, false);

    // Add to the beginning of the dependent variables
    // (since the other dependent variable might depend on them)
    this->i.insert(this->i.begin(), z_exp.begin(), z_exp.end());
    this->idef.insert(this->idef.begin(), f_exp.begin(), f_exp.end());

    // Save new algebraic equations
    this->z = z_imp;
    this->alg = f_imp;

    // Eliminate new dependent variables from the other equations
    eliminate_i();
  }

  void DaeBuilder::makeExplicit() {
    // Only works if there are no i
    eliminate_i();

    // Start by transforming to semi-explicit form
    makeSemiExplicit();

    // Then eliminate the algebraic variables
    eliminate_alg();

    // Error if still algebraic variables
    casadi_assert_message(this->z.empty(), "Failed to eliminate algebraic variables");
  }

  const Variable& DaeBuilder::variable(const std::string& name) const {
    return const_cast<DaeBuilder*>(this)->variable(name);
  }

  Variable& DaeBuilder::variable(const std::string& name) {
    // Find the variable
    VarMap::iterator it = varmap_.find(name);
    if (it==varmap_.end()) {
      casadi_error("No such variable: \"" << name << "\".");
    }

    // Return the variable
    return it->second;
  }

  void DaeBuilder::addVariable(const std::string& name, const Variable& var) {
    // Try to find the component
    if (varmap_.find(name)!=varmap_.end()) {
      stringstream ss;
      casadi_error("Variable \"" << name << "\" has already been added.");
    }

    // Add to the map of all variables
    varmap_[name] = var;
  }

  MX DaeBuilder::addVariable(const std::string& name) {
    Variable v(name);
    addVariable(name, v);
    return v.v;
  }

  MX DaeBuilder::add_x(const std::string& name) {
    if (name.empty()) // Generate a name
      return add_x("x" + CodeGenerator::to_string(this->x.size()));
    MX new_x = addVariable(name);
    this->x.push_back(new_x);
    return new_x;
  }

  MX DaeBuilder::add_q(const std::string& name) {
    if (name.empty()) // Generate a name
      return add_q("q" + CodeGenerator::to_string(this->q.size()));
    MX new_q = addVariable(name);
    this->q.push_back(new_q);
    return new_q;
  }

  std::pair<MX, MX> DaeBuilder::add_s(const std::string& name) {
    if (name.empty()) // Generate a name
      return add_s("s" + CodeGenerator::to_string(this->s.size()));
    Variable v(name);
    addVariable(name, v);
    this->s.push_back(v.v);
    this->sdot.push_back(v.d);
    return std::pair<MX, MX>(v.v, v.d);
  }

  MX DaeBuilder::add_z(const std::string& name) {
    if (name.empty()) // Generate a name
      return add_z("z" + CodeGenerator::to_string(this->z.size()));
    MX new_z = addVariable(name);
    this->z.push_back(new_z);
    return new_z;
  }

  MX DaeBuilder::add_p(const std::string& name) {
    if (name.empty()) // Generate a name
      return add_p("p" + CodeGenerator::to_string(this->p.size()));
    MX new_p = addVariable(name);
    this->p.push_back(new_p);
    return new_p;
  }

  MX DaeBuilder::add_u(const std::string& name) {
    if (name.empty()) // Generate a name
      return add_u("u" + CodeGenerator::to_string(this->u.size()));
    MX new_u = addVariable(name);
    this->u.push_back(new_u);
    return new_u;
  }

  MX DaeBuilder::add_y(const std::string& name) {
    if (name.empty()) // Generate a name
      return add_y("y" + CodeGenerator::to_string(this->y.size()));
    MX new_y = addVariable(name);
    this->y.push_back(new_y);
    return new_y;
  }

  MX DaeBuilder::add_i(const std::string& name) {
    if (name.empty()) // Generate a name
      return add_i("i" + CodeGenerator::to_string(this->i.size()));
    MX new_i = addVariable(name);
    this->i.push_back(new_i);
    return new_i;
  }

  void DaeBuilder::add_ode(const MX& new_ode, const std::string& name) {
    if (name.empty()) // Generate a name
      return add_ode(new_ode, "ode" + CodeGenerator::to_string(this->ode.size()));
    this->ode.push_back(new_ode);
    this->lam_ode.push_back(MX::sym("lam_" + name, new_ode.sparsity()));
  }

  void DaeBuilder::add_dae(const MX& new_dae, const std::string& name) {
    if (name.empty()) // Generate a name
      return add_dae(new_dae, "dae" + CodeGenerator::to_string(this->dae.size()));
    this->dae.push_back(new_dae);
    this->lam_dae.push_back(MX::sym("lam_" + name, new_dae.sparsity()));
  }

  void DaeBuilder::add_alg(const MX& new_alg, const std::string& name) {
    if (name.empty()) // Generate a name
      return add_alg(new_alg, "alg" + CodeGenerator::to_string(this->alg.size()));
    this->alg.push_back(new_alg);
    this->lam_alg.push_back(MX::sym("lam_" + name, new_alg.sparsity()));
  }

  void DaeBuilder::add_quad(const MX& new_quad, const std::string& name) {
    if (name.empty()) // Generate a name
      return add_quad(new_quad, "quad" + CodeGenerator::to_string(this->quad.size()));
    this->quad.push_back(new_quad);
    this->lam_quad.push_back(MX::sym("lam_" + name, new_quad.sparsity()));
  }

  void DaeBuilder::add_idef(const MX& new_idef, const std::string& name) {
    if (name.empty()) // Generate a name
      return add_idef(new_idef, "idef" + CodeGenerator::to_string(this->idef.size()));
    this->idef.push_back(new_idef);
    this->lam_idef.push_back(MX::sym("lam_" + name, new_idef.sparsity()));
  }

  void DaeBuilder::add_ydef(const MX& new_ydef, const std::string& name) {
    if (name.empty()) // Generate a name
      return add_ydef(new_ydef, "ydef" + CodeGenerator::to_string(this->ydef.size()));
    this->ydef.push_back(new_ydef);
    this->lam_ydef.push_back(MX::sym("lam_" + name, new_ydef.sparsity()));
  }

  void DaeBuilder::sanityCheck() const {
    // Time
    casadi_assert_message(this->t.isSymbolic(), "Non-symbolic time t");
    casadi_assert_message(this->t.isScalar(), "Non-scalar time t");

    // Differential states
    casadi_assert_message(this->x.size()==this->ode.size(),
                          "x and ode have different lengths");
    for (int i=0; i<this->x.size(); ++i) {
      casadi_assert_message(this->x[i].shape()==this->ode[i].shape(),
                            "ode has wrong dimensions");
      casadi_assert_message(this->x[i].isSymbolic(), "Non-symbolic state x");
    }

    // DAE
    casadi_assert_message(this->s.size()==this->sdot.size(),
                          "s and sdot have different lengths");
    casadi_assert_message(this->s.size()==this->dae.size(),
                          "s and dae have different lengths");
    for (int i=0; i<this->s.size(); ++i) {
      casadi_assert_message(this->s[i].isSymbolic(), "Non-symbolic state s");
      casadi_assert_message(this->s[i].shape()==this->sdot[i].shape(),
                            "sdot has wrong dimensions");
      casadi_assert_message(this->s[i].shape()==this->dae[i].shape(),
                            "dae has wrong dimensions");
    }

    // Algebraic variables/equations
    casadi_assert_message(this->z.size()==this->alg.size(),
                          "z and alg have different lengths");
    for (int i=0; i<this->z.size(); ++i) {
      casadi_assert_message(this->z[i].isSymbolic(), "Non-symbolic algebraic variable z");
      casadi_assert_message(this->z[i].shape()==this->alg[i].shape(),
                            "alg has wrong dimensions");
    }

    // Quadrature states/equations
    casadi_assert_message(this->q.size()==this->quad.size(),
                          "q and quad have different lengths");
    for (int i=0; i<this->q.size(); ++i) {
      casadi_assert_message(this->q[i].isSymbolic(), "Non-symbolic quadrature state q");
      casadi_assert_message(this->q[i].shape()==this->quad[i].shape(),
                            "quad has wrong dimensions");
    }

    // Intermediate variables
    casadi_assert_message(this->i.size()==this->idef.size(),
                          "i and idef have different lengths");
    for (int i=0; i<this->i.size(); ++i) {
      casadi_assert_message(this->i[i].isSymbolic(), "Non-symbolic intermediate variable i");
      casadi_assert_message(this->i[i].shape()==this->idef[i].shape(),
                            "idef has wrong dimensions");
    }

    // Output equations
    casadi_assert_message(this->y.size()==this->ydef.size(),
                          "y and ydef have different lengths");
    for (int i=0; i<this->y.size(); ++i) {
      casadi_assert_message(this->y[i].isSymbolic(), "Non-symbolic output y");
      casadi_assert_message(this->y[i].shape()==this->ydef[i].shape(),
                            "ydef has wrong dimensions");
    }

    // Control
    for (int i=0; i<this->u.size(); ++i) {
      casadi_assert_message(this->u[i].isSymbolic(), "Non-symbolic control u");
    }

    // Parameter
    for (int i=0; i<this->p.size(); ++i) {
      casadi_assert_message(this->p[i].isSymbolic(), "Non-symbolic parameter p");
    }
  }

  std::string DaeBuilder::qualifiedName(const XmlNode& nn) {
    // Stringstream to assemble name
    stringstream qn;

    for (int i=0; i<nn.size(); ++i) {
      // Add a dot
      if (i!=0) qn << ".";

      // Get the name part
      qn << nn[i].getAttribute("name");

      // Get the index, if any
      if (nn[i].size()>0) {
        int ind;
        nn[i]["exp:ArraySubscripts"]["exp:IndexExpression"]["exp:IntegerLiteral"].getText(ind);
        qn << "[" << ind << "]";
      }
    }

    // Return the name
    return qn.str();
  }

  MX DaeBuilder::operator()(const std::string& name) const {
    return variable(name).v;
  }

  MX DaeBuilder::der(const std::string& name) const {
    return variable(name).d;
  }

  MX DaeBuilder::der(const MX& var) const {
    casadi_assert(var.isVector() && var.isSymbolic());
    MX ret = MX::zeros(var.sparsity());
    for (int i=0; i<ret.nnz(); ++i) {
      ret[i] = der(var.at(i).getName());
    }
    return ret;
  }

  void DaeBuilder::split_dae() {
    // Only works if there are no i
    eliminate_i();

    // Quick return if no s
    if (this->s.empty()) return;

    // We investigate the interdependencies in sdot -> dae
    vector<MX> f_in;
    f_in.push_back(vertcat(this->sdot));
    MXFunction f(f_in, vertcat(this->dae));
    f.init();

    // Number of s
    int ns = f.input().nnz();
    casadi_assert(f.output().nnz()==ns);

    // Input/output arrays
    bvec_t* f_sdot = reinterpret_cast<bvec_t*>(f.input().ptr());
    bvec_t* f_dae = reinterpret_cast<bvec_t*>(f.output().ptr());

    // First find out which equations depend on sdot
    f.spInit(true);

    // Seed all inputs
    std::fill(f_sdot, f_sdot+ns, bvec_t(1));

    // Propagate to f_dae
    std::fill(f_dae, f_dae+ns, bvec_t(0));
    f.spEvaluate(true);

    // Get the new differential and algebraic equations
    vector<MX> new_dae, new_alg;
    for (int i=0; i<ns; ++i) {
      if (f_dae[i]==bvec_t(1)) {
        new_dae.push_back(this->dae[i]);
      } else {
        casadi_assert(f_dae[i]==bvec_t(0));
        new_alg.push_back(this->dae[i]);
      }
    }

    // Now find out what sdot enter in the equations
    f.spInit(false);

    // Seed all outputs
    std::fill(f_dae, f_dae+ns, bvec_t(1));

    // Propagate to f_sdot
    std::fill(f_sdot, f_sdot+ns, bvec_t(0));
    f.spEvaluate(false);

    // Get the new algebraic variables and new states
    vector<MX> new_s, new_sdot, new_z;
    for (int i=0; i<ns; ++i) {
      if (f_sdot[i]==bvec_t(1)) {
        new_s.push_back(this->s[i]);
        new_sdot.push_back(this->sdot[i]);
      } else {
        casadi_assert(f_sdot[i]==bvec_t(0));
        new_z.push_back(this->s[i]);
      }
    }

    // Make sure split was successful
    casadi_assert(new_dae.size()==new_s.size());

    // Divide up the s and dae
    this->dae = new_dae;
    this->s = new_s;
    this->sdot = new_sdot;
    this->alg.insert(this->alg.end(), new_alg.begin(), new_alg.end());
    this->z.insert(this->z.end(), new_z.begin(), new_z.end());
  }

  std::string DaeBuilder::unit(const std::string& name) const {
    return variable(name).unit;
  }

  std::string DaeBuilder::unit(const MX& var) const {
    casadi_assert_message(!var.isVector() && var.isValidInput(),
                          "DaeBuilder::unit: Argument must be a symbolic vector");
    if (var.isEmpty()) {
      return "n/a";
    } else {
      std::vector<MX> prim = var.getPrimitives();
      string ret = unit(prim.at(0).getName());
      for (int i=1; i<prim.size(); ++i) {
        casadi_assert_message(ret == unit(prim.at(i).getName()),
                              "DaeBuilder::unit: Argument has mixed units");
      }
      return ret;
    }
  }

  void DaeBuilder::setUnit(const std::string& name, const std::string& val) {
    variable(name).unit = val;
  }

  double DaeBuilder::nominal(const std::string& name) const {
    return variable(name).nominal;
  }

  void DaeBuilder::setNominal(const std::string& name, double val) {
    variable(name).nominal = val;
  }

  std::vector<double> DaeBuilder::nominal(const MX& var) const {
    casadi_assert_message(var.isVector() && var.isValidInput(),
                          "DaeBuilder::nominal: Argument must be a symbolic vector");
    std::vector<double> ret(var.nnz());
    std::vector<MX> prim = var.getPrimitives();
    for (int i=0; i<prim.size(); ++i) {
      casadi_assert(prim[i].nnz()==1);
      ret[i] = nominal(prim.at(i).getName());
    }
    return ret;
  }

  void DaeBuilder::setNominal(const MX& var, const std::vector<double>& val) {
    casadi_assert_message(var.isVector() && var.isValidInput(),
                          "DaeBuilder::nominal: Argument must be a symbolic vector");
    casadi_assert_message(var.nnz()==var.nnz(), "DaeBuilder::nominal: Dimension mismatch");
    std::vector<MX> prim = var.getPrimitives();
    for (int i=0; i<prim.size(); ++i) {
      casadi_assert(prim[i].nnz()==1);
      setNominal(prim.at(i).getName(), val.at(i));
    }
  }

  std::vector<double> DaeBuilder::attribute(getAtt f, const MX& var, bool normalized) const {
    casadi_assert_message(var.isVector() && var.isValidInput(),
                          "DaeBuilder::attribute: Argument must be a symbolic vector");
    std::vector<double> ret(var.nnz());
    std::vector<MX> prim = var.getPrimitives();
    for (int i=0; i<prim.size(); ++i) {
      casadi_assert(prim[i].nnz()==1);
      ret[i] = (this->*f)(prim[i].getName(), normalized);
    }
    return ret;
  }

  MX DaeBuilder::attribute(getAttS f, const MX& var) const {
    casadi_assert_message(var.isVector() && var.isValidInput(),
                          "DaeBuilder::attribute: Argument must be a symbolic vector");
    MX ret = MX::zeros(var.sparsity());
    std::vector<MX> prim = var.getPrimitives();
    for (int i=0; i<prim.size(); ++i) {
      casadi_assert(prim[i].nnz()==1);
      ret[i] = (this->*f)(prim[i].getName());
    }
    return ret;
  }

  void DaeBuilder::setAttribute(setAtt f, const MX& var, const std::vector<double>& val,
                                 bool normalized) {
    casadi_assert_message(var.isVector() && var.isValidInput(),
                          "DaeBuilder::setAttribute: Argument must be a symbolic vector");
    casadi_assert_message(var.nnz()==val.size(), "DaeBuilder::setAttribute: Dimension mismatch");
    std::vector<MX> prim = var.getPrimitives();
    for (int i=0; i<prim.size(); ++i) {
      casadi_assert(prim[i].nnz()==1);
      (this->*f)(prim[i].getName(), val[i], normalized);
    }
  }

  void DaeBuilder::setAttribute(setAttS f, const MX& var, const MX& val) {
    casadi_assert_message(var.isVector() && var.isValidInput(),
                          "DaeBuilder::setAttribute: Argument must be a symbolic vector");
    casadi_assert_message(var.sparsity()==val.sparsity(),
                          "DaeBuilder::setAttribute: Sparsity mismatch");
    std::vector<MX> prim = var.getPrimitives();
    for (int i=0; i<prim.size(); ++i) {
      casadi_assert(prim[i].nnz()==1);
      (this->*f)(var[i].getName(), val[i]);
    }
  }

  double DaeBuilder::min(const std::string& name, bool normalized) const {
    const Variable& v = variable(name);
    return normalized ? v.min / v.nominal : v.min;
  }

  std::vector<double> DaeBuilder::min(const MX& var, bool normalized) const {
    return attribute(&DaeBuilder::min, var, normalized);
  }

  void DaeBuilder::setMin(const std::string& name, double val, bool normalized) {
    Variable& v = variable(name);
    v.min = normalized ? val*v.nominal : val;
  }

  void DaeBuilder::setMin(const MX& var, const std::vector<double>& val, bool normalized) {
    setAttribute(&DaeBuilder::setMin, var, val, normalized);
  }

  double DaeBuilder::max(const std::string& name, bool normalized) const {
    const Variable& v = variable(name);
    return normalized ? v.max / v.nominal : v.max;
  }

  std::vector<double> DaeBuilder::max(const MX& var, bool normalized) const {
    return attribute(&DaeBuilder::max, var, normalized);
  }

  void DaeBuilder::setMax(const std::string& name, double val, bool normalized) {
    Variable& v = variable(name);
    v.max = normalized ? val*v.nominal : val;
  }

  void DaeBuilder::setMax(const MX& var, const std::vector<double>& val, bool normalized) {
    setAttribute(&DaeBuilder::setMax, var, val, normalized);
  }

  double DaeBuilder::initialGuess(const std::string& name, bool normalized) const {
    const Variable& v = variable(name);
    return normalized ? v.initialGuess / v.nominal : v.initialGuess;
  }

  std::vector<double> DaeBuilder::initialGuess(const MX& var, bool normalized) const {
    return attribute(&DaeBuilder::initialGuess, var, normalized);
  }

  void DaeBuilder::setInitialGuess(const std::string& name, double val, bool normalized) {
    Variable& v = variable(name);
    v.initialGuess = normalized ? val*v.nominal : val;
  }

  void DaeBuilder::setInitialGuess(const MX& var, const std::vector<double>& val,
                                    bool normalized) {
    setAttribute(&DaeBuilder::setInitialGuess, var, val, normalized);
  }

  double DaeBuilder::start(const std::string& name, bool normalized) const {
    const Variable& v = variable(name);
    return normalized ? v.start / v.nominal : v.start;
  }

  std::vector<double> DaeBuilder::start(const MX& var, bool normalized) const {
    return attribute(&DaeBuilder::start, var, normalized);
  }

  void DaeBuilder::setStart(const std::string& name, double val, bool normalized) {
    Variable& v = variable(name);
    v.start = normalized ? val*v.nominal : val;
  }

  void DaeBuilder::setStart(const MX& var, const std::vector<double>& val, bool normalized) {
    setAttribute(&DaeBuilder::setStart, var, val, normalized);
  }

  double DaeBuilder::derivativeStart(const std::string& name, bool normalized) const {
    const Variable& v = variable(name);
    return normalized ? v.derivativeStart / v.nominal : v.derivativeStart;
  }

  std::vector<double> DaeBuilder::derivativeStart(const MX& var, bool normalized) const {
    return attribute(&DaeBuilder::derivativeStart, var, normalized);
  }

  void DaeBuilder::setDerivativeStart(const std::string& name, double val, bool normalized) {
    Variable& v = variable(name);
    v.derivativeStart = normalized ? val*v.nominal : val;
  }

  void DaeBuilder::setDerivativeStart(const MX& var, const std::vector<double>& val,
                                       bool normalized) {
    setAttribute(&DaeBuilder::setDerivativeStart, var, val, normalized);
  }

  void DaeBuilder::generateFunctionHeader(std::ostream &stream, const std::string& fname,
                                           bool fwd, bool adj, bool foa) {
    stream << "void " << fname << "_work(int *ni, int *nr);" << endl;
    stream << "int " << fname
           << "(const double* const* arg, double* const* res, int* iw, double* w);" << endl;

    // Codegen derivative information
    if (fwd) generateFunctionHeader(stream, fname+"_fwd");
    if (adj) generateFunctionHeader(stream, fname+"_adj");
    if (foa) generateFunctionHeader(stream, fname+"_foa");
  }

  void DaeBuilder::generateHeader(const std::string& filename, const std::string& prefix) {
    // Create header file
    ofstream s;
    s.open(filename.c_str());

    // Print header
    s << "/* This file was automatically generated by CasADi */" << endl << endl;

    // Include guards
    s << "#ifndef " << prefix << "IVP_HEADER_FILE" << endl;
    s << "#define "<< prefix << "IVP_HEADER_FILE" << endl << endl;

    // C linkage
    s << "#ifdef __cplusplus" << endl;
    s << "extern \"C\" {" << endl;
    s << "#endif" << endl << endl;

    // Typedef input enum corresponding to generated format
    s << "/* Input convension */" << endl;
    s << "typedef enum {"
      << "IVP_T, "
      << "IVP_X, "
      << "IVP_S, "
      << "IVP_SDOT, "
      << "IVP_Z, "
      << "IVP_U, "
      << "IVP_Q, "
      << "IVP_I, "
      << "IVP_Y, "
      << "IVP_P, "
      << "IVP_NUM_IN} " << prefix << "ivp_input_t;" << endl << endl;

    // Typedef input enum corresponding to generated format
    s << "/* Output convension */" << endl;
    s << "typedef enum {"
      << "IVP_ODE, "
      << "IVP_DAE, "
      << "IVP_ALG, "
      << "IVP_QUAD, "
      << "IVP_IDEF, "
      << "IVP_YDEF, "
      << "IVP_NUM_OUT} " << prefix << "ivp_output_t;" << endl << endl;

    // Input dimensions and offset
    s << "/* Input dimensions and offsets */" << endl
      << "void " << prefix
      << "input_dims(const int **dims, const int **offset, const int **inv_offset);" << endl
      << endl;

    // Output dimensions and offset
    s << "/* Output dimensions and offsets */" << endl
      << "void " << prefix
      << "output_dims(const int **dims, const int **offset, const int **inv_offset);" << endl
      << endl;

    s << "/* Function declarations */" << endl;
    generateFunctionHeader(s, prefix+"eval_ode");
    generateFunctionHeader(s, prefix+"eval_dae");
    generateFunctionHeader(s, prefix+"eval_alg");
    generateFunctionHeader(s, prefix+"eval_quad");
    generateFunctionHeader(s, prefix+"eval_idef");
    generateFunctionHeader(s, prefix+"eval_ydef");
    generateFunctionHeader(s, prefix+"eval", true, true, true);
    s << endl;

    s << "/* Jacobian of all outputs w.r.t. all inputs */" << endl;
    generateFunctionHeader(s, prefix+"eval_jac");
    s << "void " << prefix <<
      "jac_sparsity(int *nrow, int *ncol, const int **colind, const int **row);" << endl;
    s << endl;

    s << "/* Hessian of all outputs w.r.t. all inputs */" << endl;
    generateFunctionHeader(s, prefix+"eval_hes");
    s << "void " << prefix <<
      "hes_sparsity(int *nrow, int *ncol, const int **colind, const int **row);" << endl;
    s << endl;

    // C linkage
    s << "#ifdef __cplusplus" << endl;
    s << "} /* extern \"C\" */" << endl;
    s << "#endif" << endl << endl;

    // Include guards
    s << "#endif /* " << prefix << "IVP_HEADER_FILE */" << endl << endl;

    s.close();
  }

  void DaeBuilder::generateFunction(const std::string& fname,
                                     const std::vector<MX>& f_in,
                                     const std::vector<MX>& f_out,
                                     CodeGenerator& g,
                                     bool fwd, bool adj, bool foa) {
    MXFunction f(f_in, f_out);
    f.setOption("name", fname);
    f.init();
    g.add(f, fname);
    size_t ni, nr;
    f.nTmp(ni, nr);

    // Forward mode directional derivative
    if (fwd) {
      MXFunction f_fwd = shared_cast<MXFunction>(f.derForward(1));
      generateFunction(fname+"_fwd",
                       f_fwd.inputExpr(), f_fwd.outputExpr(), g);
    }

    // Reverse mode mode directional derivative
    if (adj || foa) {
      MXFunction f_adj = shared_cast<MXFunction>(f.derReverse(1));
      if (adj) {
        generateFunction(fname+"_adj",
                         f_adj.inputExpr(), f_adj.outputExpr(), g);
      }
      // Forward-over-reverse mode directional derivative
      if (foa) {
        MXFunction f_foa = shared_cast<MXFunction>(f_adj.derForward(1));
        generateFunction(fname+"_foa",
                         f_foa.inputExpr(), f_foa.outputExpr(), g);
      }
    }
  }

  void DaeBuilder::generateCode(const std::string& filename,
                                 const Dictionary& options) {
    // Create a code generator object
    CodeGenerator g(options);
    if (!g.include.empty()) {
      g.addInclude(g.include, true);
    }

    // All inputs
    vector<MX> v_in;
    v_in.push_back(this->t);
    v_in.push_back(vertcat(this->x));
    v_in.push_back(vertcat(this->s));
    v_in.push_back(vertcat(this->sdot));
    v_in.push_back(vertcat(this->z));
    v_in.push_back(vertcat(this->u));
    v_in.push_back(vertcat(this->q));
    v_in.push_back(vertcat(this->i));
    v_in.push_back(vertcat(this->y));
    v_in.push_back(vertcat(this->p));

    // Input dimensions
    vector<int> dims(v_in.size());
    for (int i=0; i<v_in.size(); ++i) {
      dims[i] = v_in[i].nnz();
    }
    int dims_ind = g.getConstant(dims, true);

    // Corresponding offsets
    dims.insert(dims.begin(), 0);
    vector<int> inv_offset;
    for (int i=0; i<v_in.size(); ++i) {
      dims[i+1] += dims[i]; // cumsum
      inv_offset.resize(dims[i+1], i);
    }
    int offset_ind = g.getConstant(dims, true);
    int inv_offset_ind = g.getConstant(inv_offset, true);
    g.body
      << "void " << g.prefix
      << "input_dims(const int **dims, const int **offset, const int **inv_offset) {" << endl
      << "  if (dims) *dims = s" << dims_ind << ";" << endl
      << "  if (offset) *offset = s" << offset_ind << ";" << endl
      << "  if (inv_offset) *inv_offset = s" << inv_offset_ind << ";" << endl
      << "}" << endl << endl;

    // All outputs
    vector<MX> v_out;
    v_out.push_back(vertcat(this->ode));
    v_out.push_back(vertcat(this->dae));
    v_out.push_back(vertcat(this->alg));
    v_out.push_back(vertcat(this->quad));
    v_out.push_back(vertcat(this->idef));
    v_out.push_back(vertcat(this->ydef));

    // Output dimensions
    dims.resize(v_out.size());
    for (int i=0; i<v_out.size(); ++i) {
      dims[i] = v_out[i].nnz();
    }
    dims_ind = g.getConstant(dims, true);

    // Corresponding offsets
    dims.insert(dims.begin(), 0);
    inv_offset.clear();
    for (int i=0; i<v_out.size(); ++i) {
      dims[i+1] += dims[i]; // cumsum
      inv_offset.resize(dims[i+1], i);
    }
    offset_ind = g.getConstant(dims, true);
    inv_offset_ind = g.getConstant(inv_offset, true);
    g.body
      << "void " << g.prefix
      << "output_dims(const int **dims, const int **offset, const int **inv_offset) {" << endl
      << "  if (dims) *dims = s" << dims_ind << ";" << endl
      << "  if (offset) *offset = s" << offset_ind << ";" << endl
      << "  if (inv_offset) *inv_offset = s" << inv_offset_ind << ";" << endl
      << "}" << endl << endl;

    // Basic functions individually
    generateFunction(g.prefix+"eval_ode", v_in,
                     vector<MX>(1, vertcat(this->ode)), g);
    generateFunction(g.prefix+"eval_dae", v_in,
                     vector<MX>(1, vertcat(this->dae)), g);
    generateFunction(g.prefix+"eval_alg", v_in,
                     vector<MX>(1, vertcat(this->alg)), g);
    generateFunction(g.prefix+"eval_quad", v_in,
                     vector<MX>(1, vertcat(this->quad)), g);
    generateFunction(g.prefix+"eval_idef", v_in,
                     vector<MX>(1, vertcat(this->idef)), g);
    generateFunction(g.prefix+"eval_ydef", v_in,
                     vector<MX>(1, vertcat(this->ydef)), g);

    // All functions at once, with derivatives
    generateFunction(g.prefix+"eval", v_in, v_out, g, true, true, true);

    // Jacobian of all input w.r.t. all outputs
    MX v_in_all = vertcat(v_in);
    MX v_out_all = vertcat(v_out);
    MX J = jacobian(v_out_all, v_in_all);

    // Codegen it
    generateFunction(g.prefix+"eval_jac", v_in, vector<MX>(1, J), g);
    int Jsp_ind = g.addSparsity(J.sparsity());
    g.body
      << "void " << g.prefix
      << "jac_sparsity(int *nrow, int *ncol, const int **colind, const int **row) {" << endl
      << "  const int *s = s" << Jsp_ind << ";" << endl
      << "  if (nrow) *nrow = s[0];" << endl
      << "  if (ncol) *ncol = s[1];" << endl
      << "  if (colind) *colind = s+2;" << endl
      << "  if (row) *row = s+2+s[1]+1;" << endl
      << "}" << endl << endl;

    // Introduce lagrange multipliers
    vector<MX> lam;
    lam.insert(lam.end(), this->lam_ode.begin(), this->lam_ode.end());
    lam.insert(lam.end(), this->lam_dae.begin(), this->lam_dae.end());
    lam.insert(lam.end(), this->lam_alg.begin(), this->lam_alg.end());
    lam.insert(lam.end(), this->lam_quad.begin(), this->lam_quad.end());
    lam.insert(lam.end(), this->lam_idef.begin(), this->lam_idef.end());
    lam.insert(lam.end(), this->lam_ydef.begin(), this->lam_ydef.end());

    // Jacobian of all input w.r.t. all outputs
    MX lam_all = vertcat(lam);
    MX gamma = inner_prod(v_out_all, lam_all);
    MX H = hessian(gamma, v_in_all);
    H = triu(H); // Upper triangular half
    v_in.insert(v_in.begin(), lam.begin(), lam.end());

    // Codegen it
    generateFunction(g.prefix+"eval_hes", v_in, vector<MX>(1, H), g);
    int Hsp_ind = g.addSparsity(H.sparsity());
    g.body
      << "void " << g.prefix
      << "hes_sparsity(int *nrow, int *ncol, const int **colind, const int **row) {" << endl
      << "  const int *s = s" << Hsp_ind << ";" << endl
      << "  if (nrow) *nrow = s[0];" << endl
      << "  if (ncol) *ncol = s[1];" << endl
      << "  if (colind) *colind = s+2;" << endl
      << "  if (row) *row = s+2+s[1]+1;" << endl
      << "}" << endl << endl;

    // Flush to a file
    g.generate(filename);
  }

} // namespace casadi
