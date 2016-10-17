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


#include <iostream>
#include <fstream>
#include <ctime>
#include <casadi/casadi.hpp>

using namespace casadi;
using namespace std;

class MyCallback : public Callback {
 private:
   // Data members
  
 public:
   // Creator function, creates an owning reference
   static Function create(const std::string& name, const Dict& opts=Dict()) {
     return Callback::create(name, new MyCallback(), opts);
   }

   // Initialize the object
   virtual void init() {
     std::cout << "initializing object" << std::endl;
   }

   // Number of inputs and outputs
   virtual int get_n_in() { return 2;}
   virtual int get_n_out() { return 2;}

   // Evaluate numerically
   virtual std::vector<DM> eval(std::vector<DM>& arg) {
    double x = arg.at(0).scalar();
    double y = arg.at(1).scalar();
    return {x*2,y*2};
   }
   
   virtual bool has_jacobian() const { return true; }
   virtual Function get_jacobian(const std::string& name, const Dict& opts) {
    SX x = SX::sym("x");
    SX y = SX::sym("y");
    return Function('f',{x,y},{DM::zeros(2,2)});
   }

 };

int main()
{

	
	MX X = MX::sym("X",2,1);

  Function f = MyCallback::create("f");
  vector<MX> arg={X[0],X[1]};
  std::vector<MX> res = f(arg);
  MX driveSpline=0.01*(pow(X[0]-res.at(0),2)+pow(X[1]-res.at(0),2));
  MX J = driveSpline;

	// NLP 
	MXDict nlp = {{"x", X}, {"f", J}};

	// Set options
	Dict opts;
	opts["ipopt.tol"] = 1e-5;
	opts["ipopt.max_iter"] = 200;
	//opts["ipopt.linear_solver"] = "ma27";

	// Create an NLP solver and buffers
	Function solver = nlpsol("nlpsol", "ipopt", nlp, opts);


}
