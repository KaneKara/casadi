
// File: index.xml

// File: classcasadi_1_1Assertion.xml


// File: classcasadi_1_1Bilin.xml


// File: classcasadi_1_1BinaryMX.xml


// File: classcasadi_1_1BinarySX.xml


// File: classcasadi_1_1Call.xml


// File: classcasadi_1_1Callback.xml
%feature("docstring") casadi::Callback::size2_in "

Get input dimension.

";

%feature("docstring") casadi::Callback::call "

Evaluate the function symbolically or numerically.

";

%feature("docstring") casadi::Callback::generate_dependencies "

Export / Generate C code for the dependency function.

";

%feature("docstring") casadi::Callback::get_n_in "

Get the number of inputs This function is called during construction.

";

%feature("docstring") casadi::Callback::size_in "

Get input dimension.

";

%feature("docstring") casadi::Callback::Callback "

>  Callback()
------------------------------------------------------------------------

Default constructor.

>  Callback(Callback obj)
------------------------------------------------------------------------

Copy constructor (throws an error)

";

%feature("docstring") casadi::Callback::get_name_in "

Get the sparsity of an input This function is called during construction.

";

%feature("docstring") casadi::Callback::getWorkSize "

Get the length of the work vector.

";

%feature("docstring") casadi::Callback::sz_arg "[INTERNAL]  Get required
length of arg field.

";

%feature("docstring") casadi::Callback::slice "

returns a new function with a selection of inputs/outputs of the original

";

%feature("docstring") casadi::Callback::print "

Print a description of the object.

";

%feature("docstring") casadi::Callback::rootfinder_jac "

Access Jacobian of the ths function for a rootfinder.

";

%feature("docstring") casadi::Callback::nnz_out "

Get of number of output nonzeros For a particular output or for all for all
of the outputs.

";

%feature("docstring") casadi::Callback::mapsum "

Evaluate symbolically in parallel and sum (matrix graph)

Parameters:
-----------

parallelization:  Type of parallelization used: unroll|serial|openmp

";

%feature("docstring") casadi::Callback::free_sx "

Get all the free variables of the function.

";

%feature("docstring") casadi::Callback::name_out "

>  [str] Function.name_out() const 
------------------------------------------------------------------------

Get output scheme.

>  str Function.name_out(int ind) const 
------------------------------------------------------------------------

Get output scheme name by index.

";

%feature("docstring") casadi::Callback::nnz_in "

Get of number of input nonzeros For a particular input or for all for all of
the inputs.

";

%feature("docstring") casadi::Callback::forward "

>  void Function.forward([MX ] arg, [MX ] res, [[MX ] ] fseed,[[MX ] ] output_fsens, bool always_inline=false, bool never_inline=false)

>  void Function.forward([SX ] arg, [SX ] res, [[SX ] ] fseed,[[SX ] ] output_fsens, bool always_inline=false, bool never_inline=false)

>  void Function.forward([DM ] arg, [DM ] res, [[DM ] ] fseed,[[DM ] ] output_fsens, bool always_inline=false, bool never_inline=false)
------------------------------------------------------------------------

Create call to (cached) derivative function, forward mode.

>  Function Function.forward(int nfwd)
------------------------------------------------------------------------

Get a function that calculates nfwd forward derivatives.

Returns a function with n_in + n_out +nfwd*n_in inputs and nfwd*n_out
outputs. The first n_in inputs correspond to nondifferentiated inputs. The
next n_out inputs correspond to nondifferentiated outputs. and the last
nfwd*n_in inputs correspond to forward seeds, one direction at a time The
nfwd*n_out outputs correspond to forward sensitivities, one direction at a
time. * (n_in = n_in(), n_out = n_out())

The functions returned are cached, meaning that if called multiple timed
with the same value, then multiple references to the same function will be
returned.

";

%feature("docstring") casadi::Callback::default_in "

Get default input value (NOTE: constant reference)

";

%feature("docstring") casadi::Callback::init "

Initialize the object This function is called after the object construction
(for the whole class hierarchy) is complete, but before the finalization
step. It is called recursively for the whole class hierarchy, starting with
the lowest level.

";

%feature("docstring") casadi::Callback::numel_out "

Get of number of output elements For a particular output or for all for all
of the outputs.

";

%feature("docstring") casadi::Callback::get_n_out "

Get the number of outputs This function is called during construction.

";

%feature("docstring") casadi::Callback::generate "

Export / Generate C code for the function.

";

%feature("docstring") casadi::Callback::getAtomicInput "

Get the (integer) input arguments of an atomic operation.

";

%feature("docstring") casadi::Callback::printOption "

Print all information there is to know about a certain option.

";

%feature("docstring") casadi::Callback::getAtomicInputReal "

Get the floating point output argument of an atomic operation.

";

%feature("docstring") casadi::Callback::checkInputs "[INTERNAL]  Check if
the numerical values of the supplied bounds make sense.

";

%feature("docstring") casadi::Callback::is_null "

Is a null pointer?

";

%feature("docstring") casadi::Callback::get_n_reverse "

Return function that calculates adjoint derivatives reverse(nadj) returns a
cached instance if available, and calls  Function get_reverse(int nadj) if
no cached version is available.

";

%feature("docstring") casadi::Callback::type_name "

Get type name.

";

%feature("docstring") casadi::Callback::jacobian "

Generate a Jacobian function of output oind with respect to input iind.

Parameters:
-----------

iind:  The index of the input

oind:  The index of the output

The default behavior of this class is defined by the derived class. If
compact is set to true, only the nonzeros of the input and output
expressions are considered. If symmetric is set to true, the Jacobian being
calculated is known to be symmetric (usually a Hessian), which can be
exploited by the algorithm.

The generated Jacobian has one more output than the calling function
corresponding to the Jacobian and the same number of inputs.

";

%feature("docstring") casadi::Callback::sz_iw "[INTERNAL]  Get required
length of iw field.

";

%feature("docstring") casadi::Callback::set_reverse "

Set a function that calculates nadj adjoint derivatives NOTE: Does not take
ownership, only weak references to the derivatives are kept internally.

";

%feature("docstring") casadi::Callback::n_nodes "

Number of nodes in the algorithm.

";

%feature("docstring") casadi::Callback::sx_out "

Get symbolic primitives equivalent to the output expressions There is no
guarantee that subsequent calls return unique answers.

";

%feature("docstring") casadi::Callback::checkout "

Checkout a memory object.

";

%feature("docstring") casadi::Callback::reverse "

>  void Function.reverse([MX ] arg, [MX ] res, [[MX ] ] aseed,[[MX ] ] output_asens, bool always_inline=false, bool never_inline=false)

>  void Function.reverse([SX ] arg, [SX ] res, [[SX ] ] aseed,[[SX ] ] output_asens, bool always_inline=false, bool never_inline=false)

>  void Function.reverse([DM ] arg, [DM ] res, [[DM ] ] aseed,[[DM ] ] output_asens, bool always_inline=false, bool never_inline=false)
------------------------------------------------------------------------

Create call to (cached) derivative function, reverse mode.

>  Function Function.reverse(int nadj)
------------------------------------------------------------------------

Get a function that calculates nadj adjoint derivatives.

Returns a function with n_in + n_out +nadj*n_out inputs and nadj*n_in
outputs. The first n_in inputs correspond to nondifferentiated inputs. The
next n_out inputs correspond to nondifferentiated outputs. and the last
nadj*n_out inputs correspond to adjoint seeds, one direction at a time The
nadj*n_in outputs correspond to adjoint sensitivities, one direction at a
time. * (n_in = n_in(), n_out = n_out())

(n_in = n_in(), n_out = n_out())

The functions returned are cached, meaning that if called multiple timed
with the same value, then multiple references to the same function will be
returned.

";

%feature("docstring") casadi::Callback::~Callback "

Destructor.

";

%feature("docstring") casadi::Callback::getAtomicOutput "

Get the (integer) output argument of an atomic operation.

";

%feature("docstring") casadi::Callback::hessian "

Generate a Hessian function of output oind with respect to input iind.

Parameters:
-----------

iind:  The index of the input

oind:  The index of the output

The generated Hessian has two more outputs than the calling function
corresponding to the Hessian and the gradients.

";

%feature("docstring") casadi::Callback::linsol_cholesky "

Obtain a numeric Cholesky factorization Only for Cholesky solvers.

";

%feature("docstring") casadi::Callback::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::Callback::linsol_cholesky_sparsity "

Obtain a symbolic Cholesky factorization Only for Cholesky solvers.

";

%feature("docstring") casadi::Callback::setFullJacobian "

Set the Jacobian of all the input nonzeros with respect to all output
nonzeros NOTE: Does not take ownership, only weak references to the Jacobian
are kept internally

";

%feature("docstring") casadi::Callback::repr "

Print a representation of the object.

";

%feature("docstring") casadi::Callback::index_out "

Find the index for a string describing a particular entry of an output
scheme.

example: schemeEntry(\"x_opt\") -> returns NLPSOL_X if FunctionInternal
adheres to SCHEME_NLPINput

";

%feature("docstring") casadi::Callback::construct "

Construct internal object This is the step that actually construct the
internal object, as the class constructor only creates a null pointer. It
should be called from the user constructor.

";

%feature("docstring") casadi::Callback::get_n_forward "

Return function that calculates forward derivatives forward(nfwd) returns a
cached instance if available, and calls  Function get_forward(int nfwd) if
no cached version is available.

";

%feature("docstring") casadi::Callback::sx_in "

Get symbolic primitives equivalent to the input expressions There is no
guarantee that subsequent calls return unique answers.

";

%feature("docstring") casadi::Callback::get_forward "

Return function that calculates forward derivatives forward(nfwd) returns a
cached instance if available, and calls  Function get_forward(int nfwd) if
no cached version is available.

";

%feature("docstring") casadi::Callback::sparsity_jac "

Get, if necessary generate, the sparsity of a Jacobian block

";

%feature("docstring") casadi::Callback::rootfinder_fun "

Access rhs function for a rootfinder.

";

%feature("docstring") casadi::Callback::size1_out "

Get output dimension.

";

%feature("docstring") casadi::Callback::spCanEvaluate "[INTERNAL]  Is the
class able to propagate seeds through the algorithm?

(for usage, see the example propagating_sparsity.cpp)

";

%feature("docstring") casadi::Callback::has_jacobian "

Return Jacobian of all input elements with respect to all output elements.

";

%feature("docstring") casadi::Callback::finalize "

Finalize the object This function is called after the construction and init
steps are completed, but before user functions are called. It is called
recursively for the whole class hierarchy, starting with the highest level.

";

%feature("docstring") casadi::Callback::size2_out "

Get output dimension.

";

%feature("docstring") casadi::Callback::mx_in "

Get symbolic primitives equivalent to the input expressions There is no
guarantee that subsequent calls return unique answers.

";

%feature("docstring") casadi::Callback::release "

Release a memory object.

";

%feature("docstring") casadi::Callback::is_a "

Check if the function is of a particular type Optionally check if name
matches one of the base classes (default true)

";

%feature("docstring") casadi::Callback::integrator_dae "

Get the DAE for an integrator.

";

%feature("docstring") casadi::Callback::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::Callback::get_sparsity_out "

Get the sparsity of an output This function is called during construction.

";

%feature("docstring") casadi::Callback::name_in "

>  [str] Function.name_in() const 
------------------------------------------------------------------------

Get input scheme.

>  str Function.name_in(int ind) const 
------------------------------------------------------------------------

Get input scheme name by index.

";

%feature("docstring") casadi::Callback::has_free "

Does the function have free variables.

";

%feature("docstring") casadi::Callback::get_sparsity_in "

Get the sparsity of an input This function is called during construction.

";

%feature("docstring") casadi::Callback::tangent "

Generate a tangent function of output oind with respect to input iind.

Parameters:
-----------

iind:  The index of the input

oind:  The index of the output

The default behavior of this class is defined by the derived class. Note
that the input must be scalar. In other cases, use the Jacobian instead.

";

%feature("docstring") casadi::Callback::derivative "

>  void Function.derivative([DM] arg, [DM] &output_res, [DMVector] fseed, [DMVector] &output_fsens, [DMVector] aseed, [DMVector] &output_asens, bool always_inline=false, bool never_inline=false)

>  void Function.derivative([SX] arg, [SX] &output_res, [SXVector] fseed, [SXVector] &output_fsens, [SXVector] aseed, [SXVector] &output_asens, bool always_inline=false, bool never_inline=false)

>  void Function.derivative([MX] arg, [MX] &output_res, [MXVector] fseed, [MXVector] &output_fsens, [MXVector] aseed, [MXVector] &output_asens, bool always_inline=false, bool never_inline=false)
------------------------------------------------------------------------
[INTERNAL] 
Evaluate the function symbolically or numerically with directional
derivatives The first two arguments are the nondifferentiated inputs
and results of the evaluation, the next two arguments are a set of
forward directional seeds and the resulting forward directional
derivatives, the length of the vector being the number of forward
directions. The next two arguments are a set of adjoint directional
seeds and the resulting adjoint directional derivatives, the length of
the vector being the number of adjoint directions.

>  Function Function.derivative(int nfwd, int nadj)
------------------------------------------------------------------------

Get a function that calculates nfwd forward derivatives and nadj adjoint
derivatives Legacy function: Use forward and reverse instead.

Returns a function with (1+nfwd)*n_in+nadj*n_out inputs and (1+nfwd)*n_out +
nadj*n_in outputs. The first n_in inputs correspond to nondifferentiated
inputs. The next nfwd*n_in inputs correspond to forward seeds, one direction
at a time and the last nadj*n_out inputs correspond to adjoint seeds, one
direction at a time. The first n_out outputs correspond to nondifferentiated
outputs. The next nfwd*n_out outputs correspond to forward sensitivities,
one direction at a time and the last nadj*n_in outputs corresponds to
adjoint sensitivities, one direction at a time.

(n_in = n_in(), n_out = n_out())

";

%feature("docstring") casadi::Callback::n_in "

Get the number of function inputs.

";

%feature("docstring") casadi::Callback::sz_w "[INTERNAL]  Get required
length of w field.

";

%feature("docstring") casadi::Callback::gradient "

Generate a gradient function of output oind with respect to input iind.

Parameters:
-----------

iind:  The index of the input

oind:  The index of the output

The default behavior of this class is defined by the derived class. Note
that the output must be scalar. In other cases, use the Jacobian instead.

";

%feature("docstring") casadi::Callback::printPtr "[INTERNAL]  Print the
pointer to the internal class

";

%feature("docstring") casadi::Callback::kernel_sum "

kernel_sum Consider a dense matrix V.

KernelSum computes

F(V,X) = sum_i sum_j f ( [i;j], V(i,j), X)

with X: [x;y]

where the summation is taken for all entries (i,j) that are a distance r
away from X.

This function assumes that V is fixed: sensitivities with respect to it are
not computed.

This allows for improved speed of evaluation.

Having V fixed is a common use case: V may be a large bitmap (observation),
onto which a kernel is fitted.

Joris Gillis

";

%feature("docstring") casadi::Callback::set_jac_sparsity "

Generate the sparsity of a Jacobian block

";

%feature("docstring") casadi::Callback::qpsol_debug "

Generate native code in the interfaced language for debugging

";

%feature("docstring") casadi::Callback::getAlgorithmSize "

Get the number of atomic operations.

";

%feature("docstring") casadi::Callback::sz_res "[INTERNAL]  Get required
length of res field.

";

%feature("docstring") casadi::Callback::printOptions "

Print options to a stream.

";

%feature("docstring") casadi::Callback::get_jacobian "

Return Jacobian of all input elements with respect to all output elements.

";

%feature("docstring") casadi::Callback::printDimensions "

Print dimensions of inputs and outputs.

";

%feature("docstring") casadi::Callback::setJacobian "

Set the Jacobian function of output oind with respect to input iind NOTE:
Does not take ownership, only weak references to the Jacobians are kept
internally

";

%feature("docstring") casadi::Callback::linsol_solve "

Create a solve node.

";

%feature("docstring") casadi::Callback::eval "

Evaluate numerically, temporary matrices and work vectors.

";

%feature("docstring") casadi::Callback::numel_in "

Get of number of input elements For a particular input or for all for all of
the inputs.

";

%feature("docstring") casadi::Callback::getAtomicOperation "

Get an atomic operation operator index.

";

%feature("docstring") casadi::Callback::map "

>  Function Function.map(str name, str parallelization, int n, [int ] reduce_in, [int ] reduce_out, Dict opts=Dict())

>  Function Function.map(str name, str parallelization, int n, [str ] reduce_in, [str ] reduce_out, Dict opts=Dict())

>  Function Function.map(str name, str parallelization, int n, Dict opts=Dict())
------------------------------------------------------------------------

Create a mapped version of this function.

Suppose the function has a signature of:

::

     f: (a, p) -> ( s )
  



The the mapaccumulated version has the signature:

::

     F: (A, P) -> (S )
  
      with
          A: horzcat([a0, a1, ..., a_(N-1)])
          P: horzcat([p0, p1, ..., p_(N-1)])
          S: horzcat([s0, s1, ..., s_(N-1)])
      and
          s0 <- f(a0, p0)
          s1 <- f(a1, p1)
          ...
          s_(N-1) <- f(a_(N-1), p_(N-1))
  



Parameters:
-----------

parallelization:  Type of parallelization used: unroll|serial|openmp

>  [MX] Function.map([MX ] arg, str parallelization=\"serial\")

>  std.map<str, MX> Function.map(const std.map< str, MX > &arg, str parallelization=\"serial\")
------------------------------------------------------------------------

Evaluate symbolically in parallel (matrix graph)

Parameters:
-----------

parallelization:  Type of parallelization used: unroll|serial|openmp

";

%feature("docstring") casadi::Callback::mapaccum "

Create a mapaccumulated version of this function.

Suppose the function has a signature of:

::

     f: (x, u) -> (x_next , y )
  



The the mapaccumulated version has the signature:

::

     F: (x0, U) -> (X , Y )
  
      with
          U: horzcat([u0, u1, ..., u_(N-1)])
          X: horzcat([x1, x2, ..., x_N])
          Y: horzcat([y0, y1, ..., y_(N-1)])
  
      and
          x1, y0 <- f(x0, u0)
          x2, y1 <- f(x1, u1)
          ...
          x_N, y_(N-1) <- f(x_(N-1), u_(N-1))
  



";

%feature("docstring") casadi::Callback::size1_in "

Get input dimension.

";

%feature("docstring") casadi::Callback::sparsity_in "

Get sparsity of a given input.

";

%feature("docstring") casadi::Callback::rootfinder_linsol "

Access linear solver of a rootfinder.

";

%feature("docstring") casadi::Callback::get_reverse "

Return function that calculates adjoint derivatives reverse(nadj) returns a
cached instance if available, and calls  Function get_reverse(int nadj) if
no cached version is available.

";

%feature("docstring") casadi::Callback::expand "

Expand a function to SX.

";

%feature("docstring") casadi::Callback "

Callback function functionality This class provides a public API to the
FunctionInternal class that can be subclassed by the user, who is then able
to implement the different virtual method. Note that the Function class also
provides a public API to FunctionInternal, but only allows calling, not
being called.

The user is responsible for not deleting this class for the lifetime of the
internal function object.

Joris Gillis, Joel Andersson

C++ includes: callback.hpp ";

%feature("docstring") casadi::Callback::addMonitor "

Add modules to be monitored.

";

%feature("docstring") casadi::Callback::n_out "

Get the number of function outputs.

";

%feature("docstring") casadi::Callback::removeMonitor "

Remove modules to be monitored.

";

%feature("docstring") casadi::Callback::__hash__ "

Returns a number that is unique for a given Node. If the Object does not
point to any node, \"0\" is returned.

";

%feature("docstring") casadi::Callback::free_mx "

Get all the free variables of the function.

";

%feature("docstring") casadi::Callback::index_in "

Find the index for a string describing a particular entry of an input
scheme.

example: schemeEntry(\"x_opt\") -> returns NLPSOL_X if FunctionInternal
adheres to SCHEME_NLPINput

";

%feature("docstring") casadi::Callback::generate_lifted "

Extract the functions needed for the Lifted Newton method.

";

%feature("docstring") casadi::Callback::sparsity_out "

Get sparsity of a given output.

";

%feature("docstring") casadi::Callback::stats "

Get all statistics obtained at the end of the last evaluate call.

";

%feature("docstring") casadi::Callback::name "

Name of the function.

";

%feature("docstring") casadi::Callback::size_out "

Get output dimension.

";

%feature("docstring") casadi::Callback::fullJacobian "

Generate a Jacobian function of all the inputs elements with respect to all
the output elements).

";

%feature("docstring") casadi::Callback::set_forward "

Set a function that calculates nfwd forward derivatives NOTE: Does not take
ownership, only weak references to the derivatives are kept internally.

";

%feature("docstring") casadi::Callback::get_name_out "

Get the sparsity of an output This function is called during construction.

";

%feature("docstring") casadi::Callback::mx_out "

Get symbolic primitives equivalent to the output expressions There is no
guarantee that subsequent calls return unique answers.

";


// File: classcasadi_1_1casadi__limits.xml
%feature("docstring") casadi::casadi_limits "

casadi_limits class

The following class, which acts as a complements to the standard
numeric_limits class, allows specifying certain properties of scalar
objects. The template can be specialized for e.g. symbolic scalars Joel
Andersson

C++ includes: casadi_limits.hpp ";


// File: classcasadi_1_1CasadiException.xml
%feature("docstring") casadi::CasadiException::what "throw () Display
error.

";

%feature("docstring") casadi::CasadiException::CasadiException "

>  CasadiException()
------------------------------------------------------------------------

Default constructor.

>  CasadiException(str msg)
------------------------------------------------------------------------

Form message string.

";

%feature("docstring") casadi::CasadiException "

Casadi exception class.

Joel Andersson

C++ includes: exception.hpp ";

%feature("docstring") casadi::CasadiException::~CasadiException "throw ()
Destructor.

";


// File: classcasadi_1_1CasadiMeta.xml
%feature("docstring") casadi::CasadiMeta "

Collects global CasADi meta information.

Joris Gillis

C++ includes: casadi_meta.hpp ";


// File: classcasadi_1_1ClangCompiler.xml


// File: classcasadi_1_1CodeGenerator.xml
%feature("docstring") casadi::CodeGenerator::addInclude "

Add an include file optionally using a relative path \"...\" instead of an
absolute path <...>

";

%feature("docstring") casadi::CodeGenerator "C++ includes:
code_generator.hpp ";

%feature("docstring") casadi::CodeGenerator::compile "

Compile and load function.

";

%feature("docstring") casadi::CodeGenerator::add "

Add a function (name generated)

";

%feature("docstring") casadi::CodeGenerator::CodeGenerator "

Constructor.

";

%feature("docstring") casadi::CodeGenerator::generate "

>  void CodeGenerator.generate(str name) const 
------------------------------------------------------------------------

Generate a file.

>  str CodeGenerator.generate() const 
------------------------------------------------------------------------

Generate a file, return code as string.

";


// File: classcasadi_1_1CollocationIntegrator.xml


// File: classcasadi_1_1Compiler.xml
%feature("docstring") casadi::Compiler::plugin_name "

Query plugin name.

";

%feature("docstring") casadi::Compiler::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::Compiler::print "

Print a description of the object.

";

%feature("docstring") casadi::Compiler::printPtr "[INTERNAL]  Print the
pointer to the internal class

";

%feature("docstring") casadi::Compiler::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::Compiler::repr "

Print a representation of the object.

";

%feature("docstring") casadi::Compiler "

Compiler.

Just-in-time compilation of code

General information
===================



List of plugins
===============



- clang

- shell

Note: some of the plugins in this list might not be available on your
system. Also, there might be extra plugins available to you that are not
listed here. You can obtain their documentation with
Compiler.doc(\"myextraplugin\")



--------------------------------------------------------------------------------

clang
-----



Interface to the JIT compiler CLANG

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| flags                  | OT_STRINGVECTOR        | Compile flags for the  |
|                        |                        | JIT compiler. Default: |
|                        |                        | None                   |
+------------------------+------------------------+------------------------+
| include_path           | OT_STRING              | Include paths for the  |
|                        |                        | JIT compiler. The      |
|                        |                        | include directory      |
|                        |                        | shipped with CasADi    |
|                        |                        | will be automatically  |
|                        |                        | appended.              |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

shell
-----



Interface to the JIT compiler SHELL

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| compiler               | OT_STRING              | Compiler command       |
+------------------------+------------------------+------------------------+
| compiler_setup         | OT_STRING              | Compiler setup         |
|                        |                        | command. Intended to   |
|                        |                        | be fixed. The 'flag'   |
|                        |                        | option is the prefered |
|                        |                        | way to set custom      |
|                        |                        | flags.                 |
+------------------------+------------------------+------------------------+
| flags                  | OT_STRINGVECTOR        | Compile flags for the  |
|                        |                        | JIT compiler. Default: |
|                        |                        | None                   |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



Joris Gillis

C++ includes: compiler.hpp ";

%feature("docstring") casadi::Compiler::is_null "

Is a null pointer?

";

%feature("docstring") casadi::Compiler::__hash__ "

Returns a number that is unique for a given Node. If the Object does not
point to any node, \"0\" is returned.

";

%feature("docstring") casadi::Compiler::Compiler "

>  Compiler()
------------------------------------------------------------------------

Default constructor.

>  Compiler(str name, str compiler, Dict opts=Dict())
------------------------------------------------------------------------

Compiler factory (new syntax, includes initialization)

";


// File: classcasadi_1_1Concat.xml


// File: classcasadi_1_1Constant.xml


// File: classcasadi_1_1ConstantDM.xml


// File: classcasadi_1_1ConstantMX.xml


// File: classcasadi_1_1ConstantSX.xml


// File: classcasadi_1_1DaeBuilder.xml


/*  Variables and equations  */

/* Public data members

*/

/*  Symbolic modeling  */

/* Formulate an optimal control problem

*/

/*  Manipulation  */

/* Reformulate the dynamic optimization problem.

*/

/*  Import and export  */ %feature("docstring") casadi::DaeBuilder::add_s "

Add a implicit state.

";

%feature("docstring") casadi::DaeBuilder::sanity_check "

Check if dimensions match.

";

%feature("docstring") casadi::DaeBuilder::set_unit "

Set the unit for a component.

";

%feature("docstring") casadi::DaeBuilder::DaeBuilder "

Default constructor.

";

%feature("docstring") casadi::DaeBuilder::sort_dae "

Sort the DAE and implicitly defined states.

";

%feature("docstring") casadi::DaeBuilder::start "

>  double DaeBuilder.start(str name, bool normalized=false) const 
------------------------------------------------------------------------

Get the (optionally normalized) value at time 0 by name.

>  [double] DaeBuilder.start(MX var, bool normalized=false) const 
------------------------------------------------------------------------

Get the (optionally normalized) value(s) at time 0 by expression.

";

%feature("docstring") casadi::DaeBuilder::add_variable "

>  void DaeBuilder.add_variable(str name, Variable var)
------------------------------------------------------------------------

Add a variable.

>  MX DaeBuilder.add_variable(str name, int n=1)

>  MX DaeBuilder.add_variable(str name, Sparsity sp)
------------------------------------------------------------------------

Add a new variable: returns corresponding symbolic expression.

";

%feature("docstring") casadi::DaeBuilder::min "

>  double DaeBuilder.min(str name, bool normalized=false) const 
------------------------------------------------------------------------

Get the lower bound by name.

>  [double] DaeBuilder.min(MX var, bool normalized=false) const 
------------------------------------------------------------------------

Get the lower bound(s) by expression.

";

%feature("docstring") casadi::DaeBuilder::add_quad "

Add a quadrature equation.

";

%feature("docstring") casadi::DaeBuilder::nominal "

>  double DaeBuilder.nominal(str name) const 
------------------------------------------------------------------------

Get the nominal value by name.

>  [double] DaeBuilder.nominal(MX var) const 
------------------------------------------------------------------------

Get the nominal value(s) by expression.

";

%feature("docstring") casadi::DaeBuilder::max "

>  double DaeBuilder.max(str name, bool normalized=false) const 
------------------------------------------------------------------------

Get the upper bound by name.

>  [double] DaeBuilder.max(MX var, bool normalized=false) const 
------------------------------------------------------------------------

Get the upper bound(s) by expression.

";

%feature("docstring") casadi::DaeBuilder::make_semi_explicit "

Transform the implicit DAE to a semi-explicit DAE.

";

%feature("docstring") casadi::DaeBuilder::set_derivative_start "

>  void DaeBuilder.set_derivative_start(str name, double val, bool normalized=false)
------------------------------------------------------------------------

Set the (optionally normalized) derivative value at time 0 by name.

>  void DaeBuilder.set_derivative_start(MX var, [double ] val, bool normalized=false)
------------------------------------------------------------------------

Set the (optionally normalized) derivative value(s) at time 0 by expression.

";

%feature("docstring") casadi::DaeBuilder::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::DaeBuilder::add_lc "

Add a named linear combination of output expressions.

";

%feature("docstring") casadi::DaeBuilder::sort_d "

Sort dependent parameters.

";

%feature("docstring") casadi::DaeBuilder::print "

Print description.

";

%feature("docstring") casadi::DaeBuilder::sort_alg "

Sort the algebraic equations and algebraic states.

";

%feature("docstring") casadi::DaeBuilder::add_dae "

Add a differential-algebraic equation.

";

%feature("docstring") casadi::DaeBuilder::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::DaeBuilder::make_explicit "

Transform the implicit DAE or semi-explicit DAE into an explicit ODE.

";

%feature("docstring") casadi::DaeBuilder "

An initial-value problem in differential-algebraic equations.

Independent variables:
======================





::

  t:      time
  



Time-continuous variables:
==========================





::

  x:      states defined by ODE
  s:      implicitly defined states
  z:      algebraic variables
  u:      control signals
  q:      quadrature states
  y:      outputs
  



Time-constant variables:
========================





::

  p:      free parameters
  d:      dependent parameters
  



Dynamic constraints (imposed everywhere):
=========================================





::

  ODE                    \\\\dot{x} ==  ode(t, x, s, z, u, p, d)
  DAE or implicit ODE:         0 ==  dae(t, x, s, z, u, p, d, sdot)
  algebraic equations:         0 ==  alg(t, x, s, z, u, p, d)
  quadrature equations:  \\\\dot{q} == quad(t, x, s, z, u, p, d)
  dependent parameters:        d == ddef(t, x, s, z, u, p, d)
  output equations:            y == ydef(t, x, s, z, u, p, d)
  



Point constraints (imposed pointwise):
======================================





::

  Initial equations:           0 == init(t, x, s, z, u, p, d, sdot)
  



Joel Andersson

C++ includes: dae_builder.hpp ";

%feature("docstring") casadi::DaeBuilder::add_q "

Add a new quadrature state.

";

%feature("docstring") casadi::DaeBuilder::add_p "

Add a new parameter

";

%feature("docstring") casadi::DaeBuilder::add_u "

Add a new control.

";

%feature("docstring") casadi::DaeBuilder::set_guess "

>  void DaeBuilder.set_guess(str name, double val, bool normalized=false)
------------------------------------------------------------------------

Set the initial guess by name.

>  void DaeBuilder.set_guess(MX var, [double ] val, bool normalized=false)
------------------------------------------------------------------------

Set the initial guess(es) by expression.

";

%feature("docstring") casadi::DaeBuilder::add_z "

Add a new algebraic variable.

";

%feature("docstring") casadi::DaeBuilder::add_y "

Add a new output.

";

%feature("docstring") casadi::DaeBuilder::add_x "

Add a new differential state.

";

%feature("docstring") casadi::DaeBuilder::eliminate_quad "

Eliminate quadrature states and turn them into ODE states.

";

%feature("docstring") casadi::DaeBuilder::derivative_start "

>  double DaeBuilder.derivative_start(str name, bool normalized=false) const 
------------------------------------------------------------------------

Get the (optionally normalized) derivative value at time 0 by name.

>  [double] DaeBuilder.derivative_start(MX var, bool normalized=false) const 
------------------------------------------------------------------------

Get the (optionally normalized) derivative value(s) at time 0 by expression.

";

%feature("docstring") casadi::DaeBuilder::set_nominal "

>  void DaeBuilder.set_nominal(str name, double val)
------------------------------------------------------------------------

Set the nominal value by name.

>  void DaeBuilder.set_nominal(MX var, [double ] val)
------------------------------------------------------------------------

Set the nominal value(s) by expression.

";

%feature("docstring") casadi::DaeBuilder::parse_fmi "

Import existing problem from FMI/XML

";

%feature("docstring") casadi::DaeBuilder::add_d "

Add a new dependent parameter.

";

%feature("docstring") casadi::DaeBuilder::split_dae "

Identify and separate the algebraic variables and equations in the DAE.

";

%feature("docstring") casadi::DaeBuilder::add_alg "

Add an algebraic equation.

";

%feature("docstring") casadi::DaeBuilder::repr "

Print representation.

";

%feature("docstring") casadi::DaeBuilder::create "

Construct a function object.

";

%feature("docstring") casadi::DaeBuilder::guess "

>  double DaeBuilder.guess(str name, bool normalized=false) const 
------------------------------------------------------------------------

Get the initial guess by name.

>  [double] DaeBuilder.guess(MX var, bool normalized=false) const 
------------------------------------------------------------------------

Get the initial guess(es) by expression.

";

%feature("docstring") casadi::DaeBuilder::variable "

Access a variable by name

";

%feature("docstring") casadi::DaeBuilder::set_max "

>  void DaeBuilder.set_max(str name, double val, bool normalized=false)
------------------------------------------------------------------------

Set the upper bound by name.

>  void DaeBuilder.set_max(MX var, [double ] val, bool normalized=false)
------------------------------------------------------------------------

Set the upper bound(s) by expression.

";

%feature("docstring") casadi::DaeBuilder::eliminate_alg "

Eliminate algebraic variables and equations transforming them into outputs.

";

%feature("docstring") casadi::DaeBuilder::set_min "

>  void DaeBuilder.set_min(str name, double val, bool normalized=false)
------------------------------------------------------------------------

Set the lower bound by name.

>  void DaeBuilder.set_min(MX var, [double ] val, bool normalized=false)
------------------------------------------------------------------------

Set the lower bound(s) by expression.

";

%feature("docstring") casadi::DaeBuilder::set_start "

>  void DaeBuilder.set_start(str name, double val, bool normalized=false)
------------------------------------------------------------------------

Set the (optionally normalized) value at time 0 by name.

>  void DaeBuilder.set_start(MX var, [double ] val, bool normalized=false)
------------------------------------------------------------------------

Set the (optionally normalized) value(s) at time 0 by expression.

";

%feature("docstring") casadi::DaeBuilder::unit "

>  str DaeBuilder.unit(str name) const 
------------------------------------------------------------------------

Get the unit for a component.

>  str DaeBuilder.unit(MX var) const 
------------------------------------------------------------------------

Get the unit given a vector of symbolic variables (all units must be
identical)

";

%feature("docstring") casadi::DaeBuilder::scale_variables "

Scale the variables.

";

%feature("docstring") casadi::DaeBuilder::split_d "

Eliminate interdependencies amongst dependent parameters.

";

%feature("docstring") casadi::DaeBuilder::eliminate_d "

Eliminate dependent parameters.

";

%feature("docstring") casadi::DaeBuilder::der "

>  MX DaeBuilder.der(str name) const 
------------------------------------------------------------------------

Get a derivative expression by name.

>  MX DaeBuilder.der(MX var) const 
------------------------------------------------------------------------

Get a derivative expression by non-differentiated expression.

";

%feature("docstring") casadi::DaeBuilder::scale_equations "

Scale the implicit equations.

";

%feature("docstring") casadi::DaeBuilder::add_ode "

Add an ordinary differential equation.

";


// File: classcasadi_1_1DenseMultiplication.xml


// File: classcasadi_1_1DenseTranspose.xml


// File: classcasadi_1_1Determinant.xml


// File: classcasadi_1_1Diagcat.xml


// File: classcasadi_1_1Diagsplit.xml


// File: classcasadi_1_1DllLibrary.xml


// File: classcasadi_1_1Dot.xml


// File: classcasadi_1_1External.xml


// File: classcasadi_1_1Find.xml


// File: classcasadi_1_1FixedStepIntegrator.xml


// File: classcasadi_1_1FStats.xml
%feature("docstring") casadi::FStats::tic "[INTERNAL]  Start timing.

";

%feature("docstring") casadi::FStats::reset "[INTERNAL]  Reset the
statistics.

";

%feature("docstring") casadi::FStats::FStats "[INTERNAL]  Constructor.

";

%feature("docstring") casadi::FStats "[INTERNAL]  Timer class

FStats hack; hack.tic(); .... hack.toc();

C++ includes: timing.hpp ";

%feature("docstring") casadi::FStats::toc "[INTERNAL]  Stop timing.

";


// File: classcasadi_1_1Function.xml
%feature("docstring") casadi::Function::repr "

Print a representation of the object.

";

%feature("docstring") casadi::Function::gradient "

Generate a gradient function of output oind with respect to input iind.

Parameters:
-----------

iind:  The index of the input

oind:  The index of the output

The default behavior of this class is defined by the derived class. Note
that the output must be scalar. In other cases, use the Jacobian instead.

";

%feature("docstring") casadi::Function::nnz_in "

Get of number of input nonzeros For a particular input or for all for all of
the inputs.

";

%feature("docstring") casadi::Function::sz_res "[INTERNAL]  Get required
length of res field.

";

%feature("docstring") casadi::Function::name "

Name of the function.

";

%feature("docstring") casadi::Function::checkInputs "[INTERNAL]  Check if
the numerical values of the supplied bounds make sense.

";

%feature("docstring") casadi::Function::printPtr "[INTERNAL]  Print the
pointer to the internal class

";

%feature("docstring") casadi::Function::jacobian "

Generate a Jacobian function of output oind with respect to input iind.

Parameters:
-----------

iind:  The index of the input

oind:  The index of the output

The default behavior of this class is defined by the derived class. If
compact is set to true, only the nonzeros of the input and output
expressions are considered. If symmetric is set to true, the Jacobian being
calculated is known to be symmetric (usually a Hessian), which can be
exploited by the algorithm.

The generated Jacobian has one more output than the calling function
corresponding to the Jacobian and the same number of inputs.

";

%feature("docstring") casadi::Function::stats "

Get all statistics obtained at the end of the last evaluate call.

";

%feature("docstring") casadi::Function::setFullJacobian "

Set the Jacobian of all the input nonzeros with respect to all output
nonzeros NOTE: Does not take ownership, only weak references to the Jacobian
are kept internally

";

%feature("docstring") casadi::Function::sparsity_out "

Get sparsity of a given output.

";

%feature("docstring") casadi::Function::integrator_dae "

Get the DAE for an integrator.

";

%feature("docstring") casadi::Function::getAtomicOperation "

Get an atomic operation operator index.

";

%feature("docstring") casadi::Function::printOptions "

Print options to a stream.

";

%feature("docstring") casadi::Function::getAlgorithmSize "

Get the number of atomic operations.

";

%feature("docstring") casadi::Function::set_reverse "

Set a function that calculates nadj adjoint derivatives NOTE: Does not take
ownership, only weak references to the derivatives are kept internally.

";

%feature("docstring") casadi::Function::qpsol_debug "

Generate native code in the interfaced language for debugging

";

%feature("docstring") casadi::Function::numel_out "

Get of number of output elements For a particular output or for all for all
of the outputs.

";

%feature("docstring") casadi::Function::printDimensions "

Print dimensions of inputs and outputs.

";

%feature("docstring") casadi::Function::kernel_sum "

kernel_sum Consider a dense matrix V.

KernelSum computes

F(V,X) = sum_i sum_j f ( [i;j], V(i,j), X)

with X: [x;y]

where the summation is taken for all entries (i,j) that are a distance r
away from X.

This function assumes that V is fixed: sensitivities with respect to it are
not computed.

This allows for improved speed of evaluation.

Having V fixed is a common use case: V may be a large bitmap (observation),
onto which a kernel is fitted.

Joris Gillis

";

%feature("docstring") casadi::Function::__hash__ "

Returns a number that is unique for a given Node. If the Object does not
point to any node, \"0\" is returned.

";

%feature("docstring") casadi::Function::name_out "

>  [str] Function.name_out() const 
------------------------------------------------------------------------

Get output scheme.

>  str Function.name_out(int ind) const 
------------------------------------------------------------------------

Get output scheme name by index.

";

%feature("docstring") casadi::Function::n_in "

Get the number of function inputs.

";

%feature("docstring") casadi::Function::rootfinder_linsol "

Access linear solver of a rootfinder.

";

%feature("docstring") casadi::Function::mx_out "

Get symbolic primitives equivalent to the output expressions There is no
guarantee that subsequent calls return unique answers.

";

%feature("docstring") casadi::Function::generate_dependencies "

Export / Generate C code for the dependency function.

";

%feature("docstring") casadi::Function::free_sx "

Get all the free variables of the function.

";

%feature("docstring") casadi::Function::mapaccum "

Create a mapaccumulated version of this function.

Suppose the function has a signature of:

::

     f: (x, u) -> (x_next , y )
  



The the mapaccumulated version has the signature:

::

     F: (x0, U) -> (X , Y )
  
      with
          U: horzcat([u0, u1, ..., u_(N-1)])
          X: horzcat([x1, x2, ..., x_N])
          Y: horzcat([y0, y1, ..., y_(N-1)])
  
      and
          x1, y0 <- f(x0, u0)
          x2, y1 <- f(x1, u1)
          ...
          x_N, y_(N-1) <- f(x_(N-1), u_(N-1))
  



";

%feature("docstring") casadi::Function::derivative "

>  void Function.derivative([DM] arg, [DM] &output_res, [DMVector] fseed, [DMVector] &output_fsens, [DMVector] aseed, [DMVector] &output_asens, bool always_inline=false, bool never_inline=false)

>  void Function.derivative([SX] arg, [SX] &output_res, [SXVector] fseed, [SXVector] &output_fsens, [SXVector] aseed, [SXVector] &output_asens, bool always_inline=false, bool never_inline=false)

>  void Function.derivative([MX] arg, [MX] &output_res, [MXVector] fseed, [MXVector] &output_fsens, [MXVector] aseed, [MXVector] &output_asens, bool always_inline=false, bool never_inline=false)
------------------------------------------------------------------------
[INTERNAL] 
Evaluate the function symbolically or numerically with directional
derivatives The first two arguments are the nondifferentiated inputs
and results of the evaluation, the next two arguments are a set of
forward directional seeds and the resulting forward directional
derivatives, the length of the vector being the number of forward
directions. The next two arguments are a set of adjoint directional
seeds and the resulting adjoint directional derivatives, the length of
the vector being the number of adjoint directions.

>  Function Function.derivative(int nfwd, int nadj)
------------------------------------------------------------------------

Get a function that calculates nfwd forward derivatives and nadj adjoint
derivatives Legacy function: Use forward and reverse instead.

Returns a function with (1+nfwd)*n_in+nadj*n_out inputs and (1+nfwd)*n_out +
nadj*n_in outputs. The first n_in inputs correspond to nondifferentiated
inputs. The next nfwd*n_in inputs correspond to forward seeds, one direction
at a time and the last nadj*n_out inputs correspond to adjoint seeds, one
direction at a time. The first n_out outputs correspond to nondifferentiated
outputs. The next nfwd*n_out outputs correspond to forward sensitivities,
one direction at a time and the last nadj*n_in outputs corresponds to
adjoint sensitivities, one direction at a time.

(n_in = n_in(), n_out = n_out())

";

%feature("docstring") casadi::Function::Function "

>  Function(str name, [SX ] arg, [SX ] res, Dict opts=Dict())

>  Function(str name, [SX ] arg, [SX ] res, [str ] argn, [str ] resn, Dict opts=Dict())

>  Function(str name, const std.map< str, SX > &dict, [str ] argn, [str ] resn, Dict opts=Dict())
------------------------------------------------------------------------

Construct an SX function.

>  Function(str name, [MX ] arg, [MX ] res, Dict opts=Dict())

>  Function(str name, [MX ] arg, [MX ] res, [str ] argn, [str ] resn, Dict opts=Dict())

>  Function(str name, const std.map< str, MX > &dict, [str ] argn, [str ] resn, Dict opts=Dict())
------------------------------------------------------------------------

Construct an MX function.

>  Function()
------------------------------------------------------------------------

Default constructor, null pointer.

>  Function(str fname)
------------------------------------------------------------------------

Construct from a file.

";

%feature("docstring") casadi::Function::type_name "

Get type name.

";

%feature("docstring") casadi::Function::numel_in "

Get of number of input elements For a particular input or for all for all of
the inputs.

";

%feature("docstring") casadi::Function::is_null "

Is a null pointer?

";

%feature("docstring") casadi::Function::printOption "

Print all information there is to know about a certain option.

";

%feature("docstring") casadi::Function::set_forward "

Set a function that calculates nfwd forward derivatives NOTE: Does not take
ownership, only weak references to the derivatives are kept internally.

";

%feature("docstring") casadi::Function::forward "

>  void Function.forward([MX ] arg, [MX ] res, [[MX ] ] fseed,[[MX ] ] output_fsens, bool always_inline=false, bool never_inline=false)

>  void Function.forward([SX ] arg, [SX ] res, [[SX ] ] fseed,[[SX ] ] output_fsens, bool always_inline=false, bool never_inline=false)

>  void Function.forward([DM ] arg, [DM ] res, [[DM ] ] fseed,[[DM ] ] output_fsens, bool always_inline=false, bool never_inline=false)
------------------------------------------------------------------------

Create call to (cached) derivative function, forward mode.

>  Function Function.forward(int nfwd)
------------------------------------------------------------------------

Get a function that calculates nfwd forward derivatives.

Returns a function with n_in + n_out +nfwd*n_in inputs and nfwd*n_out
outputs. The first n_in inputs correspond to nondifferentiated inputs. The
next n_out inputs correspond to nondifferentiated outputs. and the last
nfwd*n_in inputs correspond to forward seeds, one direction at a time The
nfwd*n_out outputs correspond to forward sensitivities, one direction at a
time. * (n_in = n_in(), n_out = n_out())

The functions returned are cached, meaning that if called multiple timed
with the same value, then multiple references to the same function will be
returned.

";

%feature("docstring") casadi::Function::n_out "

Get the number of function outputs.

";

%feature("docstring") casadi::Function::getAtomicInput "

Get the (integer) input arguments of an atomic operation.

";

%feature("docstring") casadi::Function::index_in "

Find the index for a string describing a particular entry of an input
scheme.

example: schemeEntry(\"x_opt\") -> returns NLPSOL_X if FunctionInternal
adheres to SCHEME_NLPINput

";

%feature("docstring") casadi::Function::sz_w "[INTERNAL]  Get required
length of w field.

";

%feature("docstring") casadi::Function::~Function "

To resolve ambiguity on some compilers.

Destructor

";

%feature("docstring") casadi::Function::index_out "

Find the index for a string describing a particular entry of an output
scheme.

example: schemeEntry(\"x_opt\") -> returns NLPSOL_X if FunctionInternal
adheres to SCHEME_NLPINput

";

%feature("docstring") casadi::Function::free_mx "

Get all the free variables of the function.

";

%feature("docstring") casadi::Function::size_out "

Get output dimension.

";

%feature("docstring") casadi::Function::linsol_solve "

Create a solve node.

";

%feature("docstring") casadi::Function::addMonitor "

Add modules to be monitored.

";

%feature("docstring") casadi::Function::slice "

returns a new function with a selection of inputs/outputs of the original

";

%feature("docstring") casadi::Function::mapsum "

Evaluate symbolically in parallel and sum (matrix graph)

Parameters:
-----------

parallelization:  Type of parallelization used: unroll|serial|openmp

";

%feature("docstring") casadi::Function::nnz_out "

Get of number of output nonzeros For a particular output or for all for all
of the outputs.

";

%feature("docstring") casadi::Function::print "

Print a description of the object.

";

%feature("docstring") casadi::Function::hessian "

Generate a Hessian function of output oind with respect to input iind.

Parameters:
-----------

iind:  The index of the input

oind:  The index of the output

The generated Hessian has two more outputs than the calling function
corresponding to the Hessian and the gradients.

";

%feature("docstring") casadi::Function::expand "

Expand a function to SX.

";

%feature("docstring") casadi::Function::checkout "

Checkout a memory object.

";

%feature("docstring") casadi::Function::rootfinder_jac "

Access Jacobian of the ths function for a rootfinder.

";

%feature("docstring") casadi::Function::fullJacobian "

Generate a Jacobian function of all the inputs elements with respect to all
the output elements).

";

%feature("docstring") casadi::Function::map "

>  Function Function.map(str name, str parallelization, int n, [int ] reduce_in, [int ] reduce_out, Dict opts=Dict())

>  Function Function.map(str name, str parallelization, int n, [str ] reduce_in, [str ] reduce_out, Dict opts=Dict())

>  Function Function.map(str name, str parallelization, int n, Dict opts=Dict())
------------------------------------------------------------------------

Create a mapped version of this function.

Suppose the function has a signature of:

::

     f: (a, p) -> ( s )
  



The the mapaccumulated version has the signature:

::

     F: (A, P) -> (S )
  
      with
          A: horzcat([a0, a1, ..., a_(N-1)])
          P: horzcat([p0, p1, ..., p_(N-1)])
          S: horzcat([s0, s1, ..., s_(N-1)])
      and
          s0 <- f(a0, p0)
          s1 <- f(a1, p1)
          ...
          s_(N-1) <- f(a_(N-1), p_(N-1))
  



Parameters:
-----------

parallelization:  Type of parallelization used: unroll|serial|openmp

>  [MX] Function.map([MX ] arg, str parallelization=\"serial\")

>  std.map<str, MX> Function.map(const std.map< str, MX > &arg, str parallelization=\"serial\")
------------------------------------------------------------------------

Evaluate symbolically in parallel (matrix graph)

Parameters:
-----------

parallelization:  Type of parallelization used: unroll|serial|openmp

";

%feature("docstring") casadi::Function::sx_in "

Get symbolic primitives equivalent to the input expressions There is no
guarantee that subsequent calls return unique answers.

";

%feature("docstring") casadi::Function::size1_in "

Get input dimension.

";

%feature("docstring") casadi::Function::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::Function::n_nodes "

Number of nodes in the algorithm.

";

%feature("docstring") casadi::Function::linsol_cholesky_sparsity "

Obtain a symbolic Cholesky factorization Only for Cholesky solvers.

";

%feature("docstring") casadi::Function::sz_iw "[INTERNAL]  Get required
length of iw field.

";

%feature("docstring") casadi::Function::is_a "

Check if the function is of a particular type Optionally check if name
matches one of the base classes (default true)

";

%feature("docstring") casadi::Function::name_in "

>  [str] Function.name_in() const 
------------------------------------------------------------------------

Get input scheme.

>  str Function.name_in(int ind) const 
------------------------------------------------------------------------

Get input scheme name by index.

";

%feature("docstring") casadi::Function::mx_in "

Get symbolic primitives equivalent to the input expressions There is no
guarantee that subsequent calls return unique answers.

";

%feature("docstring") casadi::Function::reverse "

>  void Function.reverse([MX ] arg, [MX ] res, [[MX ] ] aseed,[[MX ] ] output_asens, bool always_inline=false, bool never_inline=false)

>  void Function.reverse([SX ] arg, [SX ] res, [[SX ] ] aseed,[[SX ] ] output_asens, bool always_inline=false, bool never_inline=false)

>  void Function.reverse([DM ] arg, [DM ] res, [[DM ] ] aseed,[[DM ] ] output_asens, bool always_inline=false, bool never_inline=false)
------------------------------------------------------------------------

Create call to (cached) derivative function, reverse mode.

>  Function Function.reverse(int nadj)
------------------------------------------------------------------------

Get a function that calculates nadj adjoint derivatives.

Returns a function with n_in + n_out +nadj*n_out inputs and nadj*n_in
outputs. The first n_in inputs correspond to nondifferentiated inputs. The
next n_out inputs correspond to nondifferentiated outputs. and the last
nadj*n_out inputs correspond to adjoint seeds, one direction at a time The
nadj*n_in outputs correspond to adjoint sensitivities, one direction at a
time. * (n_in = n_in(), n_out = n_out())

(n_in = n_in(), n_out = n_out())

The functions returned are cached, meaning that if called multiple timed
with the same value, then multiple references to the same function will be
returned.

";

%feature("docstring") casadi::Function "

General function.

A general function $f$ in casadi can be multi-input, multi-output. Number of
inputs: nin n_in() Number of outputs: nout n_out()  We can view this
function as a being composed of a ( nin, nout) grid of single-input, single-
output primitive functions. Each such primitive function $f_ {i, j}
\\\\forall i \\\\in [0, nin-1], j \\\\in [0, nout-1]$ can map as $\\\\mathbf
{R}^{n, m}\\\\to\\\\mathbf{R}^{p, q}$, in which n, m, p, q can take
different values for every (i, j) pair.  When passing input, you specify
which partition $i$ is active. You pass the numbers vectorized, as a vector
of size $(n*m)$. When requesting output, you specify which partition $j$ is
active. You get the numbers vectorized, as a vector of size $(p*q)$.  To
calculate Jacobians, you need to have $(m=1, q=1)$.

Write the Jacobian as $J_ {i, j} = \\\\nabla f_{i, j} = \\\\frac
{\\\\partial f_{i, j}(\\\\vec{x})}{\\\\partial \\\\vec{x}}$.

We have the following relationships for function mapping from a row vector
to a row vector:

$ \\\\vec {s}_f = \\\\nabla f_{i, j} . \\\\vec{v}$ $ \\\\vec {s}_a =
(\\\\nabla f_{i, j})^T . \\\\vec{w}$

Some quantities in these formulas must be transposed: input col: transpose $
\\\\vec {v} $ and $\\\\vec{s}_a$ output col: transpose $ \\\\vec {w} $ and
$\\\\vec{s}_f$  NOTE: Functions are allowed to modify their input arguments
when evaluating: implicitFunction, IDAS solver Further releases may disallow
this.

Joel Andersson

C++ includes: function.hpp ";

%feature("docstring") casadi::Function::set_jac_sparsity "

Generate the sparsity of a Jacobian block

";

%feature("docstring") casadi::Function::tangent "

Generate a tangent function of output oind with respect to input iind.

Parameters:
-----------

iind:  The index of the input

oind:  The index of the output

The default behavior of this class is defined by the derived class. Note
that the input must be scalar. In other cases, use the Jacobian instead.

";

%feature("docstring") casadi::Function::getAtomicInputReal "

Get the floating point output argument of an atomic operation.

";

%feature("docstring") casadi::Function::setJacobian "

Set the Jacobian function of output oind with respect to input iind NOTE:
Does not take ownership, only weak references to the Jacobians are kept
internally

";

%feature("docstring") casadi::Function::generate "

Export / Generate C code for the function.

";

%feature("docstring") casadi::Function::rootfinder_fun "

Access rhs function for a rootfinder.

";

%feature("docstring") casadi::Function::has_free "

Does the function have free variables.

";

%feature("docstring") casadi::Function::getAtomicOutput "

Get the (integer) output argument of an atomic operation.

";

%feature("docstring") casadi::Function::release "

Release a memory object.

";

%feature("docstring") casadi::Function::size_in "

Get input dimension.

";

%feature("docstring") casadi::Function::call "

Evaluate the function symbolically or numerically.

";

%feature("docstring") casadi::Function::size2_out "

Get output dimension.

";

%feature("docstring") casadi::Function::size1_out "

Get output dimension.

";

%feature("docstring") casadi::Function::generate_lifted "

Extract the functions needed for the Lifted Newton method.

";

%feature("docstring") casadi::Function::sparsity_in "

Get sparsity of a given input.

";

%feature("docstring") casadi::Function::sparsity_jac "

Get, if necessary generate, the sparsity of a Jacobian block

";

%feature("docstring") casadi::Function::linsol_cholesky "

Obtain a numeric Cholesky factorization Only for Cholesky solvers.

";

%feature("docstring") casadi::Function::getWorkSize "

Get the length of the work vector.

";

%feature("docstring") casadi::Function::sz_arg "[INTERNAL]  Get required
length of arg field.

";

%feature("docstring") casadi::Function::size2_in "

Get input dimension.

";

%feature("docstring") casadi::Function::removeMonitor "

Remove modules to be monitored.

";

%feature("docstring") casadi::Function::spCanEvaluate "[INTERNAL]  Is the
class able to propagate seeds through the algorithm?

(for usage, see the example propagating_sparsity.cpp)

";

%feature("docstring") casadi::Function::sx_out "

Get symbolic primitives equivalent to the output expressions There is no
guarantee that subsequent calls return unique answers.

";

%feature("docstring") casadi::Function::default_in "

Get default input value (NOTE: constant reference)

";

%feature("docstring") casadi::Function::getDescription "

Return a string with a description (for SWIG)

";


// File: classcasadi_1_1GenericCall.xml


// File: classcasadi_1_1GenericExpression.xml
%feature("docstring") friendwrap_floor "

Round down to nearest integer.

";

%feature("docstring") friendwrap_acos "

Arc cosine.

";

%feature("docstring") friendwrap_if_else_zero "

Conditional assignment.

";

%feature("docstring") friendwrap_exp "

Exponential function.

";

%feature("docstring") friendwrap_ceil "

Round up to nearest integer.

";

%feature("docstring") friendwrap_printme "

Debug printing.

";

%feature("docstring") friendwrap_cos "

Cosine.

";

%feature("docstring") friendwrap_asinh "

Inverse hyperbolic sine.

";

%feature("docstring") friendwrap_atanh "

Inverse hyperbolic tangent.

";

%feature("docstring") friendwrap_tan "

Tangent.

";

%feature("docstring") friendwrap_acosh "

Inverse hyperbolic cosine.

";

%feature("docstring") friendwrap_erfinv "

Invers error function.

";

%feature("docstring") friendwrap_fmod "

Remainder after division.

";

%feature("docstring") friendwrap_constpow "

Elementwise power with const power.

";

%feature("docstring") friendwrap_log "

Natural logarithm.

";

%feature("docstring") friendwrap_log10 "

Base-10 logarithm.

";

%feature("docstring") friendwrap_copysign "

Copy sign.

";

%feature("docstring") friendwrap_abs "

Absolute value.

";

%feature("docstring") friendwrap_fmax "

Largest of two values.

";

%feature("docstring") friendwrap_sqrt "

Square root.

";

%feature("docstring") friendwrap_sign "

Sine function sign(x) := -1 for x<0 sign(x) := 1 for x>0, sign(0) := 0
sign(NaN) := NaN

";

%feature("docstring") friendwrap_logic_and "

Logical and, alternative syntax.

";

%feature("docstring") friendwrap_fmin "

Smallest of two values.

";

%feature("docstring") friendwrap_erf "

Error function.

";

%feature("docstring") friendwrap_pow "

Elementwise power.

";

%feature("docstring") friendwrap_atan2 "

Two argument arc tangent.

";

%feature("docstring") friendwrap_logic_or "

Logical or, alterntive syntax.

";

%feature("docstring") friendwrap_fabs "

Absolute value.

";

%feature("docstring") friendwrap_sq "

Square.

";

%feature("docstring") friendwrap_sinh "

Hyperbolic sine.

";

%feature("docstring") friendwrap_tanh "

Hyperbolic tangent.

";

%feature("docstring") friendwrap_cosh "

Hyperbolic cosine.

";

%feature("docstring") friendwrap_logic_not "

Logical not, alternative syntax.

";

%feature("docstring") friendwrap_atan "

Arc tangent.

";

%feature("docstring") casadi::GenericExpression "

Expression interface.

This is a common base class for SX, MX and Matrix<>, introducing a uniform
syntax and implementing common functionality using the curiously recurring
template pattern (CRTP) idiom. Joel Andersson

C++ includes: generic_expression.hpp ";

%feature("docstring") friendwrap_is_equal "

Check if two nodes are equivalent up to a given depth. Depth=0 checks if the
expressions are identical, i.e. points to the same node.

a = x*x b = x*x

a.is_equal(b, 0) will return false, but a.is_equal(b, 1) will return true

";

%feature("docstring") friendwrap_sin "

Sine.

";

%feature("docstring") friendwrap_asin "

Arc sine.

";


// File: classcasadi_1_1GenericExternal.xml


// File: classcasadi_1_1GenericMatrix.xml


/*  Construct symbolic primitives  */

/* The \"sym\" function is intended to work in a similar way as \"sym\" used
in the Symbolic Toolbox for Matlab but instead creating a CasADi symbolic
primitive.

*/ %feature("docstring") casadi::GenericMatrix::nnz_upper "

Get the number of non-zeros in the upper triangular half.

";

%feature("docstring") casadi::GenericMatrix::size2 "

Get the second dimension (i.e. number of columns)

";

%feature("docstring") friendwrap_norm_inf "

Infinity-norm.

";

%feature("docstring") friendwrap_depends_on "

Check if expression depends on the argument The argument must be symbolic.

";

%feature("docstring") friendwrap_sum2 "

Return a column-wise summation of elements.

";

%feature("docstring") friendwrap_sum1 "

Return a row-wise summation of elements.

";

%feature("docstring") friendwrap_n_nodes "

Count number of nodes

";

%feature("docstring") casadi::GenericMatrix::bilin "

Matrix power x^n.

";

%feature("docstring") friendwrap_mrdivide "

Matrix divide (cf. slash '/' in MATLAB)

";

%feature("docstring") friendwrap_linspace "

Matlab's linspace command.

";

%feature("docstring") friendwrap_mldivide "

Matrix divide (cf. backslash '\\\\' in MATLAB)

";

%feature("docstring") friendwrap_sum_square "

Calculate some of squares: sum_ij X_ij^2.

";

%feature("docstring") casadi::GenericMatrix::get_colind "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") friendwrap_cross "

Matlab's cross command.

";

%feature("docstring") casadi::GenericMatrix::is_scalar "

Check if the matrix expression is scalar.

";

%feature("docstring") friendwrap_shared "

Extract shared subexpressions from an set of expressions.

";

%feature("docstring") friendwrap_solve "

>  MatType solve(MatType A, MatType b)
------------------------------------------------------------------------

Solve a system of equations: A*x = b The solve routine works similar to
Matlab's backslash when A is square and nonsingular. The algorithm used is
the following:

A simple forward or backward substitution if A is upper or lower triangular

If the linear system is at most 3-by-3, form the inverse via minor expansion
and multiply

Permute the variables and equations as to get a (structurally) nonzero
diagonal, then perform a QR factorization without pivoting and solve the
factorized system.

Note 1: If there are entries of the linear system known to be zero, these
will be removed. Elements that are very small, or will evaluate to be zero,
can still cause numerical errors, due to the lack of pivoting (which is not
possible since cannot compare the size of entries)

Note 2: When permuting the linear system, a BLT (block lower triangular)
transformation is formed. Only the permutation part of this is however used.
An improvement would be to solve block-by-block if there are multiple BLT
blocks.

>  MatType solve(MatType A, MatType b, str lsolver, Dict dict=Dict())
------------------------------------------------------------------------

Solve a system of equations: A*x = b.

";

%feature("docstring") friendwrap_det "

Matrix determinant (experimental)

";

%feature("docstring") friendwrap_nl_var "

Find out which variables enter nonlinearly.

";

%feature("docstring") casadi::GenericMatrix::sparsity "

Get the sparsity pattern.

";

%feature("docstring") casadi::GenericMatrix::is_column "

Check if the matrix is a column vector (i.e. size2()==1)

";

%feature("docstring") casadi::GenericMatrix::is_triu "

Check if the matrix is upper triangular.

";

%feature("docstring") casadi::GenericMatrix::is_empty "

Check if the sparsity is empty, i.e. if one of the dimensions is zero (or
optionally both dimensions)

";

%feature("docstring") friendwrap_conditional "

Create a switch.

If the condition

Parameters:
-----------

ind:  evaluates to the integer k, where 0<=k<f.size(), then x[k] will be
returned, otherwise

x_default:  will be returned.

";

%feature("docstring") friendwrap_densify "

>  MatType densify(MatType x)
------------------------------------------------------------------------

Make the matrix dense if not already.

>  MatType densify(MatType x, MatType val)
------------------------------------------------------------------------

Make the matrix dense and assign nonzeros to a value.

";

%feature("docstring") friendwrap_mpower "

Matrix power x^n.

";

%feature("docstring") casadi::GenericMatrix::is_row "

Check if the matrix is a row vector (i.e. size1()==1)

";

%feature("docstring") casadi::GenericMatrix::is_tril "

Check if the matrix is lower triangular.

";

%feature("docstring") friendwrap_tril2symm "

Convert a lower triangular matrix to a symmetric one.

";

%feature("docstring") friendwrap_norm_F "

Frobenius norm.

";

%feature("docstring") friendwrap_repsum "

Given a repeated matrix, computes the sum of repeated parts.

";

%feature("docstring") friendwrap_inv_skew "

Generate the 3-vector progenitor of a skew symmetric matrix.

";

%feature("docstring") friendwrap_dot "

Inner product of two matrices with x and y matrices of the same dimension.

";

%feature("docstring") casadi::GenericMatrix::colind "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") casadi::GenericMatrix::nnz_diag "

Get get the number of non-zeros on the diagonal.

";

%feature("docstring") friendwrap_rank1 "

Make a rank-1 update to a matrix A Calculates A + 1/2 * alpha * x*y'.

";

%feature("docstring") casadi::GenericMatrix::size "

>  (int,int) MatType .size() const 
------------------------------------------------------------------------

Get the shape.

>  int MatType .size(int axis) const 
------------------------------------------------------------------------

Get the size along a particular dimensions.

";

%feature("docstring") friendwrap_jtimes "

Calculate the Jacobian and multiply by a vector from the right This is
equivalent to mul(jacobian(ex, arg), v) or mul(jacobian(ex, arg).T, v) for
tr set to false and true respectively. If contrast to these expressions, it
will use directional derivatives which is typically (but not necessarily)
more efficient if the complete Jacobian is not needed and v has few rows.

";

%feature("docstring") casadi::GenericMatrix "

Matrix base class.

This is a common base class for MX and Matrix<>, introducing a uniform
syntax and implementing common functionality using the curiously recurring
template pattern (CRTP) idiom.  The class is designed with the idea that
\"everything is a matrix\", that is, also scalars and vectors. This
philosophy makes it easy to use and to interface in particularly with Python
and Matlab/Octave.  The syntax tries to stay as close as possible to the
ublas syntax when it comes to vector/matrix operations.  Index starts with
0. Index vec happens as follows: (rr, cc) -> k = rr+cc*size1() Vectors are
column vectors.  The storage format is Compressed Column Storage (CCS),
similar to that used for sparse matrices in Matlab, but unlike this format,
we do allow for elements to be structurally non-zero but numerically zero.
The sparsity pattern, which is reference counted and cached, can be accessed
with Sparsity& sparsity() Joel Andersson

C++ includes: generic_matrix.hpp ";

%feature("docstring") friendwrap_tangent "

Calculate Jacobian.

";

%feature("docstring") casadi::GenericMatrix::find "

Get the location of all non-zero elements as they would appear in a Dense
matrix A : DenseMatrix 4 x 3 B : SparseMatrix 4 x 3 , 5 structural non-
zeros.

k = A.find() A[k] will contain the elements of A that are non-zero in B

";

%feature("docstring") casadi::GenericMatrix::is_square "

Check if the matrix expression is square.

";

%feature("docstring") friendwrap_polyval "

Evaluate a polynomial with coefficients p in x.

";

%feature("docstring") casadi::GenericMatrix::dim "

Get string representation of dimensions. The representation is (nrow x ncol
= numel | size)

";

%feature("docstring") friendwrap_bilin "

Calculate bilinear form x^T A y.

";

%feature("docstring") friendwrap_symvar "

Get all symbols contained in the supplied expression Get all symbols on
which the supplied expression depends.

See:  SXFunction::getFree(), MXFunction::getFree()

";

%feature("docstring") casadi::GenericMatrix::is_vector "

Check if the matrix is a row or column vector.

";

%feature("docstring") friendwrap_norm_1 "

1-norm

";

%feature("docstring") friendwrap_unite "

Unite two matrices no overlapping sparsity.

";

%feature("docstring") casadi::GenericMatrix::get_row "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") friendwrap_trace "

Matrix trace.

";

%feature("docstring") friendwrap_gradient "

Calculate Jacobian.

";

%feature("docstring") friendwrap_print_operator "

Get a string representation for a binary MatType, using custom arguments.

";

%feature("docstring") friendwrap_substitute_inplace "

Inplace substitution with piggyback expressions Substitute variables v out
of the expressions vdef sequentially, as well as out of a number of other
expressions piggyback.

";

%feature("docstring") friendwrap_inv "

Matrix inverse (experimental)

";

%feature("docstring") casadi::GenericMatrix::size1 "

Get the first dimension (i.e. number of rows)

";

%feature("docstring") casadi::GenericMatrix::numel "

>  int MatType .numel() const 
------------------------------------------------------------------------

Get the number of elements.

>  int MatType .numel(int i) const 
------------------------------------------------------------------------

Get the number of elements in slice (cf. MATLAB)

";

%feature("docstring") casadi::GenericMatrix::zeros "

Create a dense matrix or a matrix with specified sparsity with all entries
zero.

";

%feature("docstring") friendwrap_substitute "

>  MatType substitute(MatType ex, MatType v, MatType vdef)
------------------------------------------------------------------------

Substitute variable v with expression vdef in an expression ex.

>  [MatType] substitute([MatType ] ex, [MatType ] v, [MatType ] vdef)
------------------------------------------------------------------------

Substitute variable var with expression expr in multiple expressions.

";

%feature("docstring") friendwrap_jacobian "

Calculate Jacobian.

";

%feature("docstring") casadi::GenericMatrix::is_dense "

Check if the matrix expression is dense.

";

%feature("docstring") friendwrap_nullspace "

Computes the nullspace of a matrix A.

Finds Z m-by-(m-n) such that AZ = 0 with A n-by-m with m > n

Assumes A is full rank

Inspired by Numerical Methods in Scientific Computing by Ake Bjorck

";

%feature("docstring") friendwrap_norm_2 "

2-norm

";

%feature("docstring") friendwrap_if_else "

Branching on MX nodes Ternary operator, \"cond ? if_true : if_false\".

";

%feature("docstring") friendwrap_simplify "

Simplify an expression.

";

%feature("docstring") friendwrap_diag "

Get the diagonal of a matrix or construct a diagonal When the input is
square, the diagonal elements are returned. If the input is vector- like, a
diagonal matrix is constructed with it.

";

%feature("docstring") casadi::GenericMatrix::nnz_lower "

Get the number of non-zeros in the lower triangular half.

";

%feature("docstring") friendwrap_triu2symm "

Convert a upper triangular matrix to a symmetric one.

";

%feature("docstring") friendwrap_hessian "";

%feature("docstring") casadi::GenericMatrix::nnz "

Get the number of (structural) non-zero elements.

";

%feature("docstring") casadi::GenericMatrix::sym "

>  static MatType MatType .sym(str name, int nrow=1, int ncol=1)
------------------------------------------------------------------------

Create an nrow-by-ncol symbolic primitive.

>  static MatType MatType .sym(str name, (int,int) rc)
------------------------------------------------------------------------

Construct a symbolic primitive with given dimensions.

>  MatType MatType .sym(str name, Sparsity sp)
------------------------------------------------------------------------

Create symbolic primitive with a given sparsity pattern.

>  [MatType ] MatType .sym(str name, Sparsity sp, int p)
------------------------------------------------------------------------

Create a vector of length p with with matrices with symbolic primitives of
given sparsity.

>  static[MatType ] MatType .sym(str name, int nrow, int ncol, int p)
------------------------------------------------------------------------

Create a vector of length p with nrow-by-ncol symbolic primitives.

>  [[MatType ] ] MatType .sym(str name, Sparsity sp, int p, int r)
------------------------------------------------------------------------

Create a vector of length r of vectors of length p with symbolic primitives
with given sparsity.

>  static[[MatType] ] MatType .sym(str name, int nrow, int ncol, int p, int r)
------------------------------------------------------------------------

Create a vector of length r of vectors of length p with nrow-by-ncol
symbolic primitives.

>  SX SX .sym(str name, Sparsity sp)
------------------------------------------------------------------------
[INTERNAL] 
";

%feature("docstring") friendwrap_project "

Create a new matrix with a given sparsity pattern but with the nonzeros
taken from an existing matrix.

";

%feature("docstring") casadi::GenericMatrix::row "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") friendwrap_skew "

Generate a skew symmetric matrix from a 3-vector.

";

%feature("docstring") friendwrap_pinv "

>  MatType pinv(MatType A)
------------------------------------------------------------------------

Computes the Moore-Penrose pseudo-inverse.

If the matrix A is fat (size1<size2), mul(A, pinv(A)) is unity.

pinv(A)' = (AA')^(-1) A

If the matrix A is slender (size1>size2), mul(pinv(A), A) is unity.

pinv(A) = (A'A)^(-1) A'

>  MatType pinv(MatType A, str lsolver, Dict dict=Dict())
------------------------------------------------------------------------

Computes the Moore-Penrose pseudo-inverse.

If the matrix A is fat (size1>size2), mul(A, pinv(A)) is unity. If the
matrix A is slender (size2<size1), mul(pinv(A), A) is unity.

";

%feature("docstring") casadi::GenericMatrix::ones "

Create a dense matrix or a matrix with specified sparsity with all entries
one.

";

%feature("docstring") casadi::GenericMatrix::rank1 "

Make a rank-1 update to a matrix A Calculates A + 1/2 * alpha * x*y'.

";


// File: classcasadi_1_1GenericType.xml
%feature("docstring") casadi::GenericType "

Generic data type, can hold different types such as bool, int, string etc.

Joel Andersson

C++ includes: generic_type.hpp ";


// File: classcasadi_1_1GenericTypeBase.xml


// File: classcasadi_1_1GetNonzeros.xml


// File: classcasadi_1_1GetNonzerosSlice.xml


// File: classcasadi_1_1GetNonzerosSlice2.xml


// File: classcasadi_1_1GetNonzerosVector.xml


// File: classcasadi_1_1GlobalOptions.xml
%feature("docstring") casadi::GlobalOptions "

Collects global CasADi options.

Note to developers: use sparingly. Global options are - in general - a
rather bad idea

this class must never be instantiated. Access its static members directly
Joris Gillis

C++ includes: global_options.hpp ";


// File: classcasadi_1_1Horzcat.xml


// File: classcasadi_1_1HorzRepmat.xml


// File: classcasadi_1_1HorzRepsum.xml


// File: classcasadi_1_1Horzsplit.xml


// File: classcasadi_1_1ImplicitFixedStepIntegrator.xml


// File: classcasadi_1_1ImplicitToNlp.xml


// File: classcasadi_1_1InfSX.xml


// File: classcasadi_1_1IntegerSX.xml


// File: classcasadi_1_1Integrator.xml


// File: classcasadi_1_1InterruptHandler.xml
%feature("docstring") casadi::InterruptHandler "[INTERNAL]  Takes care of
user interrupts (Ctrl+C)

This is an internal class.

Joris Gillis

C++ includes: casadi_interrupt.hpp ";


// File: classcasadi_1_1Inverse.xml


// File: classcasadi_1_1IpoptUserClass.xml
%feature("docstring") casadi::IpoptUserClass::get_starting_point "[INTERNAL]  Method to return the starting point for the algorithm

";

%feature("docstring") casadi::IpoptUserClass::finalize_solution "[INTERNAL]
This method is called when the algorithm is complete so the TNLP can
store/write the solution

";

%feature("docstring") casadi::IpoptUserClass "[INTERNAL] C++ includes:
ipopt_nlp.hpp ";

%feature("docstring")
casadi::IpoptUserClass::get_list_of_nonlinear_variables "[INTERNAL]
Specify which variables that appear in the Hessian

";

%feature("docstring") casadi::IpoptUserClass::eval_grad_f "[INTERNAL]
Method to return the gradient of the objective

";

%feature("docstring") casadi::IpoptUserClass::get_var_con_metadata "[INTERNAL]  Allows setting information about variables and constraints

";

%feature("docstring") casadi::IpoptUserClass::~IpoptUserClass "[INTERNAL]
";

%feature("docstring") casadi::IpoptUserClass::eval_g "[INTERNAL]  Method to
return the constraint residuals

";

%feature("docstring") casadi::IpoptUserClass::get_nlp_info "[INTERNAL]
Method to return some info about the nlp

";

%feature("docstring") casadi::IpoptUserClass::eval_f "[INTERNAL]  Method to
return the objective value

";

%feature("docstring")
casadi::IpoptUserClass::get_number_of_nonlinear_variables "[INTERNAL]
Specify the number of variables that appear in the Hessian

";

%feature("docstring") casadi::IpoptUserClass::eval_jac_g "[INTERNAL]
Method to return: 1) The structure of the Jacobian (if \"values\" is NULL)
2) The values of the Jacobian (if \"values\" is not NULL)

";

%feature("docstring") casadi::IpoptUserClass::finalize_metadata "[INTERNAL]
Retrieve information about variables and constraints

";

%feature("docstring") casadi::IpoptUserClass::get_bounds_info "[INTERNAL]
Method to return the bounds for my problem

";

%feature("docstring") casadi::IpoptUserClass::eval_h "[INTERNAL]  Method to
return: 1) The structure of the hessian of the Lagrangian (if \"values\" is
NULL) 2) The values of the hessian of the Lagrangian (if \"values\" is not
NULL)

";

%feature("docstring") casadi::IpoptUserClass::intermediate_callback "[INTERNAL]  This method is called at every iteration

";

%feature("docstring") casadi::IpoptUserClass::IpoptUserClass "[INTERNAL] ";


// File: classcasadi_1_1Jit.xml


// File: classcasadi_1_1JitLibrary.xml


// File: classcasadi_1_1KernelSum.xml


// File: classcasadi_1_1LapackLu.xml


// File: classcasadi_1_1LapackQr.xml


// File: classcasadi_1_1Library.xml
%feature("docstring") casadi::Library::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::Library::repr "

Print a representation of the object.

";

%feature("docstring") casadi::Library::is_null "

Is a null pointer?

";

%feature("docstring") casadi::Library::__hash__ "

Returns a number that is unique for a given Node. If the Object does not
point to any node, \"0\" is returned.

";

%feature("docstring") casadi::Library::has "";

%feature("docstring") casadi::Library "

Library, either just-in-time compiled or dynamically loaded.

C++ includes: compiler.hpp ";

%feature("docstring") casadi::Library::Library "

>  Library()
------------------------------------------------------------------------

Default constructor.

";

%feature("docstring") casadi::Library::print "

Print a description of the object.

";

%feature("docstring") casadi::Library::printPtr "[INTERNAL]  Print the
pointer to the internal class

";

%feature("docstring") casadi::Library::getRepresentation "

Return a string with a representation (for SWIG)

";


// File: classcasadi_1_1Linsol.xml


// File: classcasadi_1_1Logger.xml
%feature("docstring") casadi::Logger "

Keeps track of logging output to screen and/or files. All printout from
CasADi routines should go through this files.

Joel Andersson

C++ includes: casadi_logger.hpp ";


// File: classcasadi_1_1Mapaccum.xml


// File: classcasadi_1_1MapBase.xml


// File: classcasadi_1_1MapSerial.xml


// File: classcasadi_1_1MapSum.xml


// File: classcasadi_1_1MapSumSerial.xml


// File: singletoncasadi_1_1Matrix.xml


/*  Construct symbolic primitives  */

/* The \"sym\" function is intended to work in a similar way as \"sym\" used
in the Symbolic Toolbox for Matlab but instead creating a CasADi symbolic
primitive.

*/ %feature("docstring") casadi::Matrix::print_scalar "

Print scalar.

";

%feature("docstring") casadi::Matrix::nnz_upper "

Get the number of non-zeros in the upper triangular half.

";

%feature("docstring") casadi::Matrix::repr "

Print a representation of the object.

";

%feature("docstring") friendwrap_expand "[INTERNAL]  Expand the expression
as a weighted sum (with constant weights)

";

%feature("docstring") casadi::Matrix::is_constant "

Check if the matrix is constant (note that false negative answers are
possible)

";

%feature("docstring") friendwrap_mtaylor "

>  array(Scalar)  array(Scalar) .mtaylor(array(Scalar) ex, array(Scalar) x, array(Scalar) a, int order=1)
------------------------------------------------------------------------
[INTERNAL] 
multivariate Taylor series expansion

Do Taylor expansions until the aggregated order of a term is equal to
'order'. The aggregated order of $x^n y^m$ equals $n+m$.

>  array(Scalar)  array(Scalar) .mtaylor(array(Scalar) ex, array(Scalar) x, array(Scalar) a, int order, [int ] order_contributions)
------------------------------------------------------------------------
[INTERNAL] 
multivariate Taylor series expansion

Do Taylor expansions until the aggregated order of a term is equal to
'order'. The aggregated order of $x^n y^m$ equals $n+m$.

The argument order_contributions can denote how match each variable
contributes to the aggregated order. If x=[x, y] and order_contributions=[1,
2], then the aggregated order of $x^n y^m$ equals $1n+2m$.

Example usage

$ \\\\sin(b+a)+\\\\cos(b+a)(x-a)+\\\\cos(b+a)(y-b) $ $ y+x-(x^3+3y x^2+3 y^2
x+y^3)/6 $ $ (-3 x^2 y-x^3)/6+y+x $

";

%feature("docstring") casadi::Matrix::grad "

Gradient expression.

";

%feature("docstring") casadi::Matrix::set "

>  void array(Scalar) .set(array(Scalar) m, bool ind1, Slice rr)

>  void array(Scalar) .set(array(Scalar) m, bool ind1, IM rr)

>  void array(Scalar) .set(array(Scalar) m, bool ind1, Sparsity sp)
------------------------------------------------------------------------

Set a submatrix, single argument

>  void array(Scalar) .set(array(Scalar) m, bool ind1, Slice rr, Slice cc)

>  void array(Scalar) .set(array(Scalar) m, bool ind1, Slice rr, IM cc)

>  void array(Scalar) .set(array(Scalar) m, bool ind1, IM rr, Slice cc)

>  void array(Scalar) .set(array(Scalar) m, bool ind1, IM rr, IM cc)
------------------------------------------------------------------------

Set a submatrix, two arguments

";

%feature("docstring") casadi::Matrix::nnz "

Get the number of (structural) non-zero elements.

";

%feature("docstring") casadi::Matrix::remove "

Remove columns and rows Remove/delete rows and/or columns of a matrix.

";

%feature("docstring") casadi::Matrix::get "

>  void array(Scalar) .get(array(Scalar) output_m, bool ind1, Slice rr) const

>  void array(Scalar) .get(array(Scalar) output_m, bool ind1, IM rr) const

>  void array(Scalar) .get(array(Scalar) output_m, bool ind1, Sparsity sp) const 
------------------------------------------------------------------------

Get a submatrix, single argument

>  void array(Scalar) .get(array(Scalar) output_m, bool ind1, Slice rr, Slice cc) const

>  void array(Scalar) .get(array(Scalar) output_m, bool ind1, Slice rr, IM cc) const

>  void array(Scalar) .get(array(Scalar) output_m, bool ind1, IM rr, Slice cc) const

>  void array(Scalar) .get(array(Scalar) output_m, bool ind1, IM rr, IM cc) const 
------------------------------------------------------------------------

Get a submatrix, two arguments

";

%feature("docstring") friendwrap_triangle "[INTERNAL]  triangle function

\\\\[ \\\\begin {cases} \\\\Lambda(x) = 0 & |x| >= 1 \\\\\\\\ \\\\Lambda(x)
= 1-|x| & |x| < 1 \\\\end {cases} \\\\]

";

%feature("docstring") friendwrap_adj "[INTERNAL]   Matrix adjoint.

";

%feature("docstring") casadi::Matrix::triplet "";

%feature("docstring") casadi::Matrix::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::Matrix::sanity_check "

Check if the dimensions and colind, row vectors are compatible.

Parameters:
-----------

complete:  set to true to also check elementwise throws an error as possible
result

";

%feature("docstring") casadi::Matrix::print_split "

Get strings corresponding to the nonzeros and the interdependencies.

";

%feature("docstring") casadi::Matrix::unary "[INTERNAL]  Create nodes by
their ID.

";

%feature("docstring") casadi::Matrix::is_tril "

Check if the matrix is lower triangular.

";

%feature("docstring") friendwrap_norm_inf_mul "[INTERNAL]  Inf-norm of a
Matrix-Matrix product.

";

%feature("docstring") casadi::Matrix::is_integer "

Check if the matrix is integer-valued (note that false negative answers are
possible)

";

%feature("docstring") casadi::Matrix::numel "

>  int array(Scalar) .numel() const
------------------------------------------------------------------------

Get the number of elements.

>  int array(Scalar) .numel(int i) const
------------------------------------------------------------------------

Get the number of elements in slice (cf. MATLAB)

";

%feature("docstring") casadi::Matrix::is_triu "

Check if the matrix is upper triangular.

";

%feature("docstring") friendwrap_all "[INTERNAL]  Returns true only if
every element in the matrix is true.

";

%feature("docstring") casadi::Matrix::find "

Get the location of all non-zero elements as they would appear in a Dense
matrix A : DenseMatrix 4 x 3 B : SparseMatrix 4 x 3 , 5 structural non-
zeros.

k = A.find() A[k] will contain the elements of A that are non-zero in B

";

%feature("docstring") casadi::Matrix::is_regular "

Checks if expression does not contain NaN or Inf.

";

%feature("docstring") casadi::Matrix::set_nz "

Set a set of nonzeros

";

%feature("docstring") casadi::Matrix::nnz_diag "

Get get the number of non-zeros on the diagonal.

";

%feature("docstring") casadi::Matrix::sparsity "

Get the sparsity pattern.

";

%feature("docstring") casadi::Matrix::get_sparsity "

Get an owning reference to the sparsity pattern.

";

%feature("docstring") casadi::Matrix::bilin "

Matrix power x^n.

";

%feature("docstring") casadi::Matrix::dim "

Get string representation of dimensions. The representation is (nrow x ncol
= numel | size)

";

%feature("docstring") casadi::Matrix::get_nz "

Get a set of nonzeros

";

%feature("docstring") casadi::Matrix::T "

Transpose the matrix.

";

%feature("docstring") friendwrap_any "[INTERNAL]  Returns true only if any
element in the matrix is true.

";

%feature("docstring") casadi::Matrix::is_smooth "

Check if smooth.

";

%feature("docstring") casadi::Matrix::clear "";

%feature("docstring") casadi::Matrix::is_minus_one "

check if the matrix is -1 (note that false negative answers are possible)

";

%feature("docstring") friendwrap_poly_roots "[INTERNAL]  Attempts to find
the roots of a polynomial.

This will only work for polynomials up to order 3 It is assumed that the
roots are real.

";

%feature("docstring") casadi::Matrix::setScientific "

Set the 'precision, width & scientific' used in printing and serializing to
streams.

";

%feature("docstring") casadi::Matrix::n_dep "

Get the number of dependencies of a binary SXElem Only defined if symbolic
scalar.

";

%feature("docstring") casadi::Matrix::hess "

Hessian expression

";

%feature("docstring") casadi::Matrix::is_dense "

Check if the matrix expression is dense.

";

%feature("docstring") casadi::Matrix::print_sparse "

Print sparse matrix style.

";

%feature("docstring") casadi::Matrix::nnz_lower "

Get the number of non-zeros in the lower triangular half.

";

%feature("docstring") casadi::Matrix::reserve "";

%feature("docstring") casadi::Matrix::tang "

Tangent expression.

";

%feature("docstring") casadi::Matrix::is_one "

check if the matrix is 1 (note that false negative answers are possible)

";

%feature("docstring") casadi::Matrix::inf "

create a matrix with all inf

";

%feature("docstring") casadi::Matrix::element_hash "

Returns a number that is unique for a given symbolic scalar.

Only defined if symbolic scalar.

";

%feature("docstring") casadi::Matrix::scalar_matrix "[INTERNAL]  Create
nodes by their ID.

";

%feature("docstring") friendwrap_ramp "[INTERNAL]  ramp function

\\\\[ \\\\begin {cases} R(x) = 0 & x <= 1 \\\\\\\\ R(x) = x & x > 1 \\\\\\\\
\\\\end {cases} \\\\]

Also called: slope function

";

%feature("docstring") casadi::Matrix::print_vector "

Print vector-style.

";

%feature("docstring") casadi::Matrix::ones "

Create a dense matrix or a matrix with specified sparsity with all entries
one.

";

%feature("docstring") casadi::Matrix::is_column "

Check if the matrix is a column vector (i.e. size2()==1)

";

%feature("docstring") casadi::Matrix::resize "";

%feature("docstring") friendwrap_qr "[INTERNAL]  QR factorization using the
modified Gram-Schmidt algorithm More stable than the classical Gram-Schmidt,
but may break down if the rows of A are nearly linearly dependent See J.
Demmel: Applied Numerical Linear Algebra (algorithm 3.1.). Note that in
SWIG, Q and R are returned by value.

";

%feature("docstring") casadi::Matrix::row "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") casadi::Matrix::zeros "

Create a dense matrix or a matrix with specified sparsity with all entries
zero.

";

%feature("docstring") casadi::Matrix::rank1 "

Make a rank-1 update to a matrix A Calculates A + 1/2 * alpha * x*y'.

";

%feature("docstring") casadi::Matrix "

Sparse matrix class. SX and DM are specializations.

General sparse matrix class that is designed with the idea that \"everything
is a matrix\", that is, also scalars and vectors. This philosophy makes it
easy to use and to interface in particularly with Python and Matlab/Octave.
Index starts with 0. Index vec happens as follows: (rr, cc) -> k =
rr+cc*size1() Vectors are column vectors.  The storage format is Compressed
Column Storage (CCS), similar to that used for sparse matrices in Matlab,
but unlike this format, we do allow for elements to be structurally non-zero
but numerically zero.  Matrix<Scalar> is polymorphic with a
std::vector<Scalar> that contain all non-identical-zero elements. The
sparsity can be accessed with Sparsity& sparsity() Joel Andersson

C++ includes: casadi_types.hpp ";

%feature("docstring") casadi::Matrix::dep "

Get expressions of the children of the expression Only defined if symbolic
scalar. Wraps SXElem SXElem::dep(int ch=0) const.

";

%feature("docstring") casadi::Matrix::sym "

>  static array(Scalar)   array(Scalar) .sym(str name, int nrow=1, int ncol=1)
------------------------------------------------------------------------

Create an nrow-by-ncol symbolic primitive.

>  static array(Scalar)   array(Scalar) .sym(str name, (int,int) rc)
------------------------------------------------------------------------

Construct a symbolic primitive with given dimensions.

>  static array(Scalar)   array(Scalar) .sym(str name, Sparsity sp)
------------------------------------------------------------------------

Create symbolic primitive with a given sparsity pattern.

>  static[array(Scalar)   ] array(Scalar) .sym(str name, Sparsity sp, int p)
------------------------------------------------------------------------

Create a vector of length p with with matrices with symbolic primitives of
given sparsity.

>  static[array(Scalar)   ] array(Scalar) .sym(str name, int nrow, int ncol, int p)
------------------------------------------------------------------------

Create a vector of length p with nrow-by-ncol symbolic primitives.

>  static[[array(Scalar)  ] ] array(Scalar) .sym(str name, Sparsity sp, int p, int r)
------------------------------------------------------------------------

Create a vector of length r of vectors of length p with symbolic primitives
with given sparsity.

>  static[[array(Scalar)  ] ] array(Scalar) .sym(str name, int nrow, int ncol, int p, int r)
------------------------------------------------------------------------

Create a vector of length r of vectors of length p with nrow-by-ncol
symbolic primitives.

";

%feature("docstring") casadi::Matrix::is_leaf "

Check if SX is a leaf of the SX graph.

Only defined if symbolic scalar.

";

%feature("docstring") casadi::Matrix::jac "

Jacobian expression.

";

%feature("docstring") friendwrap_sparsify "[INTERNAL]  Make a matrix sparse
by removing numerical zeros.

";

%feature("docstring") friendwrap_cofactor "[INTERNAL]  Get the (i,j)
cofactor matrix.

";

%feature("docstring") casadi::Matrix::colind "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") casadi::Matrix::print_dense "

Print dense matrix-stype.

";

%feature("docstring") casadi::Matrix::has_nz "

Returns true if the matrix has a non-zero at location rr, cc.

";

%feature("docstring") casadi::Matrix::matrix_scalar "[INTERNAL]  Create
nodes by their ID.

";

%feature("docstring") casadi::Matrix::nan "

create a matrix with all nan

";

%feature("docstring") casadi::Matrix::is_commutative "

Check whether a binary SX is commutative.

Only defined if symbolic scalar.

";

%feature("docstring") casadi::Matrix::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::Matrix::is_valid_input "

Check if matrix can be used to define function inputs. Sparse matrices can
return true if all non-zero elements are symbolic.

";

%feature("docstring") friendwrap_getMinor "[INTERNAL]  Get the (i,j) minor
matrix.

";

%feature("docstring") friendwrap_heaviside "[INTERNAL]  Heaviside function.

\\\\[ \\\\begin {cases} H(x) = 0 & x<0 \\\\\\\\ H(x) = 1/2 & x=0 \\\\\\\\
H(x) = 1 & x>0 \\\\\\\\ \\\\end {cases} \\\\]

";

%feature("docstring") casadi::Matrix::setPrecision "

Set the 'precision, width & scientific' used in printing and serializing to
streams.

";

%feature("docstring") casadi::Matrix::size "

>  (int,int) array(Scalar) .size() const
------------------------------------------------------------------------

Get the shape.

>  int array(Scalar) .size(int axis) const
------------------------------------------------------------------------

Get the size along a particular dimensions.

";

%feature("docstring") casadi::Matrix::erase "

>  void array(Scalar) .erase([int ] rr, [int ] cc, bool ind1=false)
------------------------------------------------------------------------

Erase a submatrix (leaving structural zeros in its place) Erase rows and/or
columns of a matrix.

>  void array(Scalar) .erase([int ] rr, bool ind1=false)
------------------------------------------------------------------------

Erase a submatrix (leaving structural zeros in its place) Erase elements of
a matrix.

";

%feature("docstring") casadi::Matrix::is_row "

Check if the matrix is a row vector (i.e. size1()==1)

";

%feature("docstring") casadi::Matrix::__nonzero__ "

>  bool array(Scalar) .__nonzero__() const 
------------------------------------------------------------------------

Returns the truth value of a Matrix.

>  bool SX.__nonzero__() const
------------------------------------------------------------------------
[INTERNAL] 
";

%feature("docstring") casadi::Matrix::solve "[INTERNAL] ";

%feature("docstring") friendwrap_taylor "

>  array(Scalar)  array(Scalar) .taylor(array(Scalar) ex, array(Scalar) x, array(Scalar) a, int order=1)
------------------------------------------------------------------------
[INTERNAL] 
univariate Taylor series expansion

Calculate the Taylor expansion of expression 'ex' up to order 'order' with
respect to variable 'x' around the point 'a'

$(x)=f(a)+f'(a)(x-a)+f''(a)\\\\frac
{(x-a)^2}{2!}+f'''(a)\\\\frac{(x-a)^3}{3!}+\\\\ldots$

Example usage:

::

>>   x



>  array(Scalar)  taylor(array(Scalar) ex, array(Scalar) x)
------------------------------------------------------------------------

univariate Taylor series expansion

Calculate the Taylor expansion of expression 'ex' up to order 'order' with
respect to variable 'x' around the point 'a'

$(x)=f(a)+f'(a)(x-a)+f''(a)\\\\frac
{(x-a)^2}{2!}+f'''(a)\\\\frac{(x-a)^3}{3!}+\\\\ldots$

Example usage:

::

>>   x



";

%feature("docstring") casadi::Matrix::size2 "

Get the second dimension (i.e. number of columns)

";

%feature("docstring") casadi::Matrix::is_square "

Check if the matrix expression is square.

";

%feature("docstring") casadi::Matrix::size1 "

Get the first dimension (i.e. number of rows)

";

%feature("docstring") casadi::Matrix::is_empty "

Check if the sparsity is empty, i.e. if one of the dimensions is zero (or
optionally both dimensions)

";

%feature("docstring") casadi::Matrix::is_identity "

check if the matrix is an identity matrix (note that false negative answers
are possible)

";

%feature("docstring") friendwrap_chol "[INTERNAL]  Obtain a Cholesky
factorisation of a matrix Returns an upper triangular R such that R'R = A.
Matrix A must be positive definite.

At the moment, the algorithm is dense (Cholesky-Banachiewicz). There is an
open ticket #1212 to make it sparse.

";

%feature("docstring") friendwrap_poly_coeff "[INTERNAL]  extracts
polynomial coefficients from an expression

Parameters:
-----------

ex:  Scalar expression that represents a polynomial

x:  Scalar symbol that the polynomial is build up with

";

%feature("docstring") casadi::Matrix::printme "";

%feature("docstring") friendwrap_gauss_quadrature "

>  array(Scalar)  array(Scalar) .gauss_quadrature(array(Scalar) f, array(Scalar) x, array(Scalar) a, array(Scalar) b, int order=5)
------------------------------------------------------------------------
[INTERNAL] 
Integrate f from a to b using Gaussian quadrature with n points.

>  array(Scalar)  array(Scalar) .gauss_quadrature(array(Scalar) f, array(Scalar) x, array(Scalar) a, array(Scalar) b, int order, array(Scalar) w)
------------------------------------------------------------------------
[INTERNAL] 
 Matrix adjoint.

";

%feature("docstring") casadi::Matrix::pinv "[INTERNAL] ";

%feature("docstring") casadi::Matrix::get_row "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") friendwrap_pw_const "[INTERNAL]  Create a piecewise
constant function Create a piecewise constant function with n=val.size()
intervals.

Inputs:

Parameters:
-----------

t:  a scalar variable (e.g. time)

tval:  vector with the discrete values of t at the interval transitions
(length n-1)

val:  vector with the value of the function for each interval (length n)

";

%feature("docstring") casadi::Matrix::enlarge "

Enlarge matrix Make the matrix larger by inserting empty rows and columns,
keeping the existing non-zeros.

";

%feature("docstring") casadi::Matrix::is_symbolic "

Check if symbolic (Dense) Sparse matrices invariable return false.

";

%feature("docstring") friendwrap_pw_lin "[INTERNAL]  t a scalar variable
(e.g. time)

Create a piecewise linear function Create a piecewise linear function:

Inputs: tval vector with the the discrete values of t (monotonically
increasing) val vector with the corresponding function values (same length
as tval)

";

%feature("docstring") casadi::Matrix::setWidth "

Set the 'precision, width & scientific' used in printing and serializing to
streams.

";

%feature("docstring") casadi::Matrix::Matrix "

>  array(Scalar) ()
------------------------------------------------------------------------

constructors

empty 0-by-0 matrix constructor

>  array(Scalar) (array(Scalar) m)
------------------------------------------------------------------------

Copy constructor.

>  array(Scalar) (int nrow, int ncol)
------------------------------------------------------------------------

Create a sparse matrix with all structural zeros.

>  array(Scalar) (Sparsity sp)
------------------------------------------------------------------------

Create a sparse matrix from a sparsity pattern. Same as
Matrix::ones(sparsity)

>  array(Scalar) (Sparsity sp, array(Scalar) d)
------------------------------------------------------------------------

Construct matrix with a given sparsity and nonzeros.

>  array(Scalar) (double val)
------------------------------------------------------------------------

This constructor enables implicit type conversion from a numeric type.

>  array(Scalar) ([[double ] ] m)
------------------------------------------------------------------------

Dense matrix constructor with data given as vector of vectors.

>  array(Scalar) (array(A) x)
------------------------------------------------------------------------

Create a matrix from another matrix with a different entry type Assumes that
the scalar conversion is valid.

>  array(Scalar) ([A ] x)
------------------------------------------------------------------------

Create an expression from a vector.

>  array(Scalar) ([Scalar ] x)

>  array(Scalar) ((int,int) rc)

>  array(Scalar) (Sparsity sp, Scalar val, bool dummy)

>  array(Scalar) (Sparsity sp, [Scalar ] d, bool dummy)
------------------------------------------------------------------------
[INTERNAL] 
";

%feature("docstring") friendwrap_eig_symbolic "[INTERNAL]  Attempts to find
the eigenvalues of a symbolic matrix This will only work for up to 3x3
matrices.

";

%feature("docstring") friendwrap_rectangle "[INTERNAL]  rectangle function

\\\\[ \\\\begin {cases} \\\\Pi(x) = 1 & |x| < 1/2 \\\\\\\\ \\\\Pi(x) = 1/2 &
|x| = 1/2 \\\\\\\\ \\\\Pi(x) = 0 & |x| > 1/2 \\\\\\\\ \\\\end {cases} \\\\]

Also called: gate function, block function, band function, pulse function,
window function

";

%feature("docstring") casadi::Matrix::is_zero "

check if the matrix is 0 (note that false negative answers are possible)

";

%feature("docstring") casadi::Matrix::is_vector "

Check if the matrix is a row or column vector.

";

%feature("docstring") casadi::Matrix::matrix_matrix "[INTERNAL]  Create
nodes by their ID.

";

%feature("docstring") casadi::Matrix::binary "[INTERNAL]  Create nodes by
their ID.

";

%feature("docstring") casadi::Matrix::get_nonzeros "[INTERNAL]  Get all
nonzeros.

";

%feature("docstring") casadi::Matrix::has_duplicates "[INTERNAL]  Detect
duplicate symbolic expressions If there are symbolic primitives appearing
more than once, the function will return true and the names of the duplicate
expressions will be printed to userOut<true, PL_WARN>(). Note: Will mark the
node using SXElem::setTemp. Make sure to call resetInput() after usage.

";

%feature("docstring") casadi::Matrix::name "

Get name (only if symbolic scalar)

";

%feature("docstring") casadi::Matrix::has_zeros "

Check if the matrix has any zero entries which are not structural zeros.

";

%feature("docstring") casadi::Matrix::is_scalar "

Check if the matrix expression is scalar.

";

%feature("docstring") casadi::Matrix::get_colind "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") casadi::Matrix::print "

Print a description of the object.

";

%feature("docstring") casadi::Matrix::resetInput "[INTERNAL]  Reset the
marker for an input expression.

";


// File: classcasadi_1_1MinusInfSX.xml


// File: classcasadi_1_1MinusOneSX.xml


// File: classcasadi_1_1Monitor.xml


// File: classcasadi_1_1MultipleOutput.xml


// File: classcasadi_1_1Multiplication.xml


// File: classcasadi_1_1MX.xml


/*  Construct symbolic primitives  */

/* The \"sym\" function is intended to work in a similar way as \"sym\" used
in the Symbolic Toolbox for Matlab but instead creating a CasADi symbolic
primitive.

*/ %feature("docstring") casadi::MX::get_output "

Get the index of evaluation output - only valid when is_calloutput() is
true.

";

%feature("docstring") casadi::MX::attachAssert "

returns itself, but with an assertion attached

If y does not evaluate to 1, a runtime error is raised

";

%feature("docstring") casadi::MX "

MX - Matrix expression.

The MX class is used to build up trees made up from MXNodes. It is a more
general graph representation than the scalar expression, SX, and much less
efficient for small objects. On the other hand, the class allows much more
general operations than does SX, in particular matrix valued operations and
calls to arbitrary differentiable functions.

The MX class is designed to have identical syntax with the Matrix<> template
class, and uses Matrix<double> as its internal representation of the values
at a node. By keeping the syntaxes identical, it is possible to switch from
one class to the other, as well as inlining MX functions to SXElem
functions.

Note that an operation is always \"lazy\", making a matrix multiplication
will create a matrix multiplication node, not perform the actual
multiplication.

Joel Andersson

C++ includes: mx.hpp ";

%feature("docstring") casadi::MX::grad "

Gradient expression.

";

%feature("docstring") casadi::MX::is_scalar "

Check if the matrix expression is scalar.

";

%feature("docstring") casadi::MX::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::MX::erase "

>  void MX.erase([int ] rr, [int ] cc, bool ind1=false)
------------------------------------------------------------------------

Erase a submatrix (leaving structural zeros in its place) Erase rows and/or
columns of a matrix.

>  void MX.erase([int ] rr, bool ind1=false)
------------------------------------------------------------------------

Erase a submatrix (leaving structural zeros in its place) Erase elements of
a matrix.

";

%feature("docstring") casadi::MX::monitor "

Monitor an expression Returns itself, but with the side effect of printing
the nonzeros along with a comment.

";

%feature("docstring") casadi::MX::primitives "

Get symbolic primitives.

";

%feature("docstring") casadi::MX::is_constant "

Check if constant.

";

%feature("docstring") casadi::MX::is_output "

Check if evaluation output.

";

%feature("docstring") casadi::MX::set "

>  void MX.set(MX m, bool ind1, Slice rr)

>  void MX.set(MX m, bool ind1, IM rr)

>  void MX.set(MX m, bool ind1, Sparsity sp)
------------------------------------------------------------------------

Set a submatrix, single argument

";

%feature("docstring") casadi::MX::is_valid_input "

Check if matrix can be used to define function inputs. Valid inputs for
MXFunctions are combinations of Reshape, concatenations and SymbolicMX.

";

%feature("docstring") casadi::MX::getTemp "[INTERNAL]  Get the temporary
variable

";

%feature("docstring") casadi::MX::binary "

Create nodes by their ID.

";

%feature("docstring") casadi::MX::is_commutative "

Check if commutative operation.

";

%feature("docstring") friendwrap_lift "

Lift the expression Experimental feature.

";

%feature("docstring") casadi::MX::is_tril "

Check if the matrix is lower triangular.

";

%feature("docstring") casadi::MX::size2 "

Get the second dimension (i.e. number of columns)

";

%feature("docstring") casadi::MX::size1 "

Get the first dimension (i.e. number of rows)

";

%feature("docstring") casadi::MX::is_op "

Is it a certain operation.

";

%feature("docstring") casadi::MX::is_triu "

Check if the matrix is upper triangular.

";

%feature("docstring") casadi::MX::~MX "[INTERNAL]  Destructor.

";

%feature("docstring") casadi::MX::set_nz "

Set a set of nonzeros

";

%feature("docstring") casadi::MX::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::MX::n_out "

Number of outputs.

";

%feature("docstring") casadi::MX::print "

Print a description of the object.

";

%feature("docstring") casadi::MX::is_one "

check if zero (note that false negative answers are possible)

";

%feature("docstring") casadi::MX::is_empty "

Check if the sparsity is empty, i.e. if one of the dimensions is zero (or
optionally both dimensions)

";

%feature("docstring") casadi::MX::get_nz "

Get a set of nonzeros

";

%feature("docstring") casadi::MX::ones "

Create a dense matrix or a matrix with specified sparsity with all entries
one.

";

%feature("docstring") casadi::MX::is_transpose "

Is the expression a transpose?

";

%feature("docstring") casadi::MX::is_binary "

Is binary operation.

";

%feature("docstring") casadi::MX::inf "

create a matrix with all inf

";

%feature("docstring") friendwrap_matrix_expand "

Expand MX graph to SXFunction call.

Expand the given expression e, optionally supplying expressions contained in
it at which expansion should stop.

";

%feature("docstring") casadi::MX::colind "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") casadi::MX::is_symbolic "

Check if symbolic.

";

%feature("docstring") casadi::MX::join_primitives "

Join an expression along symbolic primitives.

";

%feature("docstring") casadi::MX::is_null "

Is a null pointer?

";

%feature("docstring") casadi::MX::get_colind "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") casadi::MX::is_multiplication "

Check if multiplication.

";

%feature("docstring") casadi::MX::printPtr "[INTERNAL]  Print the pointer
to the internal class

";

%feature("docstring") casadi::MX::nnz "

Get the number of (structural) non-zero elements.

";

%feature("docstring") casadi::MX::find "

Get the location of all non-zero elements as they would appear in a Dense
matrix A : DenseMatrix 4 x 3 B : SparseMatrix 4 x 3 , 5 structural non-
zeros.

k = A.find() A[k] will contain the elements of A that are non-zero in B

";

%feature("docstring") casadi::MX::is_identity "

check if identity

";

%feature("docstring") casadi::MX::dim "

Get string representation of dimensions. The representation is (nrow x ncol
= numel | size)

";

%feature("docstring") friendwrap_find "

Find first nonzero If failed, returns the number of rows.

";

%feature("docstring") casadi::MX::is_minus_one "

check if zero (note that false negative answers are possible)

";

%feature("docstring") friendwrap_graph_substitute "

>  MX graph_substitute(MX ex, [MX ] v, [MX ] vdef)
------------------------------------------------------------------------

Substitute single expression in graph Substitute variable v with expression
vdef in an expression ex, preserving nodes.

>  [MX] graph_substitute([MX ] ex, [MX ] v, [MX ] vdef)
------------------------------------------------------------------------

Substitute multiple expressions in graph Substitute variable var with
expression expr in multiple expressions, preserving nodes.

";

%feature("docstring") casadi::MX::get_row "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") casadi::MX::sparsity "

Get the sparsity pattern.

";

%feature("docstring") casadi::MX::is_call "

Check if evaluation.

";

%feature("docstring") casadi::MX::nnz_upper "

Get the number of non-zeros in the upper triangular half.

";

%feature("docstring") casadi::MX::__hash__ "

Returns a number that is unique for a given Node. If the Object does not
point to any node, \"0\" is returned.

";

%feature("docstring") casadi::MX::jac "

Jacobian expression.

";

%feature("docstring") casadi::MX::getOutput "

Get an output.

";

%feature("docstring") casadi::MX::zeros "

Create a dense matrix or a matrix with specified sparsity with all entries
zero.

";

%feature("docstring") casadi::MX::is_zero "

check if zero (note that false negative answers are possible)

";

%feature("docstring") casadi::MX::size "

>  (int,int) MX .size() const
------------------------------------------------------------------------

Get the shape.

>  int MX .size(int axis) const
------------------------------------------------------------------------

Get the size along a particular dimensions.

";

%feature("docstring") casadi::MX::T "

Transpose the matrix.

";

%feature("docstring") casadi::MX::is_vector "

Check if the matrix is a row or column vector.

";

%feature("docstring") casadi::MX::printme "";

%feature("docstring") casadi::MX::nan "

create a matrix with all nan

";

%feature("docstring") casadi::MX::row "

Get the sparsity pattern. See the Sparsity class for details.

";

%feature("docstring") casadi::MX::n_dep "

Get the number of dependencies of a binary SXElem.

";

%feature("docstring") casadi::MX::has_duplicates "[INTERNAL]  Detect
duplicate symbolic expressions If there are symbolic primitives appearing
more than once, the function will return true and the names of the duplicate
expressions will be printed to userOut<true, PL_WARN>(). Note: Will mark the
node using MX::setTemp. Make sure to call resetInput() after usage.

";

%feature("docstring") casadi::MX::is_norm "

Check if norm.

";

%feature("docstring") casadi::MX::numel "

>  int MX .numel() const
------------------------------------------------------------------------

Get the number of elements.

>  int MX .numel(int i) const
------------------------------------------------------------------------

Get the number of elements in slice (cf. MATLAB)

";

%feature("docstring") casadi::MX::enlarge "

Enlarge matrix Make the matrix larger by inserting empty rows and columns,
keeping the existing non-zeros.

";

%feature("docstring") casadi::MX::sym "

>  static MX  MX .sym(str name, int nrow=1, int ncol=1)
------------------------------------------------------------------------

Create an nrow-by-ncol symbolic primitive.

>  static MX  MX .sym(str name, (int,int) rc)
------------------------------------------------------------------------

Construct a symbolic primitive with given dimensions.

>  static MX  MX .sym(str name, Sparsity sp)
------------------------------------------------------------------------

Create symbolic primitive with a given sparsity pattern.

>  static[MX  ] MX .sym(str name, Sparsity sp, int p)
------------------------------------------------------------------------

Create a vector of length p with with matrices with symbolic primitives of
given sparsity.

>  static[MX  ] MX .sym(str name, int nrow, int ncol, int p)
------------------------------------------------------------------------

Create a vector of length p with nrow-by-ncol symbolic primitives.

>  static[[MX ] ] MX .sym(str name, Sparsity sp, int p, int r)
------------------------------------------------------------------------

Create a vector of length r of vectors of length p with symbolic primitives
with given sparsity.

>  static[[MX ] ] MX .sym(str name, int nrow, int ncol, int p, int r)
------------------------------------------------------------------------

Create a vector of length r of vectors of length p with nrow-by-ncol
symbolic primitives.

";

%feature("docstring") casadi::MX::is_square "

Check if the matrix expression is square.

";

%feature("docstring") casadi::MX::get "

>  void MX.get(MX &output_m, bool ind1, Slice rr) const

>  void MX.get(MX &output_m, bool ind1, IM rr) const

>  void MX.get(MX &output_m, bool ind1, Sparsity sp) const 
------------------------------------------------------------------------

Get a submatrix, single argument

>  void MX.get(MX &output_m, bool ind1, Slice rr, Slice cc) const

>  void MX.get(MX &output_m, bool ind1, Slice rr, IM cc) const

>  void MX.get(MX &output_m, bool ind1, IM rr, Slice cc) const

>  void MX.get(MX &output_m, bool ind1, IM rr, IM cc) const 
------------------------------------------------------------------------

Get a submatrix, two arguments

";

%feature("docstring") casadi::MX::repr "

Print a representation of the object.

";

%feature("docstring") casadi::MX::getFunction "

Get function.

";

%feature("docstring") casadi::MX::numFunctions "

Number of functions.

";

%feature("docstring") casadi::MX::is_column "

Check if the matrix is a column vector (i.e. size2()==1)

";

%feature("docstring") casadi::MX::__nonzero__ "

Returns the truth value of an MX expression.

";

%feature("docstring") casadi::MX::resetInput "[INTERNAL]  Reset the marker
for an input expression.

";

%feature("docstring") casadi::MX::tang "

Tangent expression.

";

%feature("docstring") casadi::MX::name "

Get the name.

";

%feature("docstring") casadi::MX::MX "

>  MX()
------------------------------------------------------------------------

Default constructor.

>  MX(int nrow, int ncol)
------------------------------------------------------------------------

Create a sparse matrix with all structural zeros.

>  MX(Sparsity sp)
------------------------------------------------------------------------

Create a sparse matrix from a sparsity pattern. Same as MX::ones(sparsity)

>  MX(Sparsity sp, MX val)
------------------------------------------------------------------------

Construct matrix with a given sparsity and nonzeros.

>  MX(double x)
------------------------------------------------------------------------

Create scalar constant (also implicit type conversion)

>  MX(MX x)
------------------------------------------------------------------------

Copy constructor.

>  MX([double ] x)
------------------------------------------------------------------------

Create vector constant (also implicit type conversion)

>  MX(DM x)
------------------------------------------------------------------------

Create sparse matrix constant (also implicit type conversion)

";

%feature("docstring") casadi::MX::rank1 "

Make a rank-1 update to a matrix A Calculates A + 1/2 * alpha * x*y'.

";

%feature("docstring") casadi::MX::bilin "

Matrix power x^n.

";

%feature("docstring") casadi::MX::nnz_lower "

Get the number of non-zeros in the lower triangular half.

";

%feature("docstring") casadi::MX::is_unary "

Is unary operation.

";

%feature("docstring") casadi::MX::setTemp "[INTERNAL]  Set the temporary
variable.

";

%feature("docstring") casadi::MX::mapping "

Get an IM representation of a GetNonzeros or SetNonzeros node.

";

%feature("docstring") casadi::MX::split_primitives "

Split up an expression along symbolic primitives.

";

%feature("docstring") casadi::MX::is_dense "

Check if the matrix expression is dense.

";

%feature("docstring") casadi::MX::op "

Get operation type.

";

%feature("docstring") casadi::MX::n_primitives "

Get the number of symbolic primitive Assumes is_valid_input() returns true.

";

%feature("docstring") casadi::MX::dep "

Get the nth dependency as MX.

";

%feature("docstring") casadi::MX::get_sparsity "

Get an owning reference to the sparsity pattern.

";

%feature("docstring") casadi::MX::nnz_diag "

Get get the number of non-zeros on the diagonal.

";

%feature("docstring") casadi::MX::unary "

Create nodes by their ID.

";

%feature("docstring") casadi::MX::is_regular "

Checks if expression does not contain NaN or Inf.

";

%feature("docstring") casadi::MX::is_row "

Check if the matrix is a row vector (i.e. size1()==1)

";


// File: classcasadi_1_1MXFunction.xml


// File: classcasadi_1_1NanSX.xml


// File: classcasadi_1_1Newton.xml


// File: classcasadi_1_1NlpBuilder.xml


/*  Symbolic representation of the NLP  */

/* Data members

*/ %feature("docstring") casadi::NlpBuilder "

A symbolic NLP representation.

Joel Andersson

C++ includes: nlp_builder.hpp ";

%feature("docstring") casadi::NlpBuilder::parse_nl "

Parse an AMPL och PyOmo NL-file.

";

%feature("docstring") casadi::NlpBuilder::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::NlpBuilder::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::NlpBuilder::repr "

Print a representation of the object.

";

%feature("docstring") casadi::NlpBuilder::print "

Print a description of the object.

";


// File: classcasadi_1_1Nlpsol.xml


// File: classcasadi_1_1NonZeros.xml
%feature("docstring") casadi::NonZeros::NonZeros "

Constructor.

";

%feature("docstring") casadi::NonZeros "

Access to a set of nonzeros.

NonZeros class for Matrix NonZeros is the return type for operator[] of the
Matrix class, it allows access to the value as well as changing the parent
object Joel Andersson

C++ includes: nonzeros.hpp ";


// File: classcasadi_1_1Norm.xml


// File: classcasadi_1_1Norm1.xml


// File: classcasadi_1_1Norm2.xml


// File: classcasadi_1_1NormF.xml


// File: classcasadi_1_1NormInf.xml


// File: classcasadi_1_1OneSX.xml


// File: classcasadi_1_1ParsedFile.xml
%feature("docstring") casadi::ParsedFile::ParsedFile "

>  ParsedFile()
------------------------------------------------------------------------

Default constructor (no commands)

>  ParsedFile(str fname)

>  ParsedFile([str ] lines, int offset=0)
------------------------------------------------------------------------

Construct from a file.

";

%feature("docstring") casadi::ParsedFile::print "

Print parsed file.

";

%feature("docstring") casadi::ParsedFile::to "

Convert to a type.

";

%feature("docstring") casadi::ParsedFile::parse "

>  void ParsedFile.parse(str fname)
------------------------------------------------------------------------

Parse a file.

>  void ParsedFile.parse([str ] lines, int offset)
------------------------------------------------------------------------

Parse a list of strings.

";

%feature("docstring") casadi::ParsedFile::has "

Does an entry exist?

";

%feature("docstring") casadi::ParsedFile::to_int "

Get entry as an integer.

";

%feature("docstring") casadi::ParsedFile::to_vector "

Get entry as a vector.

";

%feature("docstring") casadi::ParsedFile::to_text "

Get entry as a text.

";

%feature("docstring") casadi::ParsedFile::to_string "

Get entry as a string.

";

%feature("docstring") casadi::ParsedFile::to_set "

Get entry as a set.

";

%feature("docstring") casadi::ParsedFile "

A parsed file.

Joel Andersson

C++ includes: casadi_file.hpp ";


// File: classcasadi_1_1Polynomial.xml
%feature("docstring") casadi::Polynomial "

Helper class for differentiating and integrating polynomials.

Joel Andersson

C++ includes: polynomial.hpp ";

%feature("docstring") casadi::Polynomial::derivative "

Create a new polynomial for the derivative.

";

%feature("docstring") casadi::Polynomial::Polynomial "

>  Polynomial(real_t scalar=1)
------------------------------------------------------------------------

Construct a constant polynomial.

>  Polynomial(real_t p0, real_t p1)
------------------------------------------------------------------------

Construct a linear polynomial.

>  Polynomial(real_t p0, real_t p1, real_t p2)
------------------------------------------------------------------------

Construct a quadratic polynomial.

>  Polynomial(real_t p0, real_t p1, real_t p2, real_t p3)
------------------------------------------------------------------------

Construct a cubic polynomial.

>  Polynomial([T ] coeff)
------------------------------------------------------------------------

Construct from a vector of polynomial coefficients.

";

%feature("docstring") casadi::Polynomial::print "

Print a description of the object.

";

%feature("docstring") casadi::Polynomial::anti_derivative "

Create a new polynomial for the anti-derivative (primitive function)

";

%feature("docstring") casadi::Polynomial::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::Polynomial::scalar "

Get scalar value (error if degree()!=0)

";

%feature("docstring") casadi::Polynomial::degree "

Degree of the polynomial.

";

%feature("docstring") casadi::Polynomial::trim "

Remove excess zeros.

";

%feature("docstring") casadi::Polynomial::repr "

Print a representation of the object.

";

%feature("docstring") casadi::Polynomial::getRepresentation "

Return a string with a representation (for SWIG)

";


// File: classcasadi_1_1PrintableObject.xml
%feature("docstring") casadi::PrintableObject::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") friendwrap_repr "

Return a string with a representation of the object, cf. repr(Object) in
Python.

";

%feature("docstring") friendwrap_str "

Return a string with a description of the object, cf. str(Object) in Python.

";

%feature("docstring") casadi::PrintableObject "

Base class for objects that have a natural string representation.

Joel Andersson

C++ includes: printable_object.hpp ";

%feature("docstring") casadi::PrintableObject::getDescription "

Return a string with a description (for SWIG)

";


// File: classcasadi_1_1Project.xml


// File: classcasadi_1_1PureMap.xml


// File: classcasadi_1_1Qpsol.xml


// File: classcasadi_1_1QpToNlp.xml


// File: classcasadi_1_1Rank1.xml


// File: classcasadi_1_1RealtypeSX.xml


// File: classcasadi_1_1Reshape.xml


// File: classcasadi_1_1RkIntegrator.xml


// File: classcasadi_1_1Rootfinder.xml


// File: classcasadi_1_1Scpgen.xml


// File: classcasadi_1_1SetNonzeros.xml


// File: classcasadi_1_1SetNonzerosSlice.xml


// File: classcasadi_1_1SetNonzerosSlice2.xml


// File: classcasadi_1_1SetNonzerosVector.xml


// File: classcasadi_1_1SharedObject.xml
%feature("docstring") casadi::SharedObject::print "

Print a description of the object.

";

%feature("docstring") casadi::SharedObject::is_null "

Is a null pointer?

";

%feature("docstring") casadi::SharedObject::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::SharedObject::printPtr "[INTERNAL]  Print the
pointer to the internal class

";

%feature("docstring") casadi::SharedObject "

SharedObject implements a reference counting framework similar for efficient
and easily-maintained memory management.

To use the class, both the SharedObject class (the public class), and the
SharedObjectNode class (the internal class) must be inherited from. It can
be done in two different files and together with memory management, this
approach provides a clear distinction of which methods of the class are to
be considered \"public\", i.e. methods for public use that can be considered
to remain over time with small changes, and the internal memory.

When interfacing a software, which typically includes including some header
file, this is best done only in the file where the internal class is
defined, to avoid polluting the global namespace and other side effects.

The default constructor always means creating a null pointer to an internal
class only. To allocate an internal class (this works only when the internal
class isn't abstract), use the constructor with arguments.

The copy constructor and the assignment operator perform shallow copies
only, to make a deep copy you must use the clone method explicitly. This
will give a shared pointer instance.

In an inheritance hierarchy, you can cast down automatically, e.g. (
SXFunction is a child class of Function): SXFunction derived(...); Function
base = derived;

To cast up, use the shared_cast template function, which works analogously
to dynamic_cast, static_cast, const_cast etc, e.g.: SXFunction derived(...);
Function base = derived; SXFunction derived_from_base =
shared_cast<SXFunction>(base);

A failed shared_cast will result in a null pointer (cf. dynamic_cast)

Joel Andersson

C++ includes: shared_object.hpp ";

%feature("docstring") casadi::SharedObject::__hash__ "

Returns a number that is unique for a given Node. If the Object does not
point to any node, \"0\" is returned.

";

%feature("docstring") casadi::SharedObject::repr "

Print a representation of the object.

";

%feature("docstring") casadi::SharedObject::getDescription "

Return a string with a description (for SWIG)

";


// File: classcasadi_1_1ShellCompiler.xml


// File: classcasadi_1_1SimplifiedExternal.xml


// File: classcasadi_1_1Slice.xml
%feature("docstring") casadi::Slice::is_scalar "

Is the slice a scalar.

";

%feature("docstring") casadi::Slice::scalar "

Get scalar (if is_scalar)

";

%feature("docstring") casadi::Slice::print "

Print a description of the object.

";

%feature("docstring") casadi::Slice "

Class representing a Slice.

Note that Python or Octave do not need to use this class. They can just use
slicing utility from the host language ( M[0:6] in Python, M(1:7) )

C++ includes: slice.hpp ";

%feature("docstring") casadi::Slice::all "

>  [int] Slice.all(int len, bool ind1=false) const 
------------------------------------------------------------------------

Get a vector of indices.

>  [int] Slice.all(Slice outer, int len) const 
------------------------------------------------------------------------

Get a vector of indices (nested slice)

";

%feature("docstring") casadi::Slice::Slice "

>  Slice()
------------------------------------------------------------------------

Default constructor - all elements.

>  Slice(int i, bool ind1=false)
------------------------------------------------------------------------

A single element (explicit to avoid ambiguity with IM overload.

>  Slice(int start, int stop, int step=1)
------------------------------------------------------------------------

A slice.

";

%feature("docstring") casadi::Slice::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::Slice::repr "

Print a representation of the object.

";

%feature("docstring") casadi::Slice::getRepresentation "

Return a string with a representation (for SWIG)

";


// File: classcasadi_1_1Solve.xml


// File: classcasadi_1_1SparseStorage.xml
%feature("docstring") casadi::SparseStorage::has_nz "[INTERNAL]  Returns
true if the matrix has a non-zero at location rr, cc.

";

%feature("docstring") casadi::SparseStorage::elem "[INTERNAL]  get a
reference to an element

";

%feature("docstring") casadi::SparseStorage::clear "[INTERNAL] ";

%feature("docstring") casadi::SparseStorage::reserve "[INTERNAL] ";

%feature("docstring") casadi::SparseStorage::nonzeros "

>  [DataType ]  SparseStorage< DataType >.nonzeros()
------------------------------------------------------------------------
[INTERNAL] 
Access the non-zero elements.

>  [DataType ]  SparseStorage< DataType >.nonzeros() const 
------------------------------------------------------------------------
[INTERNAL] 
Const access the non-zero elements.

";

%feature("docstring") casadi::SparseStorage::resize "[INTERNAL] ";

%feature("docstring") casadi::SparseStorage::sparsityRef "[INTERNAL]
Access the sparsity, make a copy if there are multiple references to it.

";

%feature("docstring") casadi::SparseStorage "[INTERNAL] C++ includes:
sparse_storage.hpp ";

%feature("docstring") casadi::SparseStorage::sparsity "[INTERNAL]  Const
access the sparsity - reference to data member.

";

%feature("docstring") casadi::SparseStorage::SparseStorage "

>  SparseStorage(Sparsity sparsity, DataType val=DataType(0))
------------------------------------------------------------------------
[INTERNAL] 
Sparse matrix with a given sparsity

>  SparseStorage()
------------------------------------------------------------------------
[INTERNAL] 
constructors

empty 0-by-0 matrix constructor

>  SparseStorage(const SparseStorage< DataType > &m)
------------------------------------------------------------------------
[INTERNAL] 
Copy constructor.

";

%feature("docstring") casadi::SparseStorage::scalar "[INTERNAL]  Convert to
scalar type.

";


// File: classcasadi_1_1Sparsity.xml


/*  Check if two sparsity patterns are identical  */

/*  Size and element counting  */ %feature("docstring")
casadi::Sparsity::enlargeRows "

Enlarge the matrix along the first dimension (i.e. insert rows)

";

%feature("docstring") casadi::Sparsity "

General sparsity class.

The storage format is a compressed column storage (CCS) format.  In this
format, the structural non-zero elements are stored in column-major order,
starting from the upper left corner of the matrix and ending in the lower
right corner.

In addition to the dimension ( size1(), size2()), (i.e. the number of rows
and the number of columns respectively), there are also two vectors of
integers:

\"colind\" [length size2()+1], which contains the index to the first non-
zero element on or after the corresponding column. All the non-zero elements
of a particular i are thus the elements with index el that fulfills:
colind[i] <= el < colind[i+1].

\"row\" [same length as the number of non-zero elements, nnz()] The rows for
each of the structural non-zeros.

Note that with this format, it is cheap to loop over all the non-zero
elements of a particular column, at constant time per element, but expensive
to jump to access a location (i, j).

If the matrix is dense, i.e. length(row) == size1()*size2(), the format
reduces to standard dense column major format, which allows access to an
arbitrary element in constant time.

Since the object is reference counted (it inherits from SharedObject),
several matrices are allowed to share the same sparsity pattern.

The implementations of some methods of this class has been taken from the
CSparse package and modified to use C++ standard library and CasADi data
structures.

See:   Matrix

Joel Andersson

C++ includes: sparsity.hpp ";

%feature("docstring") casadi::Sparsity::largest_first "

Order the columns by decreasing degree.

";

%feature("docstring") casadi::Sparsity::dim "

Get the dimension as a string.

";

%feature("docstring") casadi::Sparsity::add_nz "

Get the index of a non-zero element Add the element if it does not exist and
copy object if it's not unique.

";

%feature("docstring") casadi::Sparsity::is_scalar "

Is scalar?

";

%feature("docstring") casadi::Sparsity::rowsSequential "

Do the rows appear sequentially on each column.

Parameters:
-----------

strictly:  if true, then do not allow multiple entries

";

%feature("docstring") casadi::Sparsity::get_diag "

Get the diagonal of the matrix/create a diagonal matrix (mapping will
contain the nonzero mapping) When the input is square, the diagonal elements
are returned. If the input is vector-like, a diagonal matrix is constructed
with it.

";

%feature("docstring") casadi::Sparsity::btf "

Calculate the block triangular form (BTF) See Direct Methods for Sparse
Linear Systems by Davis (2006).

The function computes the Dulmage-Mendelsohn decomposition, which allows you
to reorder the rows and columns of a matrix to bring it into block
triangular form (BTF).

It will not consider the distance of off-diagonal elements to the diagonal:
there is no guarantee you will get a block-diagonal matrix if you supply a
randomly permuted block-diagonal matrix.

If your matrix is symmetrical, this method is of limited use; permutation
can make it non-symmetric.

See:   scc

";

%feature("docstring") casadi::Sparsity::star_coloring2 "

Perform a star coloring of a symmetric matrix: A new greedy distance-2
coloring algorithm Algorithm 4.1 in NEW ACYCLIC AND STAR COLORING ALGORITHMS
WITH APPLICATION TO COMPUTING HESSIANS A. H. GEBREMEDHIN, A. TARAFDAR, F.
MANNE, A. POTHEN SIAM J. SCI. COMPUT. Vol. 29, No. 3, pp. 10421072 (2007)

Ordering options: None (0), largest first (1)

";

%feature("docstring") casadi::Sparsity::enlargeColumns "

Enlarge the matrix along the second dimension (i.e. insert columns)

";

%feature("docstring") casadi::Sparsity::is_vector "

Check if the pattern is a row or column vector.

";

%feature("docstring") casadi::Sparsity::hash "";

%feature("docstring") casadi::Sparsity::resize "

Resize.

";

%feature("docstring") casadi::Sparsity::find "

Get the location of all non-zero elements as they would appear in a Dense
matrix A : DenseMatrix 4 x 3 B : SparseMatrix 4 x 3 , 5 structural non-
zeros.

k = A.find() A[k] will contain the elements of A that are non-zero in B

";

%feature("docstring") casadi::Sparsity::spy_matlab "

Generate a script for Matlab or Octave which visualizes the sparsity using
the spy command.

";

%feature("docstring") casadi::Sparsity::repr "

Print a representation of the object.

";

%feature("docstring") casadi::Sparsity::bw_lower "

Lower half-bandwidth.

";

%feature("docstring") casadi::Sparsity::T "

Transpose the matrix.

";

%feature("docstring") casadi::Sparsity::nnz_diag "

Number of non-zeros on the diagonal, i.e. the number of elements (i, j) with
j==i.

";

%feature("docstring") casadi::Sparsity::Sparsity "

>  Sparsity(int dummy=0)
------------------------------------------------------------------------

Default constructor.

>  Sparsity(int nrow, int ncol)
------------------------------------------------------------------------

Pattern with all structural zeros.

>  Sparsity(int nrow, int ncol, [int ] colind, [int ] row)
------------------------------------------------------------------------

Construct from sparsity pattern vectors given in compressed column storage
format.

";

%feature("docstring") casadi::Sparsity::colind "

Get a reference to the colindex of column cc (see class description)

";

%feature("docstring") casadi::Sparsity::is_row "

Check if the pattern is a row vector (i.e. size1()==1)

";

%feature("docstring") casadi::Sparsity::numel "

The total number of elements, including structural zeros, i.e.
size2()*size1()

See:   nnz()

";

%feature("docstring") casadi::Sparsity::unite "

Union of two sparsity patterns.

";

%feature("docstring") casadi::Sparsity::get_ccs "

Get the sparsity in compressed column storage (CCS) format.

";

%feature("docstring") casadi::Sparsity::is_transpose "

Check if the sparsity is the transpose of another.

";

%feature("docstring") casadi::Sparsity::get_triplet "

Get the sparsity in sparse triplet format.

";

%feature("docstring") casadi::Sparsity::is_square "

Is square?

";

%feature("docstring") casadi::Sparsity::appendColumns "

Append another sparsity patten horizontally.

";

%feature("docstring") casadi::Sparsity::isReshape "

Check if the sparsity is a reshape of another.

";

%feature("docstring") casadi::Sparsity::removeDuplicates "

Remove duplicate entries.

The same indices will be removed from the mapping vector, which must have
the same length as the number of nonzeros

";

%feature("docstring") casadi::Sparsity::is_empty "

Check if the sparsity is empty.

A sparsity is considered empty if one of the dimensions is zero (or
optionally both dimensions)

";

%feature("docstring") casadi::Sparsity::makeDense "

Make a patten dense.

";

%feature("docstring") casadi::Sparsity::pattern_inverse "

Take the inverse of a sparsity pattern; flip zeros and non-zeros.

";

%feature("docstring") casadi::Sparsity::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::Sparsity::dense "

Create a dense rectangular sparsity pattern.

";

%feature("docstring") casadi::Sparsity::diag "

Create diagonal sparsity pattern.

";

%feature("docstring") casadi::Sparsity::get_crs "

Get the sparsity in compressed row storage (CRS) format.

";

%feature("docstring") casadi::Sparsity::sub "

>  Sparsity Sparsity.sub([int ] rr, [int ] cc,[int ] output_mapping, bool ind1=false) const 
------------------------------------------------------------------------

Get a submatrix.

Returns the sparsity of the submatrix, with a mapping such that submatrix[k]
= originalmatrix[mapping[k]]

>  Sparsity Sparsity.sub([int ] rr, Sparsity sp,[int ] output_mapping, bool ind1=false) const 
------------------------------------------------------------------------

Get a set of elements.

Returns the sparsity of the corresponding elements, with a mapping such that
submatrix[k] = originalmatrix[mapping[k]]

";

%feature("docstring") casadi::Sparsity::unit "

Create the sparsity pattern for a unit vector of length n and a nonzero on
position el.

";

%feature("docstring") casadi::Sparsity::is_null "

Is a null pointer?

";

%feature("docstring") casadi::Sparsity::append "

Append another sparsity patten vertically (NOTE: only efficient if vector)

";

%feature("docstring") casadi::Sparsity::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::Sparsity::combine "

Combine two sparsity patterns Returns the new sparsity pattern as well as a
mapping with the same length as the number of non-zero elements The mapping
matrix contains the arguments for each nonzero, the first bit indicates if
the first argument is nonzero, the second bit indicates if the second
argument is nonzero (note that none of, one of or both of the arguments can
be nonzero)

";

%feature("docstring") casadi::Sparsity::nnz "

Get the number of (structural) non-zeros.

See:   numel()

";

%feature("docstring") casadi::Sparsity::intersect "

Intersection of two sparsity patterns Returns the new sparsity pattern as
well as a mapping with the same length as the number of non-zero elements
The value is 1 if the non-zero comes from the first (i.e. this) object, 2 if
it is from the second and 3 (i.e. 1 | 2) if from both.

";

%feature("docstring") casadi::Sparsity::row "

Get the row of a non-zero element.

";

%feature("docstring") casadi::Sparsity::print "

Print a description of the object.

";

%feature("docstring") casadi::Sparsity::is_column "

Check if the pattern is a column vector (i.e. size2()==1)

";

%feature("docstring") casadi::Sparsity::get_colind "

Get the column index for each column Together with the row-vector, one
obtains the sparsity pattern in the column compressed format.

";

%feature("docstring") casadi::Sparsity::get_row "

Get the row for each non-zero entry Together with the column-vector, this
vector gives the sparsity of the matrix in sparse triplet format, and
together with the colind vector, one obtains the sparsity in column
compressed format.

";

%feature("docstring") casadi::Sparsity::is_symmetric "

Is symmetric?

";

%feature("docstring") casadi::Sparsity::has_nz "

Returns true if the pattern has a non-zero at location rr, cc.

";

%feature("docstring") casadi::Sparsity::transpose "

Transpose the matrix and get the reordering of the non-zero entries.

Parameters:
-----------

mapping:  the non-zeros of the original matrix for each non-zero of the new
matrix

";

%feature("docstring") casadi::Sparsity::pmult "

Permute rows and/or columns Multiply the sparsity with a permutation matrix
from the left and/or from the right P * A * trans(P), A * trans(P) or A *
trans(P) with P defined by an index vector containing the row for each col.
As an alternative, P can be transposed (inverted).

";

%feature("docstring") casadi::Sparsity::is_diag "

Is diagonal?

";

%feature("docstring") casadi::Sparsity::is_equal "";

%feature("docstring") casadi::Sparsity::nnz_lower "

Number of non-zeros in the lower triangular half, i.e. the number of
elements (i, j) with j<=i.

";

%feature("docstring") casadi::Sparsity::get_lower "

Get nonzeros in lower triangular part.

";

%feature("docstring") casadi::Sparsity::is_singular "

Check whether the sparsity-pattern indicates structural singularity.

";

%feature("docstring") casadi::Sparsity::print_compact "

Print a compact description of the sparsity pattern.

";

%feature("docstring") casadi::Sparsity::is_dense "

Is dense?

";

%feature("docstring") casadi::Sparsity::sanity_check "

Check if the dimensions and colind, row vectors are compatible.

Parameters:
-----------

complete:  set to true to also check elementwise throws an error as possible
result

";

%feature("docstring") casadi::Sparsity::compressed "

Create from a single vector containing the pattern in compressed column
storage format: The format: The first two entries are the number of rows
(nrow) and columns (ncol) The next ncol+1 entries are the column offsets
(colind). Note that the last element, colind[ncol], gives the number of
nonzeros The last colind[ncol] entries are the row indices

";

%feature("docstring") casadi::Sparsity::uni_coloring "

Perform a unidirectional coloring: A greedy distance-2 coloring algorithm
(Algorithm 3.1 in A. H. GEBREMEDHIN, F. MANNE, A. POTHEN)

";

%feature("docstring") casadi::Sparsity::star_coloring "

Perform a star coloring of a symmetric matrix: A greedy distance-2 coloring
algorithm Algorithm 4.1 in What Color Is Your Jacobian? Graph Coloring for
Computing Derivatives A. H. GEBREMEDHIN, F. MANNE, A. POTHEN SIAM Rev.,
47(4), 629705 (2006)

Ordering options: None (0), largest first (1)

";

%feature("docstring") casadi::Sparsity::enlarge "

Enlarge matrix Make the matrix larger by inserting empty rows and columns,
keeping the existing non-zeros.

For the matrices A to B A(m, n) length(jj)=m , length(ii)=n B(nrow, ncol)

A=enlarge(m, n, ii, jj) makes sure that

B[jj, ii] == A

";

%feature("docstring") casadi::Sparsity::compress "

Compress a sparsity pattern.

";

%feature("docstring") casadi::Sparsity::scc "

Find the strongly connected components of the bigraph defined by the
sparsity pattern of a square matrix.

See Direct Methods for Sparse Linear Systems by Davis (2006). Returns:
Number of components

Offset for each components (length: 1 + number of components)

Indices for each components, component i has indices index[offset[i]], ...,
index[offset[i+1]]

In the case that the matrix is symmetric, the result has a particular
interpretation: Given a symmetric matrix A and n = A.scc(p, r)

=> A[p, p] will appear block-diagonal with n blocks and with the indices of
the block boundaries to be found in r.

";

%feature("docstring") casadi::Sparsity::get_upper "

Get nonzeros in upper triangular part.

";

%feature("docstring") casadi::Sparsity::printPtr "[INTERNAL]  Print the
pointer to the internal class

";

%feature("docstring") casadi::Sparsity::dfs "

Depth-first search on the adjacency graph of the sparsity See Direct Methods
for Sparse Linear Systems by Davis (2006).

";

%feature("docstring") casadi::Sparsity::is_tril "

Is lower triangular?

";

%feature("docstring") casadi::Sparsity::size2 "

Get the number of columns.

";

%feature("docstring") casadi::Sparsity::get_col "

Get the column for each non-zero entry Together with the row-vector, this
vector gives the sparsity of the matrix in sparse triplet format, i.e. the
column and row for each non-zero elements.

";

%feature("docstring") casadi::Sparsity::size1 "

Get the number of rows.

";

%feature("docstring") casadi::Sparsity::is_triu "

Is upper triangular?

";

%feature("docstring") casadi::Sparsity::spy "

Print a textual representation of sparsity.

";

%feature("docstring") casadi::Sparsity::__hash__ "

Returns a number that is unique for a given Node. If the Object does not
point to any node, \"0\" is returned.

";

%feature("docstring") casadi::Sparsity::size "

>  (int,int) Sparsity.size() const 
------------------------------------------------------------------------

Get the shape.

>  int Sparsity.size(int axis) const 
------------------------------------------------------------------------

Get the size along a particular dimensions.

";

%feature("docstring") casadi::Sparsity::etree "

Calculate the elimination tree See Direct Methods for Sparse Linear Systems
by Davis (2006). If the parameter ata is false, the algorithm is equivalent
to MATLAB's etree(A), except that the indices are zero- based. If ata is
true, the algorithm is equivalent to MATLAB's etree(A, 'col').

";

%feature("docstring") casadi::Sparsity::get_nz "

>  int Sparsity.get_nz(int rr, int cc) const 
------------------------------------------------------------------------

Get the index of an existing non-zero element return -1 if the element does
not exist.

>  [int] Sparsity.get_nz([int ] rr, [int ] cc) const 
------------------------------------------------------------------------

Get a set of non-zero element return -1 if the element does not exist.

>  void Sparsity.get_nz([int ] INOUT) const 
------------------------------------------------------------------------

Get the nonzero index for a set of elements The index vector is used both
for input and outputs and must be sorted by increasing nonzero index, i.e.
column-wise. Elements not found in the sparsity pattern are set to -1.

";

%feature("docstring") casadi::Sparsity::erase "

>  [int] Sparsity.erase([int ] rr, [int ] cc, bool ind1=false)
------------------------------------------------------------------------

Erase rows and/or columns of a matrix.

>  [int] Sparsity.erase([int ] rr, bool ind1=false)
------------------------------------------------------------------------

Erase elements of a matrix.

";

%feature("docstring") casadi::Sparsity::bw_upper "

Upper half-bandwidth.

";

%feature("docstring") casadi::Sparsity::scalar "

Create a scalar sparsity pattern.

";

%feature("docstring") casadi::Sparsity::nnz_upper "

Number of non-zeros in the upper triangular half, i.e. the number of
elements (i, j) with j>=i.

";


// File: classcasadi_1_1SparsityInterface.xml
%feature("docstring") friendwrap_diagsplit "

>  [MatType ] diagsplit(MatType x, [int ] output_offset1, [int ] output_offset2)
------------------------------------------------------------------------

split diagonally, retaining square matrices

Parameters:
-----------

output_offset1:  List of all start locations (row) for each group the last
matrix will run to the end.

output_offset2:  List of all start locations (row) for each group the last
matrix will run to the end.

diagcat(diagsplit(x, ...)) = x

>  [MatType ] diagsplit(MatType x, [int ] output_offset)
------------------------------------------------------------------------

split diagonally, retaining square matrices

Parameters:
-----------

output_offset:  List of all start locations for each group the last matrix
will run to the end.

diagcat(diagsplit(x, ...)) = x

>  [MatType ] diagsplit(MatType x, int incr=1)
------------------------------------------------------------------------

split diagonally, retaining groups of square matrices

Parameters:
-----------

incr:  Size of each matrix

diagsplit(diagsplit(x, ...)) = x

>  [MatType ] diagsplit(MatType x, int incr1, int incr2)
------------------------------------------------------------------------

split diagonally, retaining fixed-sized matrices

Parameters:
-----------

incr1:  Row dimension of each matrix

incr2:  Column dimension of each matrix

diagsplit(diagsplit(x, ...)) = x

";

%feature("docstring") friendwrap_triu "

Get the upper triangular part of a matrix.

";

%feature("docstring") friendwrap_mac "

Multiply-accumulate operation Matrix product of two matrices (x and y),
adding the result to a third matrix z. The result has the same sparsity
pattern as C meaning that other entries of (x*y) are ignored. The operation
is equivalent to: z+mtimes(x,y).project(z.sparsity()).

";

%feature("docstring") friendwrap_mtimes "

>  MatType mtimes(MatType x, MatType y)
------------------------------------------------------------------------

Matrix product of two matrices.

>  MatType mtimes([MatType ] args)
------------------------------------------------------------------------

Matrix product of n matrices.

";

%feature("docstring") friendwrap_transpose "

Transpose.

";

%feature("docstring") friendwrap_tril "

Get the lower triangular part of a matrix.

";

%feature("docstring") friendwrap_offset "

Helper function, get offsets corresponding to a vector of matrices.

";

%feature("docstring") friendwrap_vec "

make a vector Reshapes/vectorizes the matrix such that the shape becomes
(expr.numel(), 1). Columns are stacked on top of each other. Same as
reshape(expr, expr.numel(), 1)

a c b d  turns into

a b c d

";

%feature("docstring") friendwrap_horzcat "

>  MatType horzcat([MatType ] v)
------------------------------------------------------------------------

Concatenate a list of matrices horizontally Alternative terminology:
horizontal stack, hstack, horizontal append, [a b].

horzcat(horzsplit(x, ...)) = x

>  MatType horzcat(MatType x, MatType y)
------------------------------------------------------------------------

Concatenate horizontally, two matrices.

>  MatType horzcat(MatType x, MatType y, MatType z)
------------------------------------------------------------------------

Concatenate horizontally, three matrices.

>  MatType horzcat(MatType x, MatType y, MatType z, MatType w)
------------------------------------------------------------------------

Concatenate horizontally, four matrices.

";

%feature("docstring") casadi::SparsityInterface "

Sparsity interface class.

This is a common base class for GenericMatrix (i.e. MX and Matrix<>) and
Sparsity, introducing a uniform syntax and implementing common functionality
using the curiously recurring template pattern (CRTP) idiom. Joel Andersson

C++ includes: sparsity_interface.hpp ";

%feature("docstring") friendwrap_horzsplit "

>  [MatType ] horzsplit(MatType x, [int ] offset)
------------------------------------------------------------------------

split horizontally, retaining groups of columns

Parameters:
-----------

offset:  List of all start columns for each group the last column group will
run to the end.

horzcat(horzsplit(x, ...)) = x

>  [MatType ] horzsplit(MatType x, int incr=1)
------------------------------------------------------------------------

split horizontally, retaining fixed-sized groups of columns

Parameters:
-----------

incr:  Size of each group of columns

horzcat(horzsplit(x, ...)) = x

";

%feature("docstring") friendwrap_veccat "

concatenate vertically while vectorizing all arguments with vec

";

%feature("docstring") friendwrap_blocksplit "

>  [[MatType ] ] blocksplit(MatType x, [int ] vert_offset, [int ] horz_offset)
------------------------------------------------------------------------

chop up into blocks

Parameters:
-----------

vert_offset:  Defines the boundaries of the block rows

horz_offset:  Defines the boundaries of the block columns

blockcat(blocksplit(x,..., ...)) = x

>  [[MatType ] ] blocksplit(MatType x, int vert_incr=1, int horz_incr=1)
------------------------------------------------------------------------

chop up into blocks

Parameters:
-----------

vert_incr:  Defines the increment for block boundaries in row dimension

horz_incr:  Defines the increment for block boundaries in column dimension

blockcat(blocksplit(x,..., ...)) = x

";

%feature("docstring") friendwrap_repmat "

Repeat matrix A n times vertically and m times horizontally.

";

%feature("docstring") friendwrap_vertcat "

>  MatType vertcat([MatType ] v)
------------------------------------------------------------------------

Concatenate a list of matrices vertically Alternative terminology: vertical
stack, vstack, vertical append, [a;b].

vertcat(vertsplit(x, ...)) = x

>  MatType vertcat(MatType x, MatType y)
------------------------------------------------------------------------

Concatenate vertically, two matrices.

>  MatType vertcat(MatType x, MatType y, MatType z)
------------------------------------------------------------------------

Concatenate vertically, three matrices.

>  MatType vertcat(MatType x, MatType y, MatType z, MatType w)
------------------------------------------------------------------------

Concatenate vertically, four matrices.

";

%feature("docstring") friendwrap_sprank "

Obtain the structural rank of a sparsity-pattern.

";

%feature("docstring") friendwrap_kron "

Kronecker tensor product.

Creates a block matrix in which each element (i, j) is a_ij*b

";

%feature("docstring") friendwrap_reshape "

>  MatType reshape(MatType x, int nrow, int ncol)
------------------------------------------------------------------------

Returns a reshaped version of the matrix.

>  MatType reshape(MatType x,(int,int) rc)
------------------------------------------------------------------------

Returns a reshaped version of the matrix, dimensions as a vector.

>  MatType reshape(MatType x, Sparsity sp)
------------------------------------------------------------------------

Reshape the matrix.

";

%feature("docstring") friendwrap_norm_0_mul "

0-norm (nonzero count) of a Matrix-matrix product

";

%feature("docstring") friendwrap_diagcat "

>  MatType diagcat([MatType ] A)
------------------------------------------------------------------------

Construct a matrix with given block on the diagonal.

>  MatType diagcat(MatType x, MatType y)
------------------------------------------------------------------------

Concatenate along diagonal, two matrices.

>  MatType diagcat(MatType x, MatType y, MatType z)
------------------------------------------------------------------------

Concatenate along diagonal, three matrices.

>  MatType diagcat(MatType x, MatType y, MatType z, MatType w)
------------------------------------------------------------------------

Concatenate along diagonal, four matrices.

";

%feature("docstring") friendwrap_vertsplit "

>  [MatType ] vertsplit(MatType x, [int ] offset)
------------------------------------------------------------------------

split vertically, retaining groups of rows

*

Parameters:
-----------

output_offset:  List of all start rows for each group the last row group
will run to the end.

vertcat(vertsplit(x, ...)) = x

>  [MatType ] vertsplit(MatType x, int incr=1)
------------------------------------------------------------------------

split vertically, retaining fixed-sized groups of rows

Parameters:
-----------

incr:  Size of each group of rows

vertcat(vertsplit(x, ...)) = x



::

  >>> print vertsplit(SX.sym(\"a\",4))
  [SX(a_0), SX(a_1), SX(a_2), SX(a_3)]
  





::

  >>> print vertsplit(SX.sym(\"a\",4),2)
  [SX([a_0, a_1]), SX([a_2, a_3])]
  



If the number of rows is not a multiple of incr, the last entry returned
will have a size smaller than incr.



::

  >>> print vertsplit(DM([0,1,2,3,4]),2)
  [DM([0, 1]), DM([2, 3]), DM(4)]
  



";

%feature("docstring") friendwrap_blockcat "

>  MatType blockcat([[MatType ] ] v)
------------------------------------------------------------------------

Construct a matrix from a list of list of blocks.

>  MatType blockcat(MatType A, MatType B, MatType C, MatType D)
------------------------------------------------------------------------

Construct a matrix from 4 blocks.

";


// File: classcasadi_1_1Split.xml


// File: classcasadi_1_1Sqpmethod.xml


// File: classcasadi_1_1Logger_1_1Stream.xml
%feature("docstring") casadi::Logger::Stream "C++ includes:
casadi_logger.hpp ";

%feature("docstring") casadi::Logger::Stream::Stream "";


// File: classcasadi_1_1Logger_1_1Streambuf.xml
%feature("docstring") casadi::Logger::Streambuf "C++ includes:
casadi_logger.hpp ";

%feature("docstring") casadi::Logger::Streambuf::Streambuf "";


// File: classcasadi_1_1SubAssign.xml


// File: classcasadi_1_1SubIndex.xml
%feature("docstring") casadi::SubIndex "

SubIndex class for Matrix Same as the above class but for single argument
return for operator() Joel Andersson

C++ includes: submatrix.hpp ";

%feature("docstring") casadi::SubIndex::SubIndex "

Constructor.

";


// File: classcasadi_1_1SubMatrix.xml
%feature("docstring") casadi::SubMatrix "

SubMatrix class for Matrix SubMatrix is the return type for operator() of
the Matrix class, it allows access to the value as well as changing the
parent object Joel Andersson

C++ includes: submatrix.hpp ";

%feature("docstring") casadi::SubMatrix::SubMatrix "

Constructor.

";


// File: classcasadi_1_1SubRef.xml


// File: classcasadi_1_1Switch.xml


// File: classcasadi_1_1SXFunction.xml


// File: classcasadi_1_1SymbolicMX.xml


// File: classcasadi_1_1SymbolicQr.xml


// File: classcasadi_1_1SymbolicSX.xml


// File: classcasadi_1_1Transpose.xml


// File: classcasadi_1_1UnaryMX.xml


// File: classcasadi_1_1UnarySX.xml


// File: classcasadi_1_1Vertcat.xml


// File: classcasadi_1_1Vertsplit.xml


// File: classcasadi_1_1WeakRef.xml
%feature("docstring") casadi::WeakRef "[INTERNAL]  Weak reference type A
weak reference to a SharedObject.

Joel Andersson

C++ includes: weak_ref.hpp ";

%feature("docstring") casadi::WeakRef::shared "[INTERNAL]  Get a shared
(owning) reference.

";

%feature("docstring") casadi::WeakRef::__hash__ "[INTERNAL]  Returns a
number that is unique for a given Node. If the Object does not point to any
node, \"0\" is returned.

";

%feature("docstring") casadi::WeakRef::print "[INTERNAL]  Print a
description of the object.

";

%feature("docstring") casadi::WeakRef::getDescription "[INTERNAL]  Return a
string with a description (for SWIG)

";

%feature("docstring") casadi::WeakRef::WeakRef "

>  WeakRef(int dummy=0)
------------------------------------------------------------------------
[INTERNAL] 
Default constructor.

>  WeakRef(SharedObject shared)
------------------------------------------------------------------------
[INTERNAL] 
Construct from a shared object (also implicit type conversion)

";

%feature("docstring") casadi::WeakRef::printPtr "[INTERNAL]  Print the
pointer to the internal class

";

%feature("docstring") casadi::WeakRef::is_null "[INTERNAL]  Is a null
pointer?

";

%feature("docstring") casadi::WeakRef::getRepresentation "[INTERNAL]
Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::WeakRef::repr "[INTERNAL]  Print a
representation of the object.

";

%feature("docstring") casadi::WeakRef::alive "[INTERNAL]  Check if alive.

";


// File: classcasadi_1_1XFunction.xml


// File: classcasadi_1_1XmlFile.xml
%feature("docstring") casadi::XmlFile "

XML parser Can be used for parsing XML files into CasADi data structures.

Joel Andersson

C++ includes: xml_file.hpp ";

%feature("docstring") casadi::XmlFile::getDescription "

Return a string with a description (for SWIG)

";

%feature("docstring") casadi::XmlFile::print "

Print a description of the object.

";

%feature("docstring") casadi::XmlFile::printPtr "[INTERNAL]  Print the
pointer to the internal class

";

%feature("docstring") casadi::XmlFile::getRepresentation "

Return a string with a representation (for SWIG)

";

%feature("docstring") casadi::XmlFile::__hash__ "

Returns a number that is unique for a given Node. If the Object does not
point to any node, \"0\" is returned.

";

%feature("docstring") casadi::XmlFile::is_null "

Is a null pointer?

";

%feature("docstring") casadi::XmlFile::XmlFile "";

%feature("docstring") casadi::XmlFile::~XmlFile "";

%feature("docstring") casadi::XmlFile::repr "

Print a representation of the object.

";


// File: classcasadi_1_1ZeroByZero.xml


// File: classcasadi_1_1ZeroSX.xml


// File: namespacecasadi.xml
%feature("docstring") casadi::matrixName< double > "
Get typename.

";

%feature("docstring") casadi::linsol_in "

Linear solver input scheme.

";

%feature("docstring") casadi::inBounds "

>  bool inBounds([T ] v, int upper)
------------------------------------------------------------------------

Check if for each element of v holds: v_i < upper.

>  bool inBounds([T ] v, int lower, int upper)
------------------------------------------------------------------------

Check if for each element of v holds: lower <= v_i < upper.

";

%feature("docstring") casadi::casadi_rank1 "

>  void rank1(real_t *A, const int *sp_A, real_t alpha, const real_t *x)
------------------------------------------------------------------------
[INTERNAL] 
Adds a multiple alpha/2 of the outer product mul(x, trans(x)) to A.

>  void rank1(real_t *A, const int *sp_A, real_t alpha, const real_t *x, const real_t *y)
------------------------------------------------------------------------
[INTERNAL] 
";

%feature("docstring") casadi::complement "

Returns the list of all i in [0, size[ not found in supplied list.

The supplied vector may contain duplicates and may be non-monotonous The
supplied vector will be checked for bounds The result vector is guaranteed
to be monotonously increasing

";

%feature("docstring") casadi::swapIndices "

swap inner and outer indices of list of lists



::

  * [[apple0,apple1,...],[pear0,pear1,...]] ->
  *   [[apple0,pear0],[apple1,pear1],...]
  * 



";

%feature("docstring") casadi::isNon_increasing "

Check if the vector is non-increasing.

";

%feature("docstring") casadi::is_zero "";

%feature("docstring") casadi::doc_linsol "

Get the documentation string for a plugin.

";

%feature("docstring") casadi::integrator_n_out "

Get the number of integrator outputs.

";

%feature("docstring") casadi::external "

>  Function external(str name, Dict opts=Dict())
------------------------------------------------------------------------

Load an external function File name is assumed to be ./<f_name>.so.

>  Function external(str name, str bin_name, Dict opts=Dict())
------------------------------------------------------------------------

Load an external function File name given.

>  Function external(str name, Compiler compiler, Dict opts=Dict())
------------------------------------------------------------------------

Load a just-in-time compiled external function File name given.

";

%feature("docstring") casadi::nlpsol_n_out "

Number of NLP solver outputs.

";

%feature("docstring") casadi::isDecreasing "

Check if the vector is strictly decreasing.

";

%feature("docstring") casadi::load_integrator "

Explicitly load a plugin dynamically.

";

%feature("docstring") casadi::load_nlpsol "

Explicitly load a plugin dynamically.

";

%feature("docstring") casadi::casadi_scal "[INTERNAL]  SCAL: x <- alpha*x.

";

%feature("docstring") casadi::load_linsol "

Explicitly load a plugin dynamically.

";

%feature("docstring") casadi::qpsol_in "

>  CASADI_EXPORT[str] qpsol_in()
------------------------------------------------------------------------

Get input scheme of QP solvers.

>  str qpsol_in(int ind)
------------------------------------------------------------------------

Get QP solver input scheme name by index.

";

%feature("docstring") casadi::hash_combine "

>  void hash_combine(std.size_t &seed, T v)

>  void hash_combine(std.size_t &seed, [int ] v)
------------------------------------------------------------------------

Generate a hash value incrementally (function taken from boost)

>  void hash_combine(std.size_t &seed, const int *v, int sz)
------------------------------------------------------------------------

Generate a hash value incrementally, array.

";

%feature("docstring") casadi::casadi_swap "[INTERNAL]  SWAP: x <-> y.

";

%feature("docstring") casadi::integrator "

>  Function integrator(str name, str solver, const str:SX &dae, Dict opts=Dict())
------------------------------------------------------------------------

Create an ODE/DAE integrator Solves an initial value problem (IVP) coupled
to a terminal value problem with differential equation given as an implicit
ODE coupled to an algebraic equation and a set of quadratures:



::

  Initial conditions at t=t0
  x(t0)  = x0
  q(t0)  = 0
  
  Forward integration from t=t0 to t=tf
  der(x) = function(x, z, p, t)                  Forward ODE
  0 = fz(x, z, p, t)                  Forward algebraic equations
  der(q) = fq(x, z, p, t)                  Forward quadratures
  
  Terminal conditions at t=tf
  rx(tf)  = rx0
  rq(tf)  = 0
  
  Backward integration from t=tf to t=t0
  der(rx) = gx(rx, rz, rp, x, z, p, t)        Backward ODE
  0 = gz(rx, rz, rp, x, z, p, t)        Backward algebraic equations
  der(rq) = gq(rx, rz, rp, x, z, p, t)        Backward quadratures
  
  where we assume that both the forward and backwards integrations are index-1
  (i.e. dfz/dz, dgz/drz are invertible) and furthermore that
  gx, gz and gq have a linear dependency on rx, rz and rp.



General information
===================



>List of available options

+-----------------+-----------------+-----------------+-----------------+
|       Id        |      Type       |   Description   |     Used in     |
+=================+=================+=================+=================+
| ad_weight       | OT_DOUBLE       | Weighting       | casadi::Functio |
|                 |                 | factor for      | nInternal       |
|                 |                 | derivative calc |                 |
|                 |                 | ulation.When    |                 |
|                 |                 | there is an     |                 |
|                 |                 | option of       |                 |
|                 |                 | either using    |                 |
|                 |                 | forward or      |                 |
|                 |                 | reverse mode    |                 |
|                 |                 | directional     |                 |
|                 |                 | derivatives,    |                 |
|                 |                 | the condition a |                 |
|                 |                 | d_weight*nf<=(1 |                 |
|                 |                 | -ad_weight)*na  |                 |
|                 |                 | is used where   |                 |
|                 |                 | nf and na are   |                 |
|                 |                 | estimates of    |                 |
|                 |                 | the number of   |                 |
|                 |                 | forward/reverse |                 |
|                 |                 | mode            |                 |
|                 |                 | directional     |                 |
|                 |                 | derivatives     |                 |
|                 |                 | needed. By      |                 |
|                 |                 | default,        |                 |
|                 |                 | ad_weight is    |                 |
|                 |                 | calculated      |                 |
|                 |                 | automatically,  |                 |
|                 |                 | but this can be |                 |
|                 |                 | overridden by   |                 |
|                 |                 | setting this    |                 |
|                 |                 | option. In      |                 |
|                 |                 | particular, 0   |                 |
|                 |                 | means forcing   |                 |
|                 |                 | forward mode    |                 |
|                 |                 | and 1 forcing   |                 |
|                 |                 | reverse mode.   |                 |
|                 |                 | Leave unset for |                 |
|                 |                 | (class          |                 |
|                 |                 | specific)       |                 |
|                 |                 | heuristics.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| ad_weight_sp    | OT_DOUBLE       | Weighting       | casadi::Functio |
|                 |                 | factor for      | nInternal       |
|                 |                 | sparsity        |                 |
|                 |                 | pattern         |                 |
|                 |                 | calculation cal |                 |
|                 |                 | culation.Overri |                 |
|                 |                 | des default     |                 |
|                 |                 | behavior. Set   |                 |
|                 |                 | to 0 and 1 to   |                 |
|                 |                 | force forward   |                 |
|                 |                 | and reverse     |                 |
|                 |                 | mode            |                 |
|                 |                 | respectively.   |                 |
|                 |                 | Cf. option      |                 |
|                 |                 | \"ad_weight\".    |                 |
+-----------------+-----------------+-----------------+-----------------+
| augmented_optio | OT_DICT         | Options to be   | casadi::Integra |
| ns              |                 | passed down to  | tor             |
|                 |                 | the augmented   |                 |
|                 |                 | integrator, if  |                 |
|                 |                 | one is          |                 |
|                 |                 | constructed.    |                 |
+-----------------+-----------------+-----------------+-----------------+
| compiler        | OT_STRING       | Just-in-time    | casadi::Functio |
|                 |                 | compiler plugin | nInternal       |
|                 |                 | to be used.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| derivative_of   | OT_FUNCTION     | The function is | casadi::Functio |
|                 |                 | a derivative of | nInternal       |
|                 |                 | another         |                 |
|                 |                 | function. The   |                 |
|                 |                 | type of         |                 |
|                 |                 | derivative      |                 |
|                 |                 | (directional    |                 |
|                 |                 | derivative,     |                 |
|                 |                 | Jacobian) is    |                 |
|                 |                 | inferred from   |                 |
|                 |                 | the function    |                 |
|                 |                 | name.           |                 |
+-----------------+-----------------+-----------------+-----------------+
| expand          | OT_BOOL         | Replace MX with | casadi::Integra |
|                 |                 | SX expressions  | tor             |
|                 |                 | in problem      |                 |
|                 |                 | formulation     |                 |
|                 |                 | [false]         |                 |
+-----------------+-----------------+-----------------+-----------------+
| gather_stats    | OT_BOOL         | Flag to         | casadi::Functio |
|                 |                 | indicate        | nInternal       |
|                 |                 | whether         |                 |
|                 |                 | statistics must |                 |
|                 |                 | be gathered     |                 |
+-----------------+-----------------+-----------------+-----------------+
| grid            | OT_DOUBLEVECTOR | Time grid       | casadi::Integra |
|                 |                 |                 | tor             |
+-----------------+-----------------+-----------------+-----------------+
| input_scheme    | OT_STRINGVECTOR | Custom input    | casadi::Functio |
|                 |                 | scheme          | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| inputs_check    | OT_BOOL         | Throw           | casadi::Functio |
|                 |                 | exceptions when | nInternal       |
|                 |                 | the numerical   |                 |
|                 |                 | values of the   |                 |
|                 |                 | inputs don't    |                 |
|                 |                 | make sense      |                 |
+-----------------+-----------------+-----------------+-----------------+
| jac_penalty     | OT_DOUBLE       | When requested  | casadi::Functio |
|                 |                 | for a number of | nInternal       |
|                 |                 | forward/reverse |                 |
|                 |                 | directions, it  |                 |
|                 |                 | may be cheaper  |                 |
|                 |                 | to compute      |                 |
|                 |                 | first the full  |                 |
|                 |                 | jacobian and    |                 |
|                 |                 | then multiply   |                 |
|                 |                 | with seeds,     |                 |
|                 |                 | rather than     |                 |
|                 |                 | obtain the      |                 |
|                 |                 | requested       |                 |
|                 |                 | directions in a |                 |
|                 |                 | straightforward |                 |
|                 |                 | manner. Casadi  |                 |
|                 |                 | uses a          |                 |
|                 |                 | heuristic to    |                 |
|                 |                 | decide which is |                 |
|                 |                 | cheaper. A high |                 |
|                 |                 | value of        |                 |
|                 |                 | 'jac_penalty'   |                 |
|                 |                 | makes it less   |                 |
|                 |                 | likely for the  |                 |
|                 |                 | heurstic to     |                 |
|                 |                 | chose the full  |                 |
|                 |                 | Jacobian        |                 |
|                 |                 | strategy. The   |                 |
|                 |                 | special value   |                 |
|                 |                 | -1 indicates    |                 |
|                 |                 | never to use    |                 |
|                 |                 | the full        |                 |
|                 |                 | Jacobian        |                 |
|                 |                 | strategy        |                 |
+-----------------+-----------------+-----------------+-----------------+
| jit             | OT_BOOL         | Use just-in-    | casadi::Functio |
|                 |                 | time compiler   | nInternal       |
|                 |                 | to speed up the |                 |
|                 |                 | evaluation      |                 |
+-----------------+-----------------+-----------------+-----------------+
| jit_options     | OT_DICT         | Options to be   | casadi::Functio |
|                 |                 | passed to the   | nInternal       |
|                 |                 | jit compiler.   |                 |
+-----------------+-----------------+-----------------+-----------------+
| monitor         | OT_STRINGVECTOR | Monitors to be  | casadi::Functio |
|                 |                 | activated       | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| number_of_finit | OT_INT          | Number of       | casadi::Integra |
| e_elements      |                 | finite elements | tor             |
+-----------------+-----------------+-----------------+-----------------+
| output_scheme   | OT_STRINGVECTOR | Custom output   | casadi::Functio |
|                 |                 | scheme          | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| output_t0       | OT_BOOL         | Output the      | casadi::Integra |
|                 |                 | state at the    | tor             |
|                 |                 | initial time    |                 |
+-----------------+-----------------+-----------------+-----------------+
| print_stats     | OT_BOOL         | Print out       | casadi::Integra |
|                 |                 | statistics      | tor             |
|                 |                 | after           |                 |
|                 |                 | integration     |                 |
+-----------------+-----------------+-----------------+-----------------+
| regularity_chec | OT_BOOL         | Throw           | casadi::Functio |
| k               |                 | exceptions when | nInternal       |
|                 |                 | NaN or Inf      |                 |
|                 |                 | appears during  |                 |
|                 |                 | evaluation      |                 |
+-----------------+-----------------+-----------------+-----------------+
| rootfinder      | OT_STRING       | An implicit     | casadi::Integra |
|                 |                 | function solver | tor             |
+-----------------+-----------------+-----------------+-----------------+
| rootfinder_opti | OT_DICT         | Options to be   | casadi::Integra |
| ons             |                 | passed to the   | tor             |
|                 |                 | NLP Solver      |                 |
+-----------------+-----------------+-----------------+-----------------+
| t0              | OT_DOUBLE       | Beginning of    | casadi::Integra |
|                 |                 | the time        | tor             |
|                 |                 | horizon         |                 |
+-----------------+-----------------+-----------------+-----------------+
| tf              | OT_DOUBLE       | End of the time | casadi::Integra |
|                 |                 | horizon         | tor             |
+-----------------+-----------------+-----------------+-----------------+
| user_data       | OT_VOIDPTR      | A user-defined  | casadi::Functio |
|                 |                 | field that can  | nInternal       |
|                 |                 | be used to      |                 |
|                 |                 | identify the    |                 |
|                 |                 | function or     |                 |
|                 |                 | pass additional |                 |
|                 |                 | information     |                 |
+-----------------+-----------------+-----------------+-----------------+
| verbose         | OT_BOOL         | Verbose         | casadi::Functio |
|                 |                 | evaluation  for | nInternal       |
|                 |                 | debugging       |                 |
+-----------------+-----------------+-----------------+-----------------+

>Input scheme: casadi::IntegratorInput (INTEGRATOR_NUM_IN = 6) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| INTEGRATOR_X0          |                        | Differential state at  |
|                        |                        | the initial time.      |
+------------------------+------------------------+------------------------+
| INTEGRATOR_P           |                        | Parameters.            |
+------------------------+------------------------+------------------------+
| INTEGRATOR_Z0          |                        | Initial guess for the  |
|                        |                        | algebraic variable.    |
+------------------------+------------------------+------------------------+
| INTEGRATOR_RX0         |                        | Backward differential  |
|                        |                        | state at the final     |
|                        |                        | time.                  |
+------------------------+------------------------+------------------------+
| INTEGRATOR_RP          |                        | Backward parameter     |
|                        |                        | vector.                |
+------------------------+------------------------+------------------------+
| INTEGRATOR_RZ0         |                        | Initial guess for the  |
|                        |                        | backwards algebraic    |
|                        |                        | variable.              |
+------------------------+------------------------+------------------------+

>Output scheme: casadi::IntegratorOutput (INTEGRATOR_NUM_OUT = 6) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| INTEGRATOR_XF          |                        | Differential state at  |
|                        |                        | the final time.        |
+------------------------+------------------------+------------------------+
| INTEGRATOR_QF          |                        | Quadrature state at    |
|                        |                        | the final time.        |
+------------------------+------------------------+------------------------+
| INTEGRATOR_ZF          |                        | Algebraic variable at  |
|                        |                        | the final time.        |
+------------------------+------------------------+------------------------+
| INTEGRATOR_RXF         |                        | Backward differential  |
|                        |                        | state at the initial   |
|                        |                        | time.                  |
+------------------------+------------------------+------------------------+
| INTEGRATOR_RQF         |                        | Backward quadrature    |
|                        |                        | state at the initial   |
|                        |                        | time.                  |
+------------------------+------------------------+------------------------+
| INTEGRATOR_RZF         |                        | Backward algebraic     |
|                        |                        | variable at the        |
|                        |                        | initial time.          |
+------------------------+------------------------+------------------------+

List of plugins
===============



- cvodes

- idas

- collocation

- rk

Note: some of the plugins in this list might not be available on your
system. Also, there might be extra plugins available to you that are not
listed here. You can obtain their documentation with
Integrator.doc(\"myextraplugin\")



--------------------------------------------------------------------------------

cvodes
------



Interface to CVodes from the Sundials suite.

A call to evaluate will integrate to the end.

You can retrieve the entire state trajectory as follows, after the evaluate
call: Call reset. Then call integrate(t_i) and getOuput for a series of
times t_i.

Note: depending on the dimension and structure of your problem, you may
experience a dramatic speed-up by using a sparse linear solver:



::

     intg.setOption(\"linear_solver\",\"csparse\")
     intg.setOption(\"linear_solver_type\",\"user_defined\")



>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| abstol                 | OT_DOUBLE              | Absolute tolerence for |
|                        |                        | the IVP solution       |
+------------------------+------------------------+------------------------+
| abstolB                | OT_DOUBLE              | Absolute tolerence for |
|                        |                        | the adjoint            |
|                        |                        | sensitivity solution   |
|                        |                        | [default: equal to     |
|                        |                        | abstol]                |
+------------------------+------------------------+------------------------+
| disable_internal_warni | OT_BOOL                | Disable SUNDIALS       |
| ngs                    |                        | internal warning       |
|                        |                        | messages               |
+------------------------+------------------------+------------------------+
| exact_jacobian         | OT_BOOL                | Use exact Jacobian     |
|                        |                        | information for the    |
|                        |                        | forward integration    |
+------------------------+------------------------+------------------------+
| exact_jacobianB        | OT_BOOL                | Use exact Jacobian     |
|                        |                        | information for the    |
|                        |                        | backward integration   |
|                        |                        | [default: equal to     |
|                        |                        | exact_jacobian]        |
+------------------------+------------------------+------------------------+
| finite_difference_fsen | OT_BOOL                | Use finite differences |
| s                      |                        | to approximate the     |
|                        |                        | forward sensitivity    |
|                        |                        | equations (if AD is    |
|                        |                        | not available)         |
+------------------------+------------------------+------------------------+
| fsens_abstol           | OT_DOUBLE              | Absolute tolerence for |
|                        |                        | the forward            |
|                        |                        | sensitivity solution   |
|                        |                        | [default: equal to     |
|                        |                        | abstol]                |
+------------------------+------------------------+------------------------+
| fsens_all_at_once      | OT_BOOL                | Calculate all right    |
|                        |                        | hand sides of the      |
|                        |                        | sensitivity equations  |
|                        |                        | at once                |
+------------------------+------------------------+------------------------+
| fsens_err_con          | OT_BOOL                | include the forward    |
|                        |                        | sensitivities in all   |
|                        |                        | error controls         |
+------------------------+------------------------+------------------------+
| fsens_reltol           | OT_DOUBLE              | Relative tolerence for |
|                        |                        | the forward            |
|                        |                        | sensitivity solution   |
|                        |                        | [default: equal to     |
|                        |                        | reltol]                |
+------------------------+------------------------+------------------------+
| fsens_scaling_factors  | OT_DOUBLEVECTOR        | Scaling factor for the |
|                        |                        | components if finite   |
|                        |                        | differences is used    |
+------------------------+------------------------+------------------------+
| fsens_sensitiviy_param | OT_INTVECTOR           | Specifies which        |
| eters                  |                        | components will be     |
|                        |                        | used when estimating   |
|                        |                        | the sensitivity        |
|                        |                        | equations              |
+------------------------+------------------------+------------------------+
| interpolation_type     | OT_STRING              | Type of interpolation  |
|                        |                        | for the adjoint        |
|                        |                        | sensitivities          |
+------------------------+------------------------+------------------------+
| iterative_solver       | OT_STRING              | Iterative solver:      |
|                        |                        | GMRES|bcgstab|tfqmr    |
+------------------------+------------------------+------------------------+
| iterative_solverB      | OT_STRING              | Iterative solver for   |
|                        |                        | backward integration   |
+------------------------+------------------------+------------------------+
| linear_multistep_metho | OT_STRING              | Integrator scheme:     |
| d                      |                        | BDF|adams              |
+------------------------+------------------------+------------------------+
| linear_solver          | OT_STRING              | A custom linear solver |
|                        |                        | creator function       |
+------------------------+------------------------+------------------------+
| linear_solverB         | OT_STRING              | A custom linear solver |
|                        |                        | creator function for   |
|                        |                        | backwards integration  |
|                        |                        | [default: equal to     |
|                        |                        | linear_solver]         |
+------------------------+------------------------+------------------------+
| linear_solver_options  | OT_DICT                | Options to be passed   |
|                        |                        | to the linear solver   |
+------------------------+------------------------+------------------------+
| linear_solver_optionsB | OT_DICT                | Options to be passed   |
|                        |                        | to the linear solver   |
|                        |                        | for backwards          |
|                        |                        | integration [default:  |
|                        |                        | equal to               |
|                        |                        | linear_solver_options] |
+------------------------+------------------------+------------------------+
| linear_solver_type     | OT_STRING              | Type of iterative      |
|                        |                        | solver: user_defined|D |
|                        |                        | ENSE|banded|iterative  |
+------------------------+------------------------+------------------------+
| linear_solver_typeB    | OT_STRING              | Linear solver for      |
|                        |                        | backward integration   |
+------------------------+------------------------+------------------------+
| lower_bandwidth        | OT_INT                 | Lower band-width of    |
|                        |                        | banded Jacobian        |
|                        |                        | (estimations)          |
+------------------------+------------------------+------------------------+
| lower_bandwidthB       | OT_INT                 | lower band-width of    |
|                        |                        | banded jacobians for   |
|                        |                        | backward integration   |
|                        |                        | [default: equal to     |
|                        |                        | lower_bandwidth]       |
+------------------------+------------------------+------------------------+
| max_krylov             | OT_INT                 | Maximum Krylov         |
|                        |                        | subspace size          |
+------------------------+------------------------+------------------------+
| max_krylovB            | OT_INT                 | Maximum krylov         |
|                        |                        | subspace size for      |
|                        |                        | backward integration   |
+------------------------+------------------------+------------------------+
| max_multistep_order    | OT_INT                 | Maximum order for the  |
|                        |                        | (variable-order)       |
|                        |                        | multistep method       |
+------------------------+------------------------+------------------------+
| max_num_steps          | OT_INT                 | Maximum number of      |
|                        |                        | integrator steps       |
+------------------------+------------------------+------------------------+
| nonlinear_solver_itera | OT_STRING              | Nonlinear solver type: |
| tion                   |                        | NEWTON|functional      |
+------------------------+------------------------+------------------------+
| pretype                | OT_STRING              | Type of                |
|                        |                        | preconditioning:       |
|                        |                        | NONE|left|right|both   |
+------------------------+------------------------+------------------------+
| pretypeB               | OT_STRING              | Preconditioner for     |
|                        |                        | backward integration   |
+------------------------+------------------------+------------------------+
| quad_err_con           | OT_BOOL                | Should the quadratures |
|                        |                        | affect the step size   |
|                        |                        | control                |
+------------------------+------------------------+------------------------+
| reltol                 | OT_DOUBLE              | Relative tolerence for |
|                        |                        | the IVP solution       |
+------------------------+------------------------+------------------------+
| reltolB                | OT_DOUBLE              | Relative tolerence for |
|                        |                        | the adjoint            |
|                        |                        | sensitivity solution   |
|                        |                        | [default: equal to     |
|                        |                        | reltol]                |
+------------------------+------------------------+------------------------+
| sensitivity_method     | OT_STRING              | Sensitivity method:    |
|                        |                        | SIMULTANEOUS|staggered |
+------------------------+------------------------+------------------------+
| steps_per_checkpoint   | OT_INT                 | Number of steps        |
|                        |                        | between two            |
|                        |                        | consecutive            |
|                        |                        | checkpoints            |
+------------------------+------------------------+------------------------+
| stop_at_end            | OT_BOOL                | Stop the integrator at |
|                        |                        | the end of the         |
|                        |                        | interval               |
+------------------------+------------------------+------------------------+
| upper_bandwidth        | OT_INT                 | Upper band-width of    |
|                        |                        | banded Jacobian        |
|                        |                        | (estimations)          |
+------------------------+------------------------+------------------------+
| upper_bandwidthB       | OT_INT                 | Upper band-width of    |
|                        |                        | banded jacobians for   |
|                        |                        | backward integration   |
|                        |                        | [default: equal to     |
|                        |                        | upper_bandwidth]       |
+------------------------+------------------------+------------------------+
| use_preconditioner     | OT_BOOL                | Precondition an        |
|                        |                        | iterative solver       |
+------------------------+------------------------+------------------------+
| use_preconditionerB    | OT_BOOL                | Precondition an        |
|                        |                        | iterative solver for   |
|                        |                        | the backwards problem  |
|                        |                        | [default: equal to     |
|                        |                        | use_preconditioner]    |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

idas
----



Interface to IDAS from the Sundials suite.

Note: depending on the dimension and structure of your problem, you may
experience a dramatic speed-up by using a sparse linear solver:



::

     intg.setOption(\"linear_solver\",\"csparse\")
     intg.setOption(\"linear_solver_type\",\"user_defined\")



>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| abstol                 | OT_DOUBLE              | Absolute tolerence for |
|                        |                        | the IVP solution       |
+------------------------+------------------------+------------------------+
| abstolB                | OT_DOUBLE              | Absolute tolerence for |
|                        |                        | the adjoint            |
|                        |                        | sensitivity solution   |
|                        |                        | [default: equal to     |
|                        |                        | abstol]                |
+------------------------+------------------------+------------------------+
| abstolv                | OT_DOUBLEVECTOR        | Absolute tolerarance   |
|                        |                        | for each component     |
+------------------------+------------------------+------------------------+
| calc_ic                | OT_BOOL                | Use IDACalcIC to get   |
|                        |                        | consistent initial     |
|                        |                        | conditions.            |
+------------------------+------------------------+------------------------+
| calc_icB               | OT_BOOL                | Use IDACalcIC to get   |
|                        |                        | consistent initial     |
|                        |                        | conditions for         |
|                        |                        | backwards system       |
|                        |                        | [default: equal to     |
|                        |                        | calc_ic].              |
+------------------------+------------------------+------------------------+
| cj_scaling             | OT_BOOL                | IDAS scaling on cj for |
|                        |                        | the user-defined       |
|                        |                        | linear solver module   |
+------------------------+------------------------+------------------------+
| disable_internal_warni | OT_BOOL                | Disable SUNDIALS       |
| ngs                    |                        | internal warning       |
|                        |                        | messages               |
+------------------------+------------------------+------------------------+
| exact_jacobian         | OT_BOOL                | Use exact Jacobian     |
|                        |                        | information for the    |
|                        |                        | forward integration    |
+------------------------+------------------------+------------------------+
| exact_jacobianB        | OT_BOOL                | Use exact Jacobian     |
|                        |                        | information for the    |
|                        |                        | backward integration   |
|                        |                        | [default: equal to     |
|                        |                        | exact_jacobian]        |
+------------------------+------------------------+------------------------+
| extra_fsens_calc_ic    | OT_BOOL                | Call calc ic an extra  |
|                        |                        | time, with fsens=0     |
+------------------------+------------------------+------------------------+
| finite_difference_fsen | OT_BOOL                | Use finite differences |
| s                      |                        | to approximate the     |
|                        |                        | forward sensitivity    |
|                        |                        | equations (if AD is    |
|                        |                        | not available)         |
+------------------------+------------------------+------------------------+
| first_time             | OT_DOUBLE              | First requested time   |
|                        |                        | as a fraction of the   |
|                        |                        | time interval          |
+------------------------+------------------------+------------------------+
| fsens_abstol           | OT_DOUBLE              | Absolute tolerence for |
|                        |                        | the forward            |
|                        |                        | sensitivity solution   |
|                        |                        | [default: equal to     |
|                        |                        | abstol]                |
+------------------------+------------------------+------------------------+
| fsens_abstolv          | OT_DOUBLEVECTOR        | Absolute tolerarance   |
|                        |                        | for each component,    |
|                        |                        | forward sensitivities  |
+------------------------+------------------------+------------------------+
| fsens_err_con          | OT_BOOL                | include the forward    |
|                        |                        | sensitivities in all   |
|                        |                        | error controls         |
+------------------------+------------------------+------------------------+
| fsens_reltol           | OT_DOUBLE              | Relative tolerence for |
|                        |                        | the forward            |
|                        |                        | sensitivity solution   |
|                        |                        | [default: equal to     |
|                        |                        | reltol]                |
+------------------------+------------------------+------------------------+
| fsens_scaling_factors  | OT_DOUBLEVECTOR        | Scaling factor for the |
|                        |                        | components if finite   |
|                        |                        | differences is used    |
+------------------------+------------------------+------------------------+
| fsens_sensitiviy_param | OT_INTVECTOR           | Specifies which        |
| eters                  |                        | components will be     |
|                        |                        | used when estimating   |
|                        |                        | the sensitivity        |
|                        |                        | equations              |
+------------------------+------------------------+------------------------+
| init_xdot              | OT_DOUBLEVECTOR        | Initial values for the |
|                        |                        | state derivatives      |
+------------------------+------------------------+------------------------+
| interpolation_type     | OT_STRING              | Type of interpolation  |
|                        |                        | for the adjoint        |
|                        |                        | sensitivities          |
+------------------------+------------------------+------------------------+
| iterative_solver       | OT_STRING              | Iterative solver:      |
|                        |                        | GMRES|bcgstab|tfqmr    |
+------------------------+------------------------+------------------------+
| iterative_solverB      | OT_STRING              | Iterative solver for   |
|                        |                        | backward integration   |
+------------------------+------------------------+------------------------+
| linear_solver          | OT_STRING              | A custom linear solver |
|                        |                        | creator function       |
+------------------------+------------------------+------------------------+
| linear_solverB         | OT_STRING              | A custom linear solver |
|                        |                        | creator function for   |
|                        |                        | backwards integration  |
|                        |                        | [default: equal to     |
|                        |                        | linear_solver]         |
+------------------------+------------------------+------------------------+
| linear_solver_options  | OT_DICT                | Options to be passed   |
|                        |                        | to the linear solver   |
+------------------------+------------------------+------------------------+
| linear_solver_optionsB | OT_DICT                | Options to be passed   |
|                        |                        | to the linear solver   |
|                        |                        | for backwards          |
|                        |                        | integration [default:  |
|                        |                        | equal to               |
|                        |                        | linear_solver_options] |
+------------------------+------------------------+------------------------+
| linear_solver_type     | OT_STRING              | Type of iterative      |
|                        |                        | solver: user_defined|D |
|                        |                        | ENSE|banded|iterative  |
+------------------------+------------------------+------------------------+
| linear_solver_typeB    | OT_STRING              | Linear solver for      |
|                        |                        | backward integration   |
+------------------------+------------------------+------------------------+
| lower_bandwidth        | OT_INT                 | Lower band-width of    |
|                        |                        | banded Jacobian        |
|                        |                        | (estimations)          |
+------------------------+------------------------+------------------------+
| lower_bandwidthB       | OT_INT                 | lower band-width of    |
|                        |                        | banded jacobians for   |
|                        |                        | backward integration   |
|                        |                        | [default: equal to     |
|                        |                        | lower_bandwidth]       |
+------------------------+------------------------+------------------------+
| max_krylov             | OT_INT                 | Maximum Krylov         |
|                        |                        | subspace size          |
+------------------------+------------------------+------------------------+
| max_krylovB            | OT_INT                 | Maximum krylov         |
|                        |                        | subspace size for      |
|                        |                        | backward integration   |
+------------------------+------------------------+------------------------+
| max_multistep_order    | OT_INT                 | Maximum order for the  |
|                        |                        | (variable-order)       |
|                        |                        | multistep method       |
+------------------------+------------------------+------------------------+
| max_num_steps          | OT_INT                 | Maximum number of      |
|                        |                        | integrator steps       |
+------------------------+------------------------+------------------------+
| max_step_size          | OT_DOUBLE              | Maximim step size      |
+------------------------+------------------------+------------------------+
| pretype                | OT_STRING              | Type of                |
|                        |                        | preconditioning:       |
|                        |                        | NONE|left|right|both   |
+------------------------+------------------------+------------------------+
| pretypeB               | OT_STRING              | Preconditioner for     |
|                        |                        | backward integration   |
+------------------------+------------------------+------------------------+
| quad_err_con           | OT_BOOL                | Should the quadratures |
|                        |                        | affect the step size   |
|                        |                        | control                |
+------------------------+------------------------+------------------------+
| reltol                 | OT_DOUBLE              | Relative tolerence for |
|                        |                        | the IVP solution       |
+------------------------+------------------------+------------------------+
| reltolB                | OT_DOUBLE              | Relative tolerence for |
|                        |                        | the adjoint            |
|                        |                        | sensitivity solution   |
|                        |                        | [default: equal to     |
|                        |                        | reltol]                |
+------------------------+------------------------+------------------------+
| sensitivity_method     | OT_STRING              | Sensitivity method:    |
|                        |                        | SIMULTANEOUS|staggered |
+------------------------+------------------------+------------------------+
| steps_per_checkpoint   | OT_INT                 | Number of steps        |
|                        |                        | between two            |
|                        |                        | consecutive            |
|                        |                        | checkpoints            |
+------------------------+------------------------+------------------------+
| stop_at_end            | OT_BOOL                | Stop the integrator at |
|                        |                        | the end of the         |
|                        |                        | interval               |
+------------------------+------------------------+------------------------+
| suppress_algebraic     | OT_BOOL                | Suppress algebraic     |
|                        |                        | variables in the error |
|                        |                        | testing                |
+------------------------+------------------------+------------------------+
| upper_bandwidth        | OT_INT                 | Upper band-width of    |
|                        |                        | banded Jacobian        |
|                        |                        | (estimations)          |
+------------------------+------------------------+------------------------+
| upper_bandwidthB       | OT_INT                 | Upper band-width of    |
|                        |                        | banded jacobians for   |
|                        |                        | backward integration   |
|                        |                        | [default: equal to     |
|                        |                        | upper_bandwidth]       |
+------------------------+------------------------+------------------------+
| use_preconditioner     | OT_BOOL                | Precondition an        |
|                        |                        | iterative solver       |
+------------------------+------------------------+------------------------+
| use_preconditionerB    | OT_BOOL                | Precondition an        |
|                        |                        | iterative solver for   |
|                        |                        | the backwards problem  |
|                        |                        | [default: equal to     |
|                        |                        | use_preconditioner]    |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

collocation
-----------



Fixed-step implicit Runge-Kutta integrator ODE/DAE integrator based on
collocation schemes

The method is still under development

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| augmented_options      | OT_DICT                | Options to be passed   |
|                        |                        | down to the augmented  |
|                        |                        | integrator, if one is  |
|                        |                        | constructed.           |
+------------------------+------------------------+------------------------+
| collocation_scheme     | OT_STRING              | Collocation scheme:    |
|                        |                        | radau|legendre         |
+------------------------+------------------------+------------------------+
| expand                 | OT_BOOL                | Replace MX with SX     |
|                        |                        | expressions in problem |
|                        |                        | formulation [false]    |
+------------------------+------------------------+------------------------+
| grid                   | OT_DOUBLEVECTOR        | Time grid              |
+------------------------+------------------------+------------------------+
| interpolation_order    | OT_INT                 | Order of the           |
|                        |                        | interpolating          |
|                        |                        | polynomials            |
+------------------------+------------------------+------------------------+
| number_of_finite_eleme | OT_INT                 | Number of finite       |
| nts                    |                        | elements               |
+------------------------+------------------------+------------------------+
| output_t0              | OT_BOOL                | Output the state at    |
|                        |                        | the initial time       |
+------------------------+------------------------+------------------------+
| print_stats            | OT_BOOL                | Print out statistics   |
|                        |                        | after integration      |
+------------------------+------------------------+------------------------+
| rootfinder             | OT_STRING              | An implicit function   |
|                        |                        | solver                 |
+------------------------+------------------------+------------------------+
| rootfinder_options     | OT_DICT                | Options to be passed   |
|                        |                        | to the NLP Solver      |
+------------------------+------------------------+------------------------+
| t0                     | OT_DOUBLE              | Beginning of the time  |
|                        |                        | horizon                |
+------------------------+------------------------+------------------------+
| tf                     | OT_DOUBLE              | End of the time        |
|                        |                        | horizon                |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

rk --



Fixed-step explicit Runge-Kutta integrator for ODEs Currently implements
RK4.

The method is still under development

--------------------------------------------------------------------------------



Joel Andersson

";

%feature("docstring") casadi::casadi_iamax "[INTERNAL]  IAMAX: index
corresponding to the entry with the largest absolute value.

";

%feature("docstring") casadi::hasNegative "

Check if the vector has negative entries.

";

%feature("docstring") casadi::doc_qpsol "

Get the documentation string for a plugin.

";

%feature("docstring") casadi::check_exposed "[INTERNAL] ";

%feature("docstring") casadi::read_matlab "

>  void read_matlab(std.istream &stream,[T ] v)
------------------------------------------------------------------------

Read vector, matlab style.

>  void read_matlab(std.ifstream &file,[[T ] ] v)
------------------------------------------------------------------------

Read matrix, matlab style.

";

%feature("docstring") casadi::qpsol_n_out "

Get the number of QP solver outputs.

";

%feature("docstring") casadi::casadi_mtimes "[INTERNAL]  Sparse matrix-
matrix multiplication: z <- z + x*y.

";

%feature("docstring") casadi::qpsol_n_in "

Get the number of QP solver inputs.

";

%feature("docstring") casadi::write_matlab "

>  void write_matlab(std.ostream &stream, [T ] v)
------------------------------------------------------------------------

Print vector, matlab style.

>  void write_matlab(std.ostream &stream, [[T ] ] v)
------------------------------------------------------------------------

Print matrix, matlab style.

";

%feature("docstring") casadi::casadi_sparsify "[INTERNAL]  Convert dense to
sparse.

";

%feature("docstring") casadi::hash_sparsity "

>  std.size_t hash_sparsity(int nrow, int ncol, [int ] colind, [int ] row)
------------------------------------------------------------------------

Hash a sparsity pattern.

";

%feature("docstring") casadi::is_slice2 "

Check if an index vector can be represented more efficiently as two nested
slices.

";

%feature("docstring") casadi::rootfinder "

Create a solver for rootfinding problems Takes a function where one of the
inputs is unknown and one of the outputs is a residual function that is
always zero, defines a new function where the the unknown input has been
replaced by a guess for the unknown and the residual output has been
replaced by the calculated value for the input.

For a function [y0, y1, ...,yi, .., yn] = F(x0, x1, ..., xj, ..., xm), where
xj is unknown and yi=0, defines a new function [y0, y1, ...,xj, .., yn] =
G(x0, x1, ..., xj_guess, ..., xm),

xj and yi must have the same dimension and d(yi)/d(xj) must be invertable.

By default, the first input is unknown and the first output is the residual.

General information
===================



>List of available options

+-----------------+-----------------+-----------------+-----------------+
|       Id        |      Type       |   Description   |     Used in     |
+=================+=================+=================+=================+
| ad_weight       | OT_DOUBLE       | Weighting       | casadi::Functio |
|                 |                 | factor for      | nInternal       |
|                 |                 | derivative calc |                 |
|                 |                 | ulation.When    |                 |
|                 |                 | there is an     |                 |
|                 |                 | option of       |                 |
|                 |                 | either using    |                 |
|                 |                 | forward or      |                 |
|                 |                 | reverse mode    |                 |
|                 |                 | directional     |                 |
|                 |                 | derivatives,    |                 |
|                 |                 | the condition a |                 |
|                 |                 | d_weight*nf<=(1 |                 |
|                 |                 | -ad_weight)*na  |                 |
|                 |                 | is used where   |                 |
|                 |                 | nf and na are   |                 |
|                 |                 | estimates of    |                 |
|                 |                 | the number of   |                 |
|                 |                 | forward/reverse |                 |
|                 |                 | mode            |                 |
|                 |                 | directional     |                 |
|                 |                 | derivatives     |                 |
|                 |                 | needed. By      |                 |
|                 |                 | default,        |                 |
|                 |                 | ad_weight is    |                 |
|                 |                 | calculated      |                 |
|                 |                 | automatically,  |                 |
|                 |                 | but this can be |                 |
|                 |                 | overridden by   |                 |
|                 |                 | setting this    |                 |
|                 |                 | option. In      |                 |
|                 |                 | particular, 0   |                 |
|                 |                 | means forcing   |                 |
|                 |                 | forward mode    |                 |
|                 |                 | and 1 forcing   |                 |
|                 |                 | reverse mode.   |                 |
|                 |                 | Leave unset for |                 |
|                 |                 | (class          |                 |
|                 |                 | specific)       |                 |
|                 |                 | heuristics.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| ad_weight_sp    | OT_DOUBLE       | Weighting       | casadi::Functio |
|                 |                 | factor for      | nInternal       |
|                 |                 | sparsity        |                 |
|                 |                 | pattern         |                 |
|                 |                 | calculation cal |                 |
|                 |                 | culation.Overri |                 |
|                 |                 | des default     |                 |
|                 |                 | behavior. Set   |                 |
|                 |                 | to 0 and 1 to   |                 |
|                 |                 | force forward   |                 |
|                 |                 | and reverse     |                 |
|                 |                 | mode            |                 |
|                 |                 | respectively.   |                 |
|                 |                 | Cf. option      |                 |
|                 |                 | \"ad_weight\".    |                 |
+-----------------+-----------------+-----------------+-----------------+
| compiler        | OT_STRING       | Just-in-time    | casadi::Functio |
|                 |                 | compiler plugin | nInternal       |
|                 |                 | to be used.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| constraints     | OT_INTVECTOR    | Constrain the   | casadi::Rootfin |
|                 |                 | unknowns. 0     | der             |
|                 |                 | (default): no   |                 |
|                 |                 | constraint on   |                 |
|                 |                 | ui, 1: ui >=    |                 |
|                 |                 | 0.0, -1: ui <=  |                 |
|                 |                 | 0.0, 2: ui >    |                 |
|                 |                 | 0.0, -2: ui <   |                 |
|                 |                 | 0.0.            |                 |
+-----------------+-----------------+-----------------+-----------------+
| derivative_of   | OT_FUNCTION     | The function is | casadi::Functio |
|                 |                 | a derivative of | nInternal       |
|                 |                 | another         |                 |
|                 |                 | function. The   |                 |
|                 |                 | type of         |                 |
|                 |                 | derivative      |                 |
|                 |                 | (directional    |                 |
|                 |                 | derivative,     |                 |
|                 |                 | Jacobian) is    |                 |
|                 |                 | inferred from   |                 |
|                 |                 | the function    |                 |
|                 |                 | name.           |                 |
+-----------------+-----------------+-----------------+-----------------+
| gather_stats    | OT_BOOL         | Flag to         | casadi::Functio |
|                 |                 | indicate        | nInternal       |
|                 |                 | whether         |                 |
|                 |                 | statistics must |                 |
|                 |                 | be gathered     |                 |
+-----------------+-----------------+-----------------+-----------------+
| implicit_input  | OT_INT          | Index of the    | casadi::Rootfin |
|                 |                 | input that      | der             |
|                 |                 | corresponds to  |                 |
|                 |                 | the actual      |                 |
|                 |                 | root-finding    |                 |
+-----------------+-----------------+-----------------+-----------------+
| implicit_output | OT_INT          | Index of the    | casadi::Rootfin |
|                 |                 | output that     | der             |
|                 |                 | corresponds to  |                 |
|                 |                 | the actual      |                 |
|                 |                 | root-finding    |                 |
+-----------------+-----------------+-----------------+-----------------+
| input_scheme    | OT_STRINGVECTOR | Custom input    | casadi::Functio |
|                 |                 | scheme          | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| inputs_check    | OT_BOOL         | Throw           | casadi::Functio |
|                 |                 | exceptions when | nInternal       |
|                 |                 | the numerical   |                 |
|                 |                 | values of the   |                 |
|                 |                 | inputs don't    |                 |
|                 |                 | make sense      |                 |
+-----------------+-----------------+-----------------+-----------------+
| jac_penalty     | OT_DOUBLE       | When requested  | casadi::Functio |
|                 |                 | for a number of | nInternal       |
|                 |                 | forward/reverse |                 |
|                 |                 | directions, it  |                 |
|                 |                 | may be cheaper  |                 |
|                 |                 | to compute      |                 |
|                 |                 | first the full  |                 |
|                 |                 | jacobian and    |                 |
|                 |                 | then multiply   |                 |
|                 |                 | with seeds,     |                 |
|                 |                 | rather than     |                 |
|                 |                 | obtain the      |                 |
|                 |                 | requested       |                 |
|                 |                 | directions in a |                 |
|                 |                 | straightforward |                 |
|                 |                 | manner. Casadi  |                 |
|                 |                 | uses a          |                 |
|                 |                 | heuristic to    |                 |
|                 |                 | decide which is |                 |
|                 |                 | cheaper. A high |                 |
|                 |                 | value of        |                 |
|                 |                 | 'jac_penalty'   |                 |
|                 |                 | makes it less   |                 |
|                 |                 | likely for the  |                 |
|                 |                 | heurstic to     |                 |
|                 |                 | chose the full  |                 |
|                 |                 | Jacobian        |                 |
|                 |                 | strategy. The   |                 |
|                 |                 | special value   |                 |
|                 |                 | -1 indicates    |                 |
|                 |                 | never to use    |                 |
|                 |                 | the full        |                 |
|                 |                 | Jacobian        |                 |
|                 |                 | strategy        |                 |
+-----------------+-----------------+-----------------+-----------------+
| jacobian_functi | OT_FUNCTION     | Function object | casadi::Rootfin |
| on              |                 | for calculating | der             |
|                 |                 | the Jacobian    |                 |
|                 |                 | (autogenerated  |                 |
|                 |                 | by default)     |                 |
+-----------------+-----------------+-----------------+-----------------+
| jit             | OT_BOOL         | Use just-in-    | casadi::Functio |
|                 |                 | time compiler   | nInternal       |
|                 |                 | to speed up the |                 |
|                 |                 | evaluation      |                 |
+-----------------+-----------------+-----------------+-----------------+
| jit_options     | OT_DICT         | Options to be   | casadi::Functio |
|                 |                 | passed to the   | nInternal       |
|                 |                 | jit compiler.   |                 |
+-----------------+-----------------+-----------------+-----------------+
| linear_solver   | OT_STRING       | User-defined    | casadi::Rootfin |
|                 |                 | linear solver   | der             |
|                 |                 | class. Needed   |                 |
|                 |                 | for             |                 |
|                 |                 | sensitivities.  |                 |
+-----------------+-----------------+-----------------+-----------------+
| linear_solver_o | OT_DICT         | Options to be   | casadi::Rootfin |
| ptions          |                 | passed to the   | der             |
|                 |                 | linear solver.  |                 |
+-----------------+-----------------+-----------------+-----------------+
| monitor         | OT_STRINGVECTOR | Monitors to be  | casadi::Functio |
|                 |                 | activated       | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| output_scheme   | OT_STRINGVECTOR | Custom output   | casadi::Functio |
|                 |                 | scheme          | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| regularity_chec | OT_BOOL         | Throw           | casadi::Functio |
| k               |                 | exceptions when | nInternal       |
|                 |                 | NaN or Inf      |                 |
|                 |                 | appears during  |                 |
|                 |                 | evaluation      |                 |
+-----------------+-----------------+-----------------+-----------------+
| user_data       | OT_VOIDPTR      | A user-defined  | casadi::Functio |
|                 |                 | field that can  | nInternal       |
|                 |                 | be used to      |                 |
|                 |                 | identify the    |                 |
|                 |                 | function or     |                 |
|                 |                 | pass additional |                 |
|                 |                 | information     |                 |
+-----------------+-----------------+-----------------+-----------------+
| verbose         | OT_BOOL         | Verbose         | casadi::Functio |
|                 |                 | evaluation  for | nInternal       |
|                 |                 | debugging       |                 |
+-----------------+-----------------+-----------------+-----------------+

List of plugins
===============



- kinsol

- nlpsol

- newton

Note: some of the plugins in this list might not be available on your
system. Also, there might be extra plugins available to you that are not
listed here. You can obtain their documentation with
Rootfinder.doc(\"myextraplugin\")



--------------------------------------------------------------------------------

kinsol
------



KINSOL interface from the Sundials suite

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| abstol                 | OT_DOUBLE              | Stopping criterion     |
|                        |                        | tolerance              |
+------------------------+------------------------+------------------------+
| disable_internal_warni | OT_BOOL                | Disable KINSOL         |
| ngs                    |                        | internal warning       |
|                        |                        | messages               |
+------------------------+------------------------+------------------------+
| exact_jacobian         | OT_BOOL                | Use exact Jacobian     |
|                        |                        | information            |
+------------------------+------------------------+------------------------+
| f_scale                | OT_DOUBLEVECTOR        | Equation scaling       |
|                        |                        | factors                |
+------------------------+------------------------+------------------------+
| iterative_solver       | OT_STRING              | gmres|bcgstab|tfqmr    |
+------------------------+------------------------+------------------------+
| linear_solver_type     | OT_STRING              | dense|banded|iterative |
|                        |                        | |user_defined          |
+------------------------+------------------------+------------------------+
| lower_bandwidth        | OT_INT                 | Lower bandwidth for    |
|                        |                        | banded linear solvers  |
+------------------------+------------------------+------------------------+
| max_iter               | OT_INT                 | Maximum number of      |
|                        |                        | Newton iterations.     |
|                        |                        | Putting 0 sets the     |
|                        |                        | default value of       |
|                        |                        | KinSol.                |
+------------------------+------------------------+------------------------+
| max_krylov             | OT_INT                 | Maximum Krylov space   |
|                        |                        | dimension              |
+------------------------+------------------------+------------------------+
| pretype                | OT_STRING              | Type of preconditioner |
+------------------------+------------------------+------------------------+
| strategy               | OT_STRING              | Globalization strategy |
+------------------------+------------------------+------------------------+
| u_scale                | OT_DOUBLEVECTOR        | Variable scaling       |
|                        |                        | factors                |
+------------------------+------------------------+------------------------+
| upper_bandwidth        | OT_INT                 | Upper bandwidth for    |
|                        |                        | banded linear solvers  |
+------------------------+------------------------+------------------------+
| use_preconditioner     | OT_BOOL                | Precondition an        |
|                        |                        | iterative solver       |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

nlpsol
------





--------------------------------------------------------------------------------





--------------------------------------------------------------------------------

newton
------



Implements simple newton iterations to solve an implicit function.

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| abstol                 | OT_DOUBLE              | Stopping criterion     |
|                        |                        | tolerance on max(|F|)  |
+------------------------+------------------------+------------------------+
| abstolStep             | OT_DOUBLE              | Stopping criterion     |
|                        |                        | tolerance on step size |
+------------------------+------------------------+------------------------+
| max_iter               | OT_INT                 | Maximum number of      |
|                        |                        | Newton iterations to   |
|                        |                        | perform before         |
|                        |                        | returning.             |
+------------------------+------------------------+------------------------+
| print_iteration        | OT_BOOL                | Print information      |
|                        |                        | about each iteration   |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



Joel Andersson

";

%feature("docstring") casadi::isStrictlyMonotone "

Check if the vector is strictly monotone.

";

%feature("docstring") casadi::dlaqge_ "[INTERNAL]  Equilibrate the system.

";

%feature("docstring") casadi::to_slice "

>  Slice to_slice(IM x, bool ind1=false)
------------------------------------------------------------------------

Convert IM to Slice.

>  Slice to_slice([int ] v, bool ind1=false)
------------------------------------------------------------------------

Construct from an index vector (requires is_slice(v) to be true)

";

%feature("docstring") casadi::casadi_copy "[INTERNAL]  COPY: y <-x.

";

%feature("docstring") casadi::dgeequ_ "[INTERNAL]  Calculate col and row
scaling.

";

%feature("docstring") casadi::lookupvector "

Returns a vector for quickly looking up entries of supplied list.

lookupvector[i]!=-1 <=> v contains i v[lookupvector[i]] == i <=> v contains
i

Duplicates are treated by looking up last occurrence

";

%feature("docstring") casadi::dormqr_ "[INTERNAL]  Multiply right hand side
with Q-transpose (lapack)

";

%feature("docstring") casadi::has_integrator "

Check if a particular plugin is available.

";

%feature("docstring") casadi::isNonDecreasing "

Check if the vector is non-decreasing.

";

%feature("docstring") casadi::jit "

Create a just-in-time compiled function from a C/C++ language string The
function can an arbitrary number of inputs and outputs that must all be
scalar-valued. Only specify the function body, assuming that the inputs are
stored in an array named 'arg' and the outputs stored in an array named
'res'. The data type used must be 'real_t', which is typically equal to
'double` or another data type with the same API as 'double'.

The final generated function will have a structure similar to:

void fname(const real_t* arg, real_t* res) { <FUNCTION_BODY> }

";

%feature("docstring") casadi::load_rootfinder "

Explicitly load a plugin dynamically.

";

%feature("docstring") casadi::dgeqrf_ "[INTERNAL]  QR-factorize dense
matrix (lapack)

";

%feature("docstring") casadi::integrator_n_in "

Get the number of integrator inputs.

";

%feature("docstring") casadi::collocation_points "

Obtain collocation points of specific order and scheme.

Parameters:
-----------

scheme:  'radau' or 'legendre'

";

%feature("docstring") casadi::linsol_n_out "

Number of linear solver outputs.

";

%feature("docstring") casadi::simpleIRK "

Construct an implicit Runge-Kutta integrator using a collocation scheme The
constructed function has three inputs, corresponding to initial state (x0),
parameter (p) and integration time (h) and one output, corresponding to
final state (xf).

Parameters:
-----------

f:  ODE function with two inputs (x and p) and one output (xdot)

N:  Number of integrator steps

order:  Order of interpolating polynomials

scheme:  Collocation scheme, as excepted by collocationPoints function.

";

%feature("docstring") casadi::ptrVec "[INTERNAL]  Convenience function,
convert vectors to vectors of pointers.

";

%feature("docstring") casadi::to_slice2 "

Construct nested slices from an index vector (requires is_slice2(v) to be
true)

";

%feature("docstring") casadi::dtrsm_ "[INTERNAL]   Solve upper triangular
system (lapack)

";

%feature("docstring") casadi::casadi_axpy "[INTERNAL]  AXPY: y <- a*x + y.

";

%feature("docstring") casadi::has_nlpsol "

Check if a particular plugin is available.

";

%feature("docstring") casadi::casadi_getu "[INTERNAL]  Get the nonzeros for
the upper triangular half.

";

%feature("docstring") casadi::doc_nlpsol "

Get the documentation string for a plugin.

";

%feature("docstring") casadi::isMonotone "

Check if the vector is monotone.

";

%feature("docstring") casadi::qpsol "

>  Function qpsol(str name, str solver, SpDict qp, Dict opts=Dict())
------------------------------------------------------------------------

Create a QP solver Solves the following strictly convex problem:



::

  min          1/2 x' H x + g' x
  x
  
  subject to
  LBA <= A x <= UBA
  LBX <= x   <= UBX
  
  with :
  H sparse (n x n) positive definite
  g dense  (n x 1)
  
  n: number of decision variables (x)
  nc: number of constraints (A)



If H is not positive-definite, the solver should throw an error.

General information
===================



>List of available options

+-----------------+-----------------+-----------------+-----------------+
|       Id        |      Type       |   Description   |     Used in     |
+=================+=================+=================+=================+
| ad_weight       | OT_DOUBLE       | Weighting       | casadi::Functio |
|                 |                 | factor for      | nInternal       |
|                 |                 | derivative calc |                 |
|                 |                 | ulation.When    |                 |
|                 |                 | there is an     |                 |
|                 |                 | option of       |                 |
|                 |                 | either using    |                 |
|                 |                 | forward or      |                 |
|                 |                 | reverse mode    |                 |
|                 |                 | directional     |                 |
|                 |                 | derivatives,    |                 |
|                 |                 | the condition a |                 |
|                 |                 | d_weight*nf<=(1 |                 |
|                 |                 | -ad_weight)*na  |                 |
|                 |                 | is used where   |                 |
|                 |                 | nf and na are   |                 |
|                 |                 | estimates of    |                 |
|                 |                 | the number of   |                 |
|                 |                 | forward/reverse |                 |
|                 |                 | mode            |                 |
|                 |                 | directional     |                 |
|                 |                 | derivatives     |                 |
|                 |                 | needed. By      |                 |
|                 |                 | default,        |                 |
|                 |                 | ad_weight is    |                 |
|                 |                 | calculated      |                 |
|                 |                 | automatically,  |                 |
|                 |                 | but this can be |                 |
|                 |                 | overridden by   |                 |
|                 |                 | setting this    |                 |
|                 |                 | option. In      |                 |
|                 |                 | particular, 0   |                 |
|                 |                 | means forcing   |                 |
|                 |                 | forward mode    |                 |
|                 |                 | and 1 forcing   |                 |
|                 |                 | reverse mode.   |                 |
|                 |                 | Leave unset for |                 |
|                 |                 | (class          |                 |
|                 |                 | specific)       |                 |
|                 |                 | heuristics.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| ad_weight_sp    | OT_DOUBLE       | Weighting       | casadi::Functio |
|                 |                 | factor for      | nInternal       |
|                 |                 | sparsity        |                 |
|                 |                 | pattern         |                 |
|                 |                 | calculation cal |                 |
|                 |                 | culation.Overri |                 |
|                 |                 | des default     |                 |
|                 |                 | behavior. Set   |                 |
|                 |                 | to 0 and 1 to   |                 |
|                 |                 | force forward   |                 |
|                 |                 | and reverse     |                 |
|                 |                 | mode            |                 |
|                 |                 | respectively.   |                 |
|                 |                 | Cf. option      |                 |
|                 |                 | \"ad_weight\".    |                 |
+-----------------+-----------------+-----------------+-----------------+
| compiler        | OT_STRING       | Just-in-time    | casadi::Functio |
|                 |                 | compiler plugin | nInternal       |
|                 |                 | to be used.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| derivative_of   | OT_FUNCTION     | The function is | casadi::Functio |
|                 |                 | a derivative of | nInternal       |
|                 |                 | another         |                 |
|                 |                 | function. The   |                 |
|                 |                 | type of         |                 |
|                 |                 | derivative      |                 |
|                 |                 | (directional    |                 |
|                 |                 | derivative,     |                 |
|                 |                 | Jacobian) is    |                 |
|                 |                 | inferred from   |                 |
|                 |                 | the function    |                 |
|                 |                 | name.           |                 |
+-----------------+-----------------+-----------------+-----------------+
| discrete        | OT_BOOLVECTOR   | Indicates which | casadi::Qpsol   |
|                 |                 | of the          |                 |
|                 |                 | variables are   |                 |
|                 |                 | discrete, i.e.  |                 |
|                 |                 | integer-valued  |                 |
+-----------------+-----------------+-----------------+-----------------+
| gather_stats    | OT_BOOL         | Flag to         | casadi::Functio |
|                 |                 | indicate        | nInternal       |
|                 |                 | whether         |                 |
|                 |                 | statistics must |                 |
|                 |                 | be gathered     |                 |
+-----------------+-----------------+-----------------+-----------------+
| input_scheme    | OT_STRINGVECTOR | Custom input    | casadi::Functio |
|                 |                 | scheme          | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| inputs_check    | OT_BOOL         | Throw           | casadi::Functio |
|                 |                 | exceptions when | nInternal       |
|                 |                 | the numerical   |                 |
|                 |                 | values of the   |                 |
|                 |                 | inputs don't    |                 |
|                 |                 | make sense      |                 |
+-----------------+-----------------+-----------------+-----------------+
| jac_penalty     | OT_DOUBLE       | When requested  | casadi::Functio |
|                 |                 | for a number of | nInternal       |
|                 |                 | forward/reverse |                 |
|                 |                 | directions, it  |                 |
|                 |                 | may be cheaper  |                 |
|                 |                 | to compute      |                 |
|                 |                 | first the full  |                 |
|                 |                 | jacobian and    |                 |
|                 |                 | then multiply   |                 |
|                 |                 | with seeds,     |                 |
|                 |                 | rather than     |                 |
|                 |                 | obtain the      |                 |
|                 |                 | requested       |                 |
|                 |                 | directions in a |                 |
|                 |                 | straightforward |                 |
|                 |                 | manner. Casadi  |                 |
|                 |                 | uses a          |                 |
|                 |                 | heuristic to    |                 |
|                 |                 | decide which is |                 |
|                 |                 | cheaper. A high |                 |
|                 |                 | value of        |                 |
|                 |                 | 'jac_penalty'   |                 |
|                 |                 | makes it less   |                 |
|                 |                 | likely for the  |                 |
|                 |                 | heurstic to     |                 |
|                 |                 | chose the full  |                 |
|                 |                 | Jacobian        |                 |
|                 |                 | strategy. The   |                 |
|                 |                 | special value   |                 |
|                 |                 | -1 indicates    |                 |
|                 |                 | never to use    |                 |
|                 |                 | the full        |                 |
|                 |                 | Jacobian        |                 |
|                 |                 | strategy        |                 |
+-----------------+-----------------+-----------------+-----------------+
| jit             | OT_BOOL         | Use just-in-    | casadi::Functio |
|                 |                 | time compiler   | nInternal       |
|                 |                 | to speed up the |                 |
|                 |                 | evaluation      |                 |
+-----------------+-----------------+-----------------+-----------------+
| jit_options     | OT_DICT         | Options to be   | casadi::Functio |
|                 |                 | passed to the   | nInternal       |
|                 |                 | jit compiler.   |                 |
+-----------------+-----------------+-----------------+-----------------+
| monitor         | OT_STRINGVECTOR | Monitors to be  | casadi::Functio |
|                 |                 | activated       | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| output_scheme   | OT_STRINGVECTOR | Custom output   | casadi::Functio |
|                 |                 | scheme          | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| regularity_chec | OT_BOOL         | Throw           | casadi::Functio |
| k               |                 | exceptions when | nInternal       |
|                 |                 | NaN or Inf      |                 |
|                 |                 | appears during  |                 |
|                 |                 | evaluation      |                 |
+-----------------+-----------------+-----------------+-----------------+
| user_data       | OT_VOIDPTR      | A user-defined  | casadi::Functio |
|                 |                 | field that can  | nInternal       |
|                 |                 | be used to      |                 |
|                 |                 | identify the    |                 |
|                 |                 | function or     |                 |
|                 |                 | pass additional |                 |
|                 |                 | information     |                 |
+-----------------+-----------------+-----------------+-----------------+
| verbose         | OT_BOOL         | Verbose         | casadi::Functio |
|                 |                 | evaluation  for | nInternal       |
|                 |                 | debugging       |                 |
+-----------------+-----------------+-----------------+-----------------+

>Input scheme: casadi::QpsolInput (QPSOL_NUM_IN = 9) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| QPSOL_H                |                        | The square matrix H:   |
|                        |                        | sparse, (n x n). Only  |
|                        |                        | the lower triangular   |
|                        |                        | part is actually used. |
|                        |                        | The matrix is assumed  |
|                        |                        | to be symmetrical.     |
+------------------------+------------------------+------------------------+
| QPSOL_G                |                        | The vector g: dense,   |
|                        |                        | (n x 1)                |
+------------------------+------------------------+------------------------+
| QPSOL_A                |                        | The matrix A: sparse,  |
|                        |                        | (nc x n) - product     |
|                        |                        | with x must be dense.  |
+------------------------+------------------------+------------------------+
| QPSOL_LBA              |                        | dense, (nc x 1)        |
+------------------------+------------------------+------------------------+
| QPSOL_UBA              |                        | dense, (nc x 1)        |
+------------------------+------------------------+------------------------+
| QPSOL_LBX              |                        | dense, (n x 1)         |
+------------------------+------------------------+------------------------+
| QPSOL_UBX              |                        | dense, (n x 1)         |
+------------------------+------------------------+------------------------+
| QPSOL_X0               |                        | dense, (n x 1)         |
+------------------------+------------------------+------------------------+
| QPSOL_LAM_X0           |                        | dense                  |
+------------------------+------------------------+------------------------+

>Output scheme: casadi::QpsolOutput (QPSOL_NUM_OUT = 4) []

+-------------+-------+---------------------------------------------------+
|  Full name  | Short |                    Description                    |
+=============+=======+===================================================+
| QPSOL_X     |       | The primal solution.                              |
+-------------+-------+---------------------------------------------------+
| QPSOL_COST  |       | The optimal cost.                                 |
+-------------+-------+---------------------------------------------------+
| QPSOL_LAM_A |       | The dual solution corresponding to linear bounds. |
+-------------+-------+---------------------------------------------------+
| QPSOL_LAM_X |       | The dual solution corresponding to simple bounds. |
+-------------+-------+---------------------------------------------------+

List of plugins
===============



- cplex

- gurobi

- ooqp

- qpoases

- sqic

- nlp

Note: some of the plugins in this list might not be available on your
system. Also, there might be extra plugins available to you that are not
listed here. You can obtain their documentation with
Qpsol.doc(\"myextraplugin\")



--------------------------------------------------------------------------------

cplex
-----



Interface to Cplex solver for sparse Quadratic Programs

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| cplex                  | OT_DICT                | Options to be passed   |
|                        |                        | to CPLEX               |
+------------------------+------------------------+------------------------+
| dep_check              | OT_INT                 | Detect redundant       |
|                        |                        | constraints.           |
+------------------------+------------------------+------------------------+
| dump_filename          | OT_STRING              | The filename to dump   |
|                        |                        | to.                    |
+------------------------+------------------------+------------------------+
| dump_to_file           | OT_BOOL                | Dumps QP to file in    |
|                        |                        | CPLEX format.          |
+------------------------+------------------------+------------------------+
| qp_method              | OT_INT                 | Determines which CPLEX |
|                        |                        | algorithm to use.      |
+------------------------+------------------------+------------------------+
| tol                    | OT_DOUBLE              | Tolerance of solver    |
+------------------------+------------------------+------------------------+
| warm_start             | OT_BOOL                | Use warm start with    |
|                        |                        | simplex methods        |
|                        |                        | (affects only the      |
|                        |                        | simplex methods).      |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

gurobi
------



Interface to the GUROBI Solver for quadratic programming

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| vtype                  | OT_STRINGVECTOR        | Type of variables: [CO |
|                        |                        | NTINUOUS|binary|intege |
|                        |                        | r|semicont|semiint]    |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

ooqp
----



Interface to the OOQP Solver for quadratic programming The current
implementation assumes that OOQP is configured with the MA27 sparse linear
solver.

NOTE: when doing multiple calls to evaluate(), check if you need to
reInit();

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| artol                  | OT_DOUBLE              | tolerance as provided  |
|                        |                        | with setArTol to OOQP  |
+------------------------+------------------------+------------------------+
| mutol                  | OT_DOUBLE              | tolerance as provided  |
|                        |                        | with setMuTol to OOQP  |
+------------------------+------------------------+------------------------+
| print_level            | OT_INT                 | Print level. OOQP      |
|                        |                        | listens to print_level |
|                        |                        | 0, 10 and 100          |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

qpoases
-------



Interface to QPOases Solver for quadratic programming

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| CPUtime                | OT_DOUBLE              | The maximum allowed    |
|                        |                        | CPU time in seconds    |
|                        |                        | for the whole          |
|                        |                        | initialisation (and    |
|                        |                        | the actually required  |
|                        |                        | one on output).        |
|                        |                        | Disabled if unset.     |
+------------------------+------------------------+------------------------+
| boundRelaxation        | OT_DOUBLE              | Initial relaxation of  |
|                        |                        | bounds to start        |
|                        |                        | homotopy and initial   |
|                        |                        | value for far bounds.  |
+------------------------+------------------------+------------------------+
| boundTolerance         | OT_DOUBLE              | If upper and lower     |
|                        |                        | bounds differ less     |
|                        |                        | than this tolerance,   |
|                        |                        | they are regarded      |
|                        |                        | equal, i.e. as         |
|                        |                        | equality constraint.   |
+------------------------+------------------------+------------------------+
| enableCholeskyRefactor | OT_INT                 | Specifies the          |
| isation                |                        | frequency of a full    |
|                        |                        | re-factorisation of    |
|                        |                        | projected Hessian      |
|                        |                        | matrix: 0: turns them  |
|                        |                        | off, 1: uses them at   |
|                        |                        | each iteration etc.    |
+------------------------+------------------------+------------------------+
| enableDriftCorrection  | OT_INT                 | Specifies the          |
|                        |                        | frequency of drift     |
|                        |                        | corrections: 0: turns  |
|                        |                        | them off.              |
+------------------------+------------------------+------------------------+
| enableEqualities       | OT_BOOL                | Specifies whether      |
|                        |                        | equalities should be   |
|                        |                        | treated as always      |
|                        |                        | active (True) or not   |
|                        |                        | (False)                |
+------------------------+------------------------+------------------------+
| enableFarBounds        | OT_BOOL                | Enables the use of far |
|                        |                        | bounds.                |
+------------------------+------------------------+------------------------+
| enableFlippingBounds   | OT_BOOL                | Enables the use of     |
|                        |                        | flipping bounds.       |
+------------------------+------------------------+------------------------+
| enableFullLITests      | OT_BOOL                | Enables condition-     |
|                        |                        | hardened (but more     |
|                        |                        | expensive) LI test.    |
+------------------------+------------------------+------------------------+
| enableNZCTests         | OT_BOOL                | Enables nonzero        |
|                        |                        | curvature tests.       |
+------------------------+------------------------+------------------------+
| enableRamping          | OT_BOOL                | Enables ramping.       |
+------------------------+------------------------+------------------------+
| enableRegularisation   | OT_BOOL                | Enables automatic      |
|                        |                        | Hessian                |
|                        |                        | regularisation.        |
+------------------------+------------------------+------------------------+
| epsDen                 | OT_DOUBLE              | Denominator tolerance  |
|                        |                        | for ratio tests.       |
+------------------------+------------------------+------------------------+
| epsFlipping            | OT_DOUBLE              | Tolerance of squared   |
|                        |                        | Cholesky diagonal      |
|                        |                        | factor which triggers  |
|                        |                        | flipping bound.        |
+------------------------+------------------------+------------------------+
| epsIterRef             | OT_DOUBLE              | Early termination      |
|                        |                        | tolerance for          |
|                        |                        | iterative refinement.  |
+------------------------+------------------------+------------------------+
| epsLITests             | OT_DOUBLE              | Tolerance for linear   |
|                        |                        | independence tests.    |
+------------------------+------------------------+------------------------+
| epsNZCTests            | OT_DOUBLE              | Tolerance for nonzero  |
|                        |                        | curvature tests.       |
+------------------------+------------------------+------------------------+
| epsNum                 | OT_DOUBLE              | Numerator tolerance    |
|                        |                        | for ratio tests.       |
+------------------------+------------------------+------------------------+
| epsRegularisation      | OT_DOUBLE              | Scaling factor of      |
|                        |                        | identity matrix used   |
|                        |                        | for Hessian            |
|                        |                        | regularisation.        |
+------------------------+------------------------+------------------------+
| finalRamping           | OT_DOUBLE              | Final value for        |
|                        |                        | ramping strategy.      |
+------------------------+------------------------+------------------------+
| growFarBounds          | OT_DOUBLE              | Factor to grow far     |
|                        |                        | bounds.                |
+------------------------+------------------------+------------------------+
| initialFarBounds       | OT_DOUBLE              | Initial size for far   |
|                        |                        | bounds.                |
+------------------------+------------------------+------------------------+
| initialRamping         | OT_DOUBLE              | Start value for        |
|                        |                        | ramping strategy.      |
+------------------------+------------------------+------------------------+
| initialStatusBounds    | OT_STRING              | Initial status of      |
|                        |                        | bounds at first        |
|                        |                        | iteration.             |
+------------------------+------------------------+------------------------+
| maxDualJump            | OT_DOUBLE              | Maximum allowed jump   |
|                        |                        | in dual variables in   |
|                        |                        | linear independence    |
|                        |                        | tests.                 |
+------------------------+------------------------+------------------------+
| maxPrimalJump          | OT_DOUBLE              | Maximum allowed jump   |
|                        |                        | in primal variables in |
|                        |                        | nonzero curvature      |
|                        |                        | tests.                 |
+------------------------+------------------------+------------------------+
| nWSR                   | OT_INT                 | The maximum number of  |
|                        |                        | working set            |
|                        |                        | recalculations to be   |
|                        |                        | performed during the   |
|                        |                        | initial homotopy.      |
|                        |                        | Default is 5(nx + nc)  |
+------------------------+------------------------+------------------------+
| numRefinementSteps     | OT_INT                 | Maximum number of      |
|                        |                        | iterative refinement   |
|                        |                        | steps.                 |
+------------------------+------------------------+------------------------+
| numRegularisationSteps | OT_INT                 | Maximum number of      |
|                        |                        | successive             |
|                        |                        | regularisation steps.  |
+------------------------+------------------------+------------------------+
| printLevel             | OT_STRING              | Defines the amount of  |
|                        |                        | text output during QP  |
|                        |                        | solution, see Section  |
|                        |                        | 5.7                    |
+------------------------+------------------------+------------------------+
| sparse                 | OT_BOOL                | Formulate the QP using |
|                        |                        | sparse matrices.       |
|                        |                        | Default: false         |
+------------------------+------------------------+------------------------+
| terminationTolerance   | OT_DOUBLE              | Relative termination   |
|                        |                        | tolerance to stop      |
|                        |                        | homotopy.              |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

sqic
----



Interface to the SQIC solver for quadratic programming

--------------------------------------------------------------------------------





--------------------------------------------------------------------------------

nlp
---



Solve QPs using an Nlpsol

>List of available options

+----------------+-----------+---------------------------------+
|       Id       |   Type    |           Description           |
+================+===========+=================================+
| nlpsol         | OT_STRING | Name of solver.                 |
+----------------+-----------+---------------------------------+
| nlpsol_options | OT_DICT   | Options to be passed to solver. |
+----------------+-----------+---------------------------------+

--------------------------------------------------------------------------------



Joel Andersson

";

%feature("docstring") casadi::zip "[INTERNAL] ";

%feature("docstring") casadi::linsol_out "

Linear solver output scheme.

";

%feature("docstring") casadi::replaceMat "[INTERNAL] ";

%feature("docstring") casadi::linsol_n_in "

Number of linear solver inputs.

";

%feature("docstring") casadi::collocationInterpolators "

Obtain collocation interpolating matrices.

Parameters:
-----------

tau_root:  location of collocation points, as obtained from
collocationPoints

C:  interpolating coefficients to obtain derivatives Length: order+1, order
+ 1



::

dX/dt @collPoint(j) ~ Sum_i C[j][i]*X@collPoint(i)



Parameters:
-----------

D:  interpolating coefficients to obtain end state Length: order+1

";

%feature("docstring") casadi::nlpsol_n_in "

Number of NLP solver inputs.

";

%feature("docstring") casadi::matrixName "

Get typename.

";

%feature("docstring") casadi::integrator_in "

>  CASADI_EXPORT[str] integrator_in()
------------------------------------------------------------------------

Get input scheme of integrators.

>  str integrator_in(int ind)
------------------------------------------------------------------------

Get integrator input scheme name by index.

";

%feature("docstring") casadi::is_regular "

>  bool is_regular([T ] v)
------------------------------------------------------------------------

Checks if array does not contain NaN or Inf.

>  bool is_regular([N_] v)
------------------------------------------------------------------------
[INTERNAL] 
";

%feature("docstring") casadi::casadi_nrm2 "[INTERNAL]  NRM2: ||x||_2 ->
return.

";

%feature("docstring") casadi::hash_value "

Hash value of an integer.

";

%feature("docstring") casadi::casadi_polyval "[INTERNAL]  Evaluate a
polynomial.

";

%feature("docstring") casadi::qpsol_out "

>  CASADI_EXPORT[str] qpsol_out()
------------------------------------------------------------------------

Get QP solver output scheme of QP solvers.

>  str qpsol_out(int ind)
------------------------------------------------------------------------

Get output scheme name by index.

";

%feature("docstring") casadi::simpleIntegrator "

Simplified wrapper for the Integrator class Constructs an integrator using
the same syntax as simpleRK and simpleIRK. The constructed function has
three inputs, corresponding to initial state (x0), parameter (p) and
integration time (h) and one output, corresponding to final state (xf).

Parameters:
-----------

f:  ODE function with two inputs (x and p) and one output (xdot)

N:  Number of integrator steps

order:  Order of interpolating polynomials

scheme:  Collocation scheme, as excepted by collocationPoints function.

";

%feature("docstring") casadi::nlpsol_default_in "

Default input for an NLP solver.

";

%feature("docstring") casadi::_nl_var "[INTERNAL] ";

%feature("docstring") casadi::casadi_norm_inf "[INTERNAL]  Inf-norm of a
vector * Returns the largest element in absolute value

";

%feature("docstring") casadi::dgetrf_ "[INTERNAL]  LU-Factorize dense
matrix (lapack)

";

%feature("docstring") casadi::has_rootfinder "

Check if a particular plugin is available.

";

%feature("docstring") casadi::casadi_project "[INTERNAL]  Sparse copy: y <-
x, w work vector (length >= number of rows)

";

%feature("docstring") casadi::matrixName< int > "

Get typename.

";

%feature("docstring") casadi::is_slice "

>  bool is_slice(IM x, bool ind1=false)
------------------------------------------------------------------------

Is the IM a Slice.

>  bool is_slice([int ] v, bool ind1=false)
------------------------------------------------------------------------

Check if an index vector can be represented more efficiently as a slice.

";

%feature("docstring") casadi::isIncreasing "

Check if the vector is strictly increasing.

";

%feature("docstring") casadi::doc_integrator "

Get the documentation string for a plugin.

";

%feature("docstring") casadi::casadi_dot "[INTERNAL]  Inner product.

";

%feature("docstring") casadi::integrator_out "

>  CASADI_EXPORT[str] integrator_out()
------------------------------------------------------------------------

Get integrator output scheme of integrators.

>  str integrator_out(int ind)
------------------------------------------------------------------------

Get output scheme name by index.

";

%feature("docstring") casadi::load_qpsol "

Explicitly load a plugin dynamically.

";

%feature("docstring") casadi::casadi_trans "[INTERNAL]  TRANS: y <-
trans(x)

";

%feature("docstring") casadi::dgetrs_ "[INTERNAL]   Solve a system of
equation using an LU-factorized matrix (lapack)

";

%feature("docstring") casadi::_jtimes "[INTERNAL] ";

%feature("docstring") casadi::casadi_bilin "[INTERNAL]  Calculates dot(x,
mul(A, y))

";

%feature("docstring") casadi::casadi_mv "[INTERNAL]  Sparse matrix-vector
multiplication: z <- z + x*y.

";

%feature("docstring") casadi::nlpsol_out "

>  CASADI_EXPORT[str] nlpsol_out()
------------------------------------------------------------------------

Get NLP solver output scheme of NLP solvers.

>Output scheme: casadi::NlpsolOutput (NLPSOL_NUM_OUT = 6) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| NLPSOL_X               |                        | Decision variables at  |
|                        |                        | the optimal solution   |
|                        |                        | (nx x 1)               |
+------------------------+------------------------+------------------------+
| NLPSOL_F               |                        | Cost function value at |
|                        |                        | the optimal solution   |
|                        |                        | (1 x 1)                |
+------------------------+------------------------+------------------------+
| NLPSOL_G               |                        | Constraints function   |
|                        |                        | at the optimal         |
|                        |                        | solution (ng x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_X           |                        | Lagrange multipliers   |
|                        |                        | for bounds on X at the |
|                        |                        | solution (nx x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_G           |                        | Lagrange multipliers   |
|                        |                        | for bounds on G at the |
|                        |                        | solution (ng x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_P           |                        | Lagrange multipliers   |
|                        |                        | for bounds on P at the |
|                        |                        | solution (np x 1)      |
+------------------------+------------------------+------------------------+

>  str nlpsol_out(int ind)
------------------------------------------------------------------------

Get output scheme name by index.

>Output scheme: casadi::NlpsolOutput (NLPSOL_NUM_OUT = 6) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| NLPSOL_X               |                        | Decision variables at  |
|                        |                        | the optimal solution   |
|                        |                        | (nx x 1)               |
+------------------------+------------------------+------------------------+
| NLPSOL_F               |                        | Cost function value at |
|                        |                        | the optimal solution   |
|                        |                        | (1 x 1)                |
+------------------------+------------------------+------------------------+
| NLPSOL_G               |                        | Constraints function   |
|                        |                        | at the optimal         |
|                        |                        | solution (ng x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_X           |                        | Lagrange multipliers   |
|                        |                        | for bounds on X at the |
|                        |                        | solution (nx x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_G           |                        | Lagrange multipliers   |
|                        |                        | for bounds on G at the |
|                        |                        | solution (ng x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_P           |                        | Lagrange multipliers   |
|                        |                        | for bounds on P at the |
|                        |                        | solution (np x 1)      |
+------------------------+------------------------+------------------------+

";

%feature("docstring") casadi::has_linsol "

Check if a particular plugin is available.

";

%feature("docstring") casadi::casadi_asum "[INTERNAL]  ASUM: ||x||_1 ->
return.

";

%feature("docstring") casadi::simpleRK "

Construct an explicit Runge-Kutta integrator The constructed function has
three inputs, corresponding to initial state (x0), parameter (p) and
integration time (h) and one output, corresponding to final state (xf).

Parameters:
-----------

f:  ODE function with two inputs (x and p) and one output (xdot)

N:  Number of integrator steps

order:  Order of interpolating polynomials

";

%feature("docstring") casadi::nlpsol_in "

>  CASADI_EXPORT[str] nlpsol_in()
------------------------------------------------------------------------

Get input scheme of NLP solvers.

>Input scheme: casadi::NlpsolInput (NLPSOL_NUM_IN = 8) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| NLPSOL_X0              |                        | Decision variables,    |
|                        |                        | initial guess (nx x 1) |
+------------------------+------------------------+------------------------+
| NLPSOL_P               |                        | Value of fixed         |
|                        |                        | parameters (np x 1)    |
+------------------------+------------------------+------------------------+
| NLPSOL_LBX             |                        | Decision variables     |
|                        |                        | lower bound (nx x 1),  |
|                        |                        | default -inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_UBX             |                        | Decision variables     |
|                        |                        | upper bound (nx x 1),  |
|                        |                        | default +inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_LBG             |                        | Constraints lower      |
|                        |                        | bound (ng x 1),        |
|                        |                        | default -inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_UBG             |                        | Constraints upper      |
|                        |                        | bound (ng x 1),        |
|                        |                        | default +inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_X0          |                        | Lagrange multipliers   |
|                        |                        | for bounds on X,       |
|                        |                        | initial guess (nx x 1) |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_G0          |                        | Lagrange multipliers   |
|                        |                        | for bounds on G,       |
|                        |                        | initial guess (ng x 1) |
+------------------------+------------------------+------------------------+

>  str nlpsol_in(int ind)
------------------------------------------------------------------------

Get NLP solver input scheme name by index.

>Input scheme: casadi::NlpsolInput (NLPSOL_NUM_IN = 8) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| NLPSOL_X0              |                        | Decision variables,    |
|                        |                        | initial guess (nx x 1) |
+------------------------+------------------------+------------------------+
| NLPSOL_P               |                        | Value of fixed         |
|                        |                        | parameters (np x 1)    |
+------------------------+------------------------+------------------------+
| NLPSOL_LBX             |                        | Decision variables     |
|                        |                        | lower bound (nx x 1),  |
|                        |                        | default -inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_UBX             |                        | Decision variables     |
|                        |                        | upper bound (nx x 1),  |
|                        |                        | default +inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_LBG             |                        | Constraints lower      |
|                        |                        | bound (ng x 1),        |
|                        |                        | default -inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_UBG             |                        | Constraints upper      |
|                        |                        | bound (ng x 1),        |
|                        |                        | default +inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_X0          |                        | Lagrange multipliers   |
|                        |                        | for bounds on X,       |
|                        |                        | initial guess (nx x 1) |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_G0          |                        | Lagrange multipliers   |
|                        |                        | for bounds on G,       |
|                        |                        | initial guess (ng x 1) |
+------------------------+------------------------+------------------------+

";

%feature("docstring") casadi::casadi_densify "[INTERNAL]  Convert sparse to
dense.

";

%feature("docstring") casadi::doc_rootfinder "

Get the documentation string for a plugin.

";

%feature("docstring") casadi::nlpsol "

>  Function nlpsol(str name, str solver, const str:SX &nlp, Dict opts=Dict())
------------------------------------------------------------------------

Create an NLP solver Creates a solver for the following parametric nonlinear
program (NLP):

::

  min          F(x, p)
  x
  
  subject to
  LBX <=   x    <= UBX
  LBG <= G(x, p) <= UBG
  p  == P
  
  nx: number of decision variables
  ng: number of constraints
  np: number of parameters



General information
===================



>List of available options

+-----------------+-----------------+-----------------+-----------------+
|       Id        |      Type       |   Description   |     Used in     |
+=================+=================+=================+=================+
| ad_weight       | OT_DOUBLE       | Weighting       | casadi::Functio |
|                 |                 | factor for      | nInternal       |
|                 |                 | derivative calc |                 |
|                 |                 | ulation.When    |                 |
|                 |                 | there is an     |                 |
|                 |                 | option of       |                 |
|                 |                 | either using    |                 |
|                 |                 | forward or      |                 |
|                 |                 | reverse mode    |                 |
|                 |                 | directional     |                 |
|                 |                 | derivatives,    |                 |
|                 |                 | the condition a |                 |
|                 |                 | d_weight*nf<=(1 |                 |
|                 |                 | -ad_weight)*na  |                 |
|                 |                 | is used where   |                 |
|                 |                 | nf and na are   |                 |
|                 |                 | estimates of    |                 |
|                 |                 | the number of   |                 |
|                 |                 | forward/reverse |                 |
|                 |                 | mode            |                 |
|                 |                 | directional     |                 |
|                 |                 | derivatives     |                 |
|                 |                 | needed. By      |                 |
|                 |                 | default,        |                 |
|                 |                 | ad_weight is    |                 |
|                 |                 | calculated      |                 |
|                 |                 | automatically,  |                 |
|                 |                 | but this can be |                 |
|                 |                 | overridden by   |                 |
|                 |                 | setting this    |                 |
|                 |                 | option. In      |                 |
|                 |                 | particular, 0   |                 |
|                 |                 | means forcing   |                 |
|                 |                 | forward mode    |                 |
|                 |                 | and 1 forcing   |                 |
|                 |                 | reverse mode.   |                 |
|                 |                 | Leave unset for |                 |
|                 |                 | (class          |                 |
|                 |                 | specific)       |                 |
|                 |                 | heuristics.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| ad_weight_sp    | OT_DOUBLE       | Weighting       | casadi::Functio |
|                 |                 | factor for      | nInternal       |
|                 |                 | sparsity        |                 |
|                 |                 | pattern         |                 |
|                 |                 | calculation cal |                 |
|                 |                 | culation.Overri |                 |
|                 |                 | des default     |                 |
|                 |                 | behavior. Set   |                 |
|                 |                 | to 0 and 1 to   |                 |
|                 |                 | force forward   |                 |
|                 |                 | and reverse     |                 |
|                 |                 | mode            |                 |
|                 |                 | respectively.   |                 |
|                 |                 | Cf. option      |                 |
|                 |                 | \"ad_weight\".    |                 |
+-----------------+-----------------+-----------------+-----------------+
| compiler        | OT_STRING       | Just-in-time    | casadi::Functio |
|                 |                 | compiler plugin | nInternal       |
|                 |                 | to be used.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| derivative_of   | OT_FUNCTION     | The function is | casadi::Functio |
|                 |                 | a derivative of | nInternal       |
|                 |                 | another         |                 |
|                 |                 | function. The   |                 |
|                 |                 | type of         |                 |
|                 |                 | derivative      |                 |
|                 |                 | (directional    |                 |
|                 |                 | derivative,     |                 |
|                 |                 | Jacobian) is    |                 |
|                 |                 | inferred from   |                 |
|                 |                 | the function    |                 |
|                 |                 | name.           |                 |
+-----------------+-----------------+-----------------+-----------------+
| eval_errors_fat | OT_BOOL         | When errors     | casadi::Nlpsol  |
| al              |                 | occur during    |                 |
|                 |                 | evaluation of   |                 |
|                 |                 | f,g,...,stop    |                 |
|                 |                 | the iterations  |                 |
+-----------------+-----------------+-----------------+-----------------+
| expand          | OT_BOOL         | Replace MX with | casadi::Nlpsol  |
|                 |                 | SX expressions  |                 |
|                 |                 | in problem      |                 |
|                 |                 | formulation     |                 |
|                 |                 | [false]         |                 |
+-----------------+-----------------+-----------------+-----------------+
| gather_stats    | OT_BOOL         | Flag to         | casadi::Functio |
|                 |                 | indicate        | nInternal       |
|                 |                 | whether         |                 |
|                 |                 | statistics must |                 |
|                 |                 | be gathered     |                 |
+-----------------+-----------------+-----------------+-----------------+
| ignore_check_ve | OT_BOOL         | If set to true, | casadi::Nlpsol  |
| c               |                 | the input shape |                 |
|                 |                 | of F will not   |                 |
|                 |                 | be checked.     |                 |
+-----------------+-----------------+-----------------+-----------------+
| input_scheme    | OT_STRINGVECTOR | Custom input    | casadi::Functio |
|                 |                 | scheme          | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| inputs_check    | OT_BOOL         | Throw           | casadi::Functio |
|                 |                 | exceptions when | nInternal       |
|                 |                 | the numerical   |                 |
|                 |                 | values of the   |                 |
|                 |                 | inputs don't    |                 |
|                 |                 | make sense      |                 |
+-----------------+-----------------+-----------------+-----------------+
| iteration_callb | OT_FUNCTION     | A function that | casadi::Nlpsol  |
| ack             |                 | will be called  |                 |
|                 |                 | at each         |                 |
|                 |                 | iteration with  |                 |
|                 |                 | the solver as   |                 |
|                 |                 | input. Check    |                 |
|                 |                 | documentation   |                 |
|                 |                 | of Callback .   |                 |
+-----------------+-----------------+-----------------+-----------------+
| iteration_callb | OT_BOOL         | If set to true, | casadi::Nlpsol  |
| ack_ignore_erro |                 | errors thrown   |                 |
| rs              |                 | by iteration_ca |                 |
|                 |                 | llback will be  |                 |
|                 |                 | ignored.        |                 |
+-----------------+-----------------+-----------------+-----------------+
| iteration_callb | OT_INT          | Only call the   | casadi::Nlpsol  |
| ack_step        |                 | callback        |                 |
|                 |                 | function every  |                 |
|                 |                 | few iterations. |                 |
+-----------------+-----------------+-----------------+-----------------+
| jac_penalty     | OT_DOUBLE       | When requested  | casadi::Functio |
|                 |                 | for a number of | nInternal       |
|                 |                 | forward/reverse |                 |
|                 |                 | directions, it  |                 |
|                 |                 | may be cheaper  |                 |
|                 |                 | to compute      |                 |
|                 |                 | first the full  |                 |
|                 |                 | jacobian and    |                 |
|                 |                 | then multiply   |                 |
|                 |                 | with seeds,     |                 |
|                 |                 | rather than     |                 |
|                 |                 | obtain the      |                 |
|                 |                 | requested       |                 |
|                 |                 | directions in a |                 |
|                 |                 | straightforward |                 |
|                 |                 | manner. Casadi  |                 |
|                 |                 | uses a          |                 |
|                 |                 | heuristic to    |                 |
|                 |                 | decide which is |                 |
|                 |                 | cheaper. A high |                 |
|                 |                 | value of        |                 |
|                 |                 | 'jac_penalty'   |                 |
|                 |                 | makes it less   |                 |
|                 |                 | likely for the  |                 |
|                 |                 | heurstic to     |                 |
|                 |                 | chose the full  |                 |
|                 |                 | Jacobian        |                 |
|                 |                 | strategy. The   |                 |
|                 |                 | special value   |                 |
|                 |                 | -1 indicates    |                 |
|                 |                 | never to use    |                 |
|                 |                 | the full        |                 |
|                 |                 | Jacobian        |                 |
|                 |                 | strategy        |                 |
+-----------------+-----------------+-----------------+-----------------+
| jit             | OT_BOOL         | Use just-in-    | casadi::Functio |
|                 |                 | time compiler   | nInternal       |
|                 |                 | to speed up the |                 |
|                 |                 | evaluation      |                 |
+-----------------+-----------------+-----------------+-----------------+
| jit_options     | OT_DICT         | Options to be   | casadi::Functio |
|                 |                 | passed to the   | nInternal       |
|                 |                 | jit compiler.   |                 |
+-----------------+-----------------+-----------------+-----------------+
| monitor         | OT_STRINGVECTOR | Monitors to be  | casadi::Functio |
|                 |                 | activated       | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| output_scheme   | OT_STRINGVECTOR | Custom output   | casadi::Functio |
|                 |                 | scheme          | nInternal       |
+-----------------+-----------------+-----------------+-----------------+
| print_time      | OT_BOOL         | print           | casadi::Nlpsol  |
|                 |                 | information     |                 |
|                 |                 | about execution |                 |
|                 |                 | time            |                 |
+-----------------+-----------------+-----------------+-----------------+
| regularity_chec | OT_BOOL         | Throw           | casadi::Functio |
| k               |                 | exceptions when | nInternal       |
|                 |                 | NaN or Inf      |                 |
|                 |                 | appears during  |                 |
|                 |                 | evaluation      |                 |
+-----------------+-----------------+-----------------+-----------------+
| user_data       | OT_VOIDPTR      | A user-defined  | casadi::Functio |
|                 |                 | field that can  | nInternal       |
|                 |                 | be used to      |                 |
|                 |                 | identify the    |                 |
|                 |                 | function or     |                 |
|                 |                 | pass additional |                 |
|                 |                 | information     |                 |
+-----------------+-----------------+-----------------+-----------------+
| verbose         | OT_BOOL         | Verbose         | casadi::Functio |
|                 |                 | evaluation  for | nInternal       |
|                 |                 | debugging       |                 |
+-----------------+-----------------+-----------------+-----------------+
| verbose_init    | OT_BOOL         | Print out       | casadi::Nlpsol  |
|                 |                 | timing          |                 |
|                 |                 | information     |                 |
|                 |                 | about the       |                 |
|                 |                 | different       |                 |
|                 |                 | stages of       |                 |
|                 |                 | initialization  |                 |
+-----------------+-----------------+-----------------+-----------------+
| warn_initial_bo | OT_BOOL         | Warn if the     | casadi::Nlpsol  |
| unds            |                 | initial guess   |                 |
|                 |                 | does not        |                 |
|                 |                 | satisfy LBX and |                 |
|                 |                 | UBX             |                 |
+-----------------+-----------------+-----------------+-----------------+

>Input scheme: casadi::NlpsolInput (NLPSOL_NUM_IN = 8) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| NLPSOL_X0              |                        | Decision variables,    |
|                        |                        | initial guess (nx x 1) |
+------------------------+------------------------+------------------------+
| NLPSOL_P               |                        | Value of fixed         |
|                        |                        | parameters (np x 1)    |
+------------------------+------------------------+------------------------+
| NLPSOL_LBX             |                        | Decision variables     |
|                        |                        | lower bound (nx x 1),  |
|                        |                        | default -inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_UBX             |                        | Decision variables     |
|                        |                        | upper bound (nx x 1),  |
|                        |                        | default +inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_LBG             |                        | Constraints lower      |
|                        |                        | bound (ng x 1),        |
|                        |                        | default -inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_UBG             |                        | Constraints upper      |
|                        |                        | bound (ng x 1),        |
|                        |                        | default +inf.          |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_X0          |                        | Lagrange multipliers   |
|                        |                        | for bounds on X,       |
|                        |                        | initial guess (nx x 1) |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_G0          |                        | Lagrange multipliers   |
|                        |                        | for bounds on G,       |
|                        |                        | initial guess (ng x 1) |
+------------------------+------------------------+------------------------+

>Output scheme: casadi::NlpsolOutput (NLPSOL_NUM_OUT = 6) []

+------------------------+------------------------+------------------------+
|       Full name        |         Short          |      Description       |
+========================+========================+========================+
| NLPSOL_X               |                        | Decision variables at  |
|                        |                        | the optimal solution   |
|                        |                        | (nx x 1)               |
+------------------------+------------------------+------------------------+
| NLPSOL_F               |                        | Cost function value at |
|                        |                        | the optimal solution   |
|                        |                        | (1 x 1)                |
+------------------------+------------------------+------------------------+
| NLPSOL_G               |                        | Constraints function   |
|                        |                        | at the optimal         |
|                        |                        | solution (ng x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_X           |                        | Lagrange multipliers   |
|                        |                        | for bounds on X at the |
|                        |                        | solution (nx x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_G           |                        | Lagrange multipliers   |
|                        |                        | for bounds on G at the |
|                        |                        | solution (ng x 1)      |
+------------------------+------------------------+------------------------+
| NLPSOL_LAM_P           |                        | Lagrange multipliers   |
|                        |                        | for bounds on P at the |
|                        |                        | solution (np x 1)      |
+------------------------+------------------------+------------------------+

List of plugins
===============



- ipopt

- knitro

- snopt

- worhp

- scpgen

- sqpmethod

Note: some of the plugins in this list might not be available on your
system. Also, there might be extra plugins available to you that are not
listed here. You can obtain their documentation with
Nlpsol.doc(\"myextraplugin\")



--------------------------------------------------------------------------------

ipopt
-----



When in warmstart mode, output NLPSOL_LAM_X may be used as input

NOTE: Even when max_iter == 0, it is not guaranteed that input(NLPSOL_X0) ==
output(NLPSOL_X). Indeed if bounds on X or constraints are unmet, they will
differ.

For a good tutorial on IPOPT,
seehttp://drops.dagstuhl.de/volltexte/2009/2089/pdf/09061.WaechterAndreas.Paper.2089.pdf

A good resource about the algorithms in IPOPT is: Wachter and L. T. Biegler,
On the Implementation of an Interior-Point Filter Line-Search Algorithm for
Large-Scale Nonlinear Programming, Mathematical Programming 106(1), pp.
25-57, 2006 (As Research Report RC 23149, IBM T. J. Watson Research Center,
Yorktown, USA

Caveats: with default options, multipliers for the decision variables are
wrong for equality constraints. Change the 'fixed_variable_treatment' to
'make_constraint' or 'relax_bounds' to obtain correct results.

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| con_integer_md         | OT_DICT                | Integer metadata (a    |
|                        |                        | dictionary with lists  |
|                        |                        | of integers) about     |
|                        |                        | constraints to be      |
|                        |                        | passed to IPOPT        |
+------------------------+------------------------+------------------------+
| con_numeric_md         | OT_DICT                | Numeric metadata (a    |
|                        |                        | dictionary with lists  |
|                        |                        | of reals) about        |
|                        |                        | constraints to be      |
|                        |                        | passed to IPOPT        |
+------------------------+------------------------+------------------------+
| con_string_md          | OT_DICT                | String metadata (a     |
|                        |                        | dictionary with lists  |
|                        |                        | of strings) about      |
|                        |                        | constraints to be      |
|                        |                        | passed to IPOPT        |
+------------------------+------------------------+------------------------+
| grad_f                 | OT_FUNCTION            | Function for           |
|                        |                        | calculating the        |
|                        |                        | gradient of the        |
|                        |                        | objective (column,     |
|                        |                        | autogenerated by       |
|                        |                        | default)               |
+------------------------+------------------------+------------------------+
| grad_f_options         | OT_DICT                | Options for the        |
|                        |                        | autogenerated gradient |
|                        |                        | of the objective.      |
+------------------------+------------------------+------------------------+
| hess_lag               | OT_FUNCTION            | Function for           |
|                        |                        | calculating the        |
|                        |                        | Hessian of the         |
|                        |                        | Lagrangian             |
|                        |                        | (autogenerated by      |
|                        |                        | default)               |
+------------------------+------------------------+------------------------+
| hess_lag_options       | OT_DICT                | Options for the        |
|                        |                        | autogenerated Hessian  |
|                        |                        | of the Lagrangian.     |
+------------------------+------------------------+------------------------+
| ipopt                  | OT_DICT                | Options to be passed   |
|                        |                        | to IPOPT               |
+------------------------+------------------------+------------------------+
| jac_g                  | OT_FUNCTION            | Function for           |
|                        |                        | calculating the        |
|                        |                        | Jacobian of the        |
|                        |                        | constraints            |
|                        |                        | (autogenerated by      |
|                        |                        | default)               |
+------------------------+------------------------+------------------------+
| jac_g_options          | OT_DICT                | Options for the        |
|                        |                        | autogenerated Jacobian |
|                        |                        | of the constraints.    |
+------------------------+------------------------+------------------------+
| pass_nonlinear_variabl | OT_BOOL                | Pass list of variables |
| es                     |                        | entering nonlinearly   |
|                        |                        | to IPOPT               |
+------------------------+------------------------+------------------------+
| var_integer_md         | OT_DICT                | Integer metadata (a    |
|                        |                        | dictionary with lists  |
|                        |                        | of integers) about     |
|                        |                        | variables to be passed |
|                        |                        | to IPOPT               |
+------------------------+------------------------+------------------------+
| var_numeric_md         | OT_DICT                | Numeric metadata (a    |
|                        |                        | dictionary with lists  |
|                        |                        | of reals) about        |
|                        |                        | variables to be passed |
|                        |                        | to IPOPT               |
+------------------------+------------------------+------------------------+
| var_string_md          | OT_DICT                | String metadata (a     |
|                        |                        | dictionary with lists  |
|                        |                        | of strings) about      |
|                        |                        | variables to be passed |
|                        |                        | to IPOPT               |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

knitro
------



KNITRO interface

>List of available options

+---------+--------------+--------------------------------+
|   Id    |     Type     |          Description           |
+=========+==============+================================+
| contype | OT_INTVECTOR | Type of constraint             |
+---------+--------------+--------------------------------+
| knitro  | OT_DICT      | Options to be passed to KNITRO |
+---------+--------------+--------------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

snopt
-----



SNOPT interface

>List of available options

+-------+---------+-------------------------------+
|  Id   |  Type   |          Description          |
+=======+=========+===============================+
| snopt | OT_DICT | Options to be passed to SNOPT |
+-------+---------+-------------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

worhp
-----



WORHP interface

>List of available options

+-------+---------+-------------------------------+
|  Id   |  Type   |          Description          |
+=======+=========+===============================+
| worhp | OT_DICT | Options to be passed to WORHP |
+-------+---------+-------------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

scpgen
------



A structure-exploiting sequential quadratic programming (to be come
sequential convex programming) method for nonlinear programming.

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| beta                   | OT_DOUBLE              | Line-search parameter, |
|                        |                        | restoration factor of  |
|                        |                        | stepsize               |
+------------------------+------------------------+------------------------+
| c1                     | OT_DOUBLE              | Armijo condition,      |
|                        |                        | coefficient of         |
|                        |                        | decrease in merit      |
+------------------------+------------------------+------------------------+
| codegen                | OT_BOOL                | C-code generation      |
+------------------------+------------------------+------------------------+
| hessian_approximation  | OT_STRING              | gauss-newton|exact     |
+------------------------+------------------------+------------------------+
| lbfgs_memory           | OT_INT                 | Size of L-BFGS memory. |
+------------------------+------------------------+------------------------+
| max_iter               | OT_INT                 | Maximum number of SQP  |
|                        |                        | iterations             |
+------------------------+------------------------+------------------------+
| max_iter_ls            | OT_INT                 | Maximum number of      |
|                        |                        | linesearch iterations  |
+------------------------+------------------------+------------------------+
| merit_memsize          | OT_INT                 | Size of memory to      |
|                        |                        | store history of merit |
|                        |                        | function values        |
+------------------------+------------------------+------------------------+
| merit_start            | OT_DOUBLE              | Lower bound for the    |
|                        |                        | merit function         |
|                        |                        | parameter              |
+------------------------+------------------------+------------------------+
| name_x                 | OT_STRINGVECTOR        | Names of the           |
|                        |                        | variables.             |
+------------------------+------------------------+------------------------+
| print_header           | OT_BOOL                | Print the header with  |
|                        |                        | problem statistics     |
+------------------------+------------------------+------------------------+
| print_x                | OT_INTVECTOR           | Which variables to     |
|                        |                        | print.                 |
+------------------------+------------------------+------------------------+
| qpsol                  | OT_STRING              | The QP solver to be    |
|                        |                        | used by the SQP method |
+------------------------+------------------------+------------------------+
| qpsol_options          | OT_DICT                | Options to be passed   |
|                        |                        | to the QP solver       |
+------------------------+------------------------+------------------------+
| reg_threshold          | OT_DOUBLE              | Threshold for the      |
|                        |                        | regularization.        |
+------------------------+------------------------+------------------------+
| regularize             | OT_BOOL                | Automatic              |
|                        |                        | regularization of      |
|                        |                        | Lagrange Hessian.      |
+------------------------+------------------------+------------------------+
| tol_du                 | OT_DOUBLE              | Stopping criterion for |
|                        |                        | dual infeasability     |
+------------------------+------------------------+------------------------+
| tol_pr                 | OT_DOUBLE              | Stopping criterion for |
|                        |                        | primal infeasibility   |
+------------------------+------------------------+------------------------+
| tol_pr_step            | OT_DOUBLE              | Stopping criterion for |
|                        |                        | the step size          |
+------------------------+------------------------+------------------------+
| tol_reg                | OT_DOUBLE              | Stopping criterion for |
|                        |                        | regularization         |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

sqpmethod
---------



A textbook SQPMethod

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| beta                   | OT_DOUBLE              | Line-search parameter, |
|                        |                        | restoration factor of  |
|                        |                        | stepsize               |
+------------------------+------------------------+------------------------+
| c1                     | OT_DOUBLE              | Armijo condition,      |
|                        |                        | coefficient of         |
|                        |                        | decrease in merit      |
+------------------------+------------------------+------------------------+
| hessian_approximation  | OT_STRING              | limited-memory|exact   |
+------------------------+------------------------+------------------------+
| lbfgs_memory           | OT_INT                 | Size of L-BFGS memory. |
+------------------------+------------------------+------------------------+
| max_iter               | OT_INT                 | Maximum number of SQP  |
|                        |                        | iterations             |
+------------------------+------------------------+------------------------+
| max_iter_ls            | OT_INT                 | Maximum number of      |
|                        |                        | linesearch iterations  |
+------------------------+------------------------+------------------------+
| merit_memory           | OT_INT                 | Size of memory to      |
|                        |                        | store history of merit |
|                        |                        | function values        |
+------------------------+------------------------+------------------------+
| min_step_size          | OT_DOUBLE              | The size (inf-norm) of |
|                        |                        | the step size should   |
|                        |                        | not become smaller     |
|                        |                        | than this.             |
+------------------------+------------------------+------------------------+
| print_header           | OT_BOOL                | Print the header with  |
|                        |                        | problem statistics     |
+------------------------+------------------------+------------------------+
| qpsol                  | OT_STRING              | The QP solver to be    |
|                        |                        | used by the SQP method |
+------------------------+------------------------+------------------------+
| qpsol_options          | OT_DICT                | Options to be passed   |
|                        |                        | to the QP solver       |
+------------------------+------------------------+------------------------+
| regularize             | OT_BOOL                | Automatic              |
|                        |                        | regularization of      |
|                        |                        | Lagrange Hessian.      |
+------------------------+------------------------+------------------------+
| tol_du                 | OT_DOUBLE              | Stopping criterion for |
|                        |                        | dual infeasability     |
+------------------------+------------------------+------------------------+
| tol_pr                 | OT_DOUBLE              | Stopping criterion for |
|                        |                        | primal infeasibility   |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



Joel Andersson

";

%feature("docstring") casadi::has_qpsol "

Check if a particular plugin is available.

";

%feature("docstring") casadi::linsol "

Create a solver for linear systems of equations Solves the linear system A*X
= B or A^T*X = B for X with A square and non-singular

If A is structurally singular, an error will be thrown during init. If A is
numerically singular, the prepare step will fail.

The usual procedure to use Linsol is: init()

set the first input (A)

prepare()

set the second input (b)

solve()

Repeat steps 4 and 5 to work with other b vectors.

The standard evaluation combines the prepare() and solve() step and may
therefore more expensive if A is invariant.

General information
===================



>Input scheme: casadi::LinsolInput (LINSOL_NUM_IN = 2) []

+-----------+-------+----------------------------------------------+
| Full name | Short |                 Description                  |
+===========+=======+==============================================+
| LINSOL_A  |       | The square matrix A: sparse, (n x n)         |
+-----------+-------+----------------------------------------------+
| LINSOL_B  |       | The right-hand-side matrix b: dense, (n x m) |
+-----------+-------+----------------------------------------------+

>Output scheme: casadi::LinsolOutput (LINSOL_NUM_OUT = 1) []

+-----------+-------+---------------------------------------------+
| Full name | Short |                 Description                 |
+===========+=======+=============================================+
| LINSOL_X  |       | Solution to the linear system of equations. |
+-----------+-------+---------------------------------------------+

List of plugins
===============



- csparsecholesky

- csparse

- lapacklu

- lapackqr

- symbolicqr

Note: some of the plugins in this list might not be available on your
system. Also, there might be extra plugins available to you that are not
listed here. You can obtain their documentation with
Linsol.doc(\"myextraplugin\")



--------------------------------------------------------------------------------

csparsecholesky
---------------



Linsol with CSparseCholesky Interface

--------------------------------------------------------------------------------





--------------------------------------------------------------------------------

csparse
-------



Linsol with CSparse Interface

--------------------------------------------------------------------------------





--------------------------------------------------------------------------------

lapacklu
--------



This class solves the linear system A.x=b by making an LU factorization of
A: A = L.U, with L lower and U upper triangular

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| allow_equilibration_fa | OT_BOOL                | Non-fatal error when   |
| ilure                  |                        | equilibration fails    |
+------------------------+------------------------+------------------------+
| equilibration          | OT_BOOL                | Equilibrate the matrix |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



--------------------------------------------------------------------------------

lapackqr
--------



This class solves the linear system A.x=b by making an QR factorization of
A: A = Q.R, with Q orthogonal and R upper triangular

--------------------------------------------------------------------------------





--------------------------------------------------------------------------------

symbolicqr
----------



Linsol based on QR factorization with sparsity pattern based reordering
without partial pivoting

>List of available options

+------------------------+------------------------+------------------------+
|           Id           |          Type          |      Description       |
+========================+========================+========================+
| codegen                | OT_BOOL                | C-code generation      |
+------------------------+------------------------+------------------------+
| compiler               | OT_STRING              | Compiler command to be |
|                        |                        | used for compiling     |
|                        |                        | generated code         |
+------------------------+------------------------+------------------------+

--------------------------------------------------------------------------------



Joel Andersson

";

%feature("docstring") casadi::userOut "";

%feature("docstring") casadi::casadi_norm_inf_mul "[INTERNAL]  Inf-norm of
a Matrix-matrix product,*

Parameters:
-----------

dwork:  A real work vector that you must allocate Minimum size: y.size1()

iwork:  A integer work vector that you must allocate Minimum size:
y.size1()+x.size2()+1

";

%feature("docstring") casadi::matrixName< SXElem > " [INTERNAL] ";

%feature("docstring") casadi::casadi_fill "[INTERNAL]  FILL: x <- alpha.

";


// File: namespaceIpopt.xml


// File: namespacestd.xml


// File: chapter1.xml


// File: chapter2.xml


// File: chapter3.xml


// File: chapter4.xml


// File: chapter5.xml


// File: chapter6.xml

