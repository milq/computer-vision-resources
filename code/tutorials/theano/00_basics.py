# Tutorial from: https://github.com/craffel/theano-tutorial.

import numpy as np
import theano
import theano.tensor as T  # By convention, the tensor submodule is loaded as T

"""
SYMBOLIC VARIABLES

In Theano, all algorithms are defined symbolically. It's more like writing
out math than writing code. The following Theano variables are symbolic; they
don't have an explicit value.
"""
print('SYMBOLIC VARIABLES:')
# The 'theano.tensor' submodule has various primitive symbolic variable types.
# Here, we're defining a scalar (0-dimensional) variable.
# The argument gives the variable its name.
foo = T.scalar('foo')
# Now, we can define another variable 'bar' which is just 'foo' squared.
bar = foo**2
# It will also be a theano variable.
print(type(bar))
print(bar.type)
# Using theano's 'pp' (pretty print) function, we see that
# 'bar' is defined symbolically as the square of 'foo'.
print(theano.pp(bar))

"""
SYMBOLIC FUNCTIONS

To actually compute things with Theano, you define symbolic functions, which
can then be called with actual values to retrieve an actual value.
"""
# We can't compute anything with 'foo' and 'bar' yet.
# We need to define a theano function first.
# The first argument of 'theano.function' defines the inputs to the function.
# Note that 'bar' relies on 'foo', so 'foo' is an input to this function.
# The submodule 'theano.function' will compile code for computing values of
# 'bar' given values of 'foo'.
print('\nSYMBOLIC FUNCTIONS:')
f = theano.function([foo], bar)
print(f(3))

# Alternatively, in some cases you can use a symbolic variable's 'eval' method.
# This can be more convenient than defining a function.
# The 'eval' method takes a dictionary where the keys are Theano variables and
# the values are values for those variables.
print(bar.eval({foo: 3}))


def square(x):   # We can also use functions to construct Theano variables.
    return x**2  # It can make syntax cleaner for more complicated examples

bar = square(foo)
print(bar.eval({foo: 3}))

"""
TENSOR

Theano also has variable types for vectors, matrices, and tensors. The
theano.tensor submodule has various functions for performing operations on
these variables.
"""
print('\nTENSOR:')
A = T.matrix('A')
x = T.vector('x')
b = T.vector('b')
y = T.dot(A, x) + b
z = T.sum(A**2)  # Note that squaring a matrix is element-wise

# The submodule 'theano.function' can compute multiple things at a time. You
# can also set default parameter values.
b_default = np.array([0, 0], dtype=theano.config.floatX)
linear_mix = theano.function([A, x, theano.Param(b, default=b_default)],
                             [y, z])
# Supplying values for 'A', 'x', and 'b'.
print(linear_mix(np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=theano.config.floatX),  # 'A'.
                 np.array([1, 2, 3], dtype=theano.config.floatX),    # 'x'.
                 np.array([4, 5], dtype=theano.config.floatX)))      # 'b'.
# Using the default value for 'b'.
print(linear_mix(np.array([[1, 2, 3],
                           [4, 5, 6]]),  # 'A'.
                 np.array([1, 2, 3])))   # 'x'.

"""
SHARED VARIABLES

Shared variables are a little different - they actually do have an explicit
value, which can be get/set and is shared across functions which use the
variable. They're also useful because they have state across function calls.
"""
print('\nSHARED VARIABLES:')
shared_var = theano.shared(np.array([[1, 2], [3, 4]],
                                    dtype=theano.config.floatX))

# The type of the shared variable is deduced from its initialization
print(shared_var.type())

# We can set the value of a shared variable using 'set_value'.
shared_var.set_value(np.array([[3, 4], [2, 1]], dtype=theano.config.floatX))
# ..and get it using 'get_value'.
print(shared_var.get_value())

shared_squared = shared_var**2
# The first argument of 'theano.function' (inputs) tells Theano what the
# arguments to the compiled function should be. Note that because 'shared_var'
# is shared, it already has a value, so it doesn't need to be an input to the
# function. Therefore, Theano implicitly considers 'shared_var' an input to a
# function using 'shared_squared' and so we don't need to include it in the
# inputs argument of 'theano.function'.
function_1 = theano.function([], shared_squared)
print(function_1())

"""
UPDATES

The value of a shared variable can be updated in a function by using the
updates argument of 'theano.function'.
"""
print('\nUPDATES:')
# We can also update the state of a shared variable in a function.
subtract = T.matrix('subtract')
# Updates takes a dictionary where keys are shared variables and values are
# the new value the shared variable should take.
# Here, updates will set 'shared_var = shared_var - subtract'.
function_2 = theano.function([subtract], shared_var,
                             updates={shared_var: shared_var - subtract})
print("shared_var before subtracting [[1, 1], [1, 1]] using function_2:")
print(shared_var.get_value())
# Subtract '[[1, 1], [1, 1]]' from 'shared_var'.
function_2(np.array([[1, 1], [1, 1]]))
print("shared_var after calling function_2:")
print(shared_var.get_value())
# This also changes the output of 'function_1', because 'shared_var' is shared!
print("New output of function_1() (shared_var**2):")
print(function_1())

"""
GRADIENTS

A pretty huge benefit of using Theano is its ability to compute gradients. This
allows you to symbolically define a function and quickly
compute its (numerical) derivative without actually deriving the derivative.
"""
print('\nGRADIENTS:')
# Recall that 'bar = foo**2'.
# We can compute the gradient of 'bar' with respect to 'foo' like so:
bar_grad = T.grad(bar, foo)
# We expect that 'bar_grad = 2*foo'
print(bar_grad.eval({foo: 10}))

# Recall that 'y = Ax + b'.
# We can also compute a Jacobian like so:
y_J = theano.gradient.jacobian(y, x)
linear_mix_J = theano.function([A, x, b], y_J)
# Because it's a linear mixed model, we expect the output to always be 'A':
print(linear_mix_J(np.array([[9, 8, 7], [4, 5, 6]]),  # 'A'.
                   np.array([1, 2, 3]),               # 'x'.
                   np.array([4, 5])))                 # 'b'.
# We could also compute the Hessian with 'theano.gradient.hessian'.

"""
DEBUGGING

Debugging in Theano can be a little tough because the code which is actually
being run is pretty far removed from the code you wrote. One simple way to
sanity check your Theano expressions before actually compiling any functions
is to use test values.
"""
print('\nDEBUGGING:\nUncomment the three lines in the code to see the errors.')
# Let's create another matrix, 'B'.
B = T.matrix('B')
# And, a symbolic variable which is just 'A' (from above) dotted against 'B'.
# At this point, Theano doesn't know the shape of 'A' or 'B', so there's no way
# for it to know whether 'A dot B' is valid.
C = T.dot(A, B)
# Now, let's try to use it:
# C.eval({A: np.zeros((3, 4)), B: np.zeros((5, 6))})  # UNCOMMENT THIS LINE.
# The above error message is a little opaque (and it would be even worse had we
# not given the Theano variables 'A' and 'B' names). Errors like this can be
# particularly confusing when the Theano expression being computed is very
# complex. They also won't ever tell you the line number in your Python code
# where 'A' dot B' was computed, because the actual code being run is not your
# Python code, it's the compiled Theano code! Fortunately, 'test values' let us
# get around this issue. However, not all Theano methods (for example, and
# significantly, 'scan') allow for test values.

# This tells Theano we're going to use test values, and to warn when there's an
# error with them. The setting 'warn' means 'warn me when I haven't supplied a
# test value'.
theano.config.compute_test_value = 'warn'
# Setting the 'tag.test_value' attribute gives the variable its test value.
A.tag.test_value = np.random.random((3, 4))
B.tag.test_value = np.random.random((5, 6))
# Now, we get an error when we compute 'C' which points us to the correct line!
# C = T.dot(A, B)  # UNCOMMENT THIS LINE.

# We won't be using test values for the rest of the tutorial.
theano.config.compute_test_value = 'off'
# Another place where debugging is useful is when an invalid calculation is
# done, e.g. one which results in 'nan'. By default, Theano will silently allow
# these 'nan' values to be computed and used, but this silence can be
# catastrophic to the rest of your Theano computation. At the cost of speed, we
# can instead have Theano compile functions in 'DebugMode', where an invalid
# computation causes an error.

# A simple division function.
num = T.scalar('num')
den = T.scalar('den')
divide = theano.function([num, den], num/den)
print(divide(10, 2))
# This will cause a 'NaN'
print(divide(0, 0))
# To compile a function in debug mode, just set mode='DebugMode'.
divide = theano.function([num, den], num/den, mode='DebugMode')
# NaNs now cause errors
# print(divide(0, 0))  # UNCOMMENT THIS LINE.

"""
USING THE CPU VS GPU

Theano can transparently compile onto different hardware. What device it uses
by default depends on your '.theanorc' file and any environment variables
defined. Currently, you should use 'float32' when using most GPUs, but most
people prefer to use 'float64' on a CPU. For convenience, Theano provides the
'floatX' configuration variable which designates what float accuracy to use.
For example, you can run a Python script with certain environment variables set
to use the CPU: 'THEANO_FLAGS=device=cpu,floatX=float64 python your_script.py'.
To set the GPU: 'THEANO_FLAGS=device=gpu,floatX=float32 python your_script.py'.
"""
print('\nUSING THE CPU VS GPU')
# You can get the values being used to configure Theano like so:
print(theano.config.device)
print(theano.config.floatX)
# You can also get/set them at runtime:
old_floatX = theano.config.floatX
theano.config.floatX = 'float32'
# Be careful that you're actually using 'floatX'!
# For example, the following will cause 'var' to be a 'float64' regardless of
# 'floatX' due to numpy defaults:
var = theano.shared(np.array([1.3, 2.4]))
print(var.type())
# So, whenever you use a 'numpy' array, make sure to set its 'dtype' to
# 'theano.config.floatX'.
var = theano.shared(np.array([1.3, 2.4], dtype=theano.config.floatX))
print(var.type())
# Revert to old value.
theano.config.floatX = old_floatX
