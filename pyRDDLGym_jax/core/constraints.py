"""
JAX-compatible constraints for RDDL domains.

Provides functionality to compute bounds for state, observation, and action fluents
based on action preconditions and state invariants.
"""

import jax.numpy as jnp
import numpy as np

from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.debug.exception import print_stack_trace, raise_warning


class JaxRDDLConstraints:
    """Computes bounds for RDDL fluents from constraints in JAX environments.

    This class analyzes action preconditions and state invariants to determine
    tight bounds on fluent values, which are used to construct observation and
    action spaces.
    """

    def __init__(self,
                 model: RDDLLiftedModel,
                 init_values: dict,
                 max_bound: float = jnp.inf,
                 inequality_tol: float = 1e-5,
                 vectorized: bool = True) -> None:
        """Initialize constraint analyzer.

        Args:
            model: The RDDL lifted model
            init_values: Dictionary of initial values for all fluents
            max_bound: Initial value for maximum possible bounds
            inequality_tol: Tolerance for inequality > and < comparisons
            vectorized: Whether bounds are represented as pairs of numpy arrays
                       (if True) or as pairs of scalars for grounded names (if False)
        """
        self.model = model
        self.init_values = init_values
        # Convert to float to ensure compatibility with numpy
        self.BigM = float(max_bound)
        self.epsilon = inequality_tol
        self.vectorized = vectorized

        # Initialize bounds to [-inf, +inf]
        # Use numpy arrays (mutable) during parsing, will convert later if needed
        self._bounds = {}
        for var, vtype in model.variable_types.items():
            if vtype in {'state-fluent', 'observ-fluent', 'action-fluent'}:
                # Get shape from init_values
                if var in init_values:
                    shape = jnp.shape(init_values[var])
                    if shape and shape != ():
                        self._bounds[var] = [
                            np.full(shape=shape, fill_value=-self.BigM, dtype=np.float32),
                            np.full(shape=shape, fill_value=+self.BigM, dtype=np.float32)
                        ]
                    else:
                        self._bounds[var] = [
                            np.float32(-self.BigM),
                            np.float32(+self.BigM)
                        ]

        # Parse action preconditions for bounds
        self._is_box_precond = []
        for index, precond in enumerate(model.preconditions):
            tag = f'Action precondition {index + 1}'
            is_box = self._parse_bounds(tag, precond, [], model.action_fluents)
            self._is_box_precond.append(is_box)

        # Parse state invariants for bounds
        self._is_box_invariant = []
        for index, invariant in enumerate(model.invariants):
            tag = f'State invariant {index + 1}'
            is_box = self._parse_bounds(tag, invariant, [], model.state_fluents)
            self._is_box_invariant.append(is_box)

        # Validate bounds
        for name, bounds in self._bounds.items():
            self._check_bounds(*bounds, f'Variable <{name}>', bounds)

        # Ground the bounds if not vectorized
        if self.vectorized:
            self._bounds = {name: tuple(value)
                           for name, value in self._bounds.items()}
        else:
            new_bounds = {}
            for var, (lower, upper) in self._bounds.items():
                lower = jnp.ravel(lower, order='C')
                upper = jnp.ravel(upper, order='C')
                gvars = model.variable_groundings[var]
                assert len(gvars) == len(lower) == len(upper)
                new_bounds.update(zip(gvars, zip(lower, upper)))
            self._bounds = new_bounds

    def _check_bounds(self, lower, upper, name, bounds):
        """Validate that bounds are consistent."""
        lower = jnp.asarray(lower)
        upper = jnp.asarray(upper)
        if jnp.any(lower > upper):
            raise_warning(
                f'{name} has inconsistent bounds {bounds}, '
                f'lower bounds must be <= upper bounds.',
                'red'
            )

    def _parse_bounds(self, tag, expr, objects, search_vars):
        """Parse expression to extract bounds on fluents.

        Args:
            tag: Description for error messages
            expr: Expression to parse
            objects: List of bound objects (from forall)
            search_vars: Set of fluents to search for

        Returns:
            bool: True if successfully parsed as box constraint
        """
        etype, op = expr.etype

        # Handle forall aggregation
        if etype == 'aggregation' and op == 'forall':
            *pvars, arg = expr.args
            new_objects = objects + [pvar for _, pvar in pvars]
            return self._parse_bounds(tag, arg, new_objects, search_vars)

        # Handle conjunction (and)
        elif etype == 'boolean' and op == '^':
            success = True
            for arg in expr.args:
                success_arg = self._parse_bounds(tag, arg, objects, search_vars)
                success = success and success_arg
            return success

        # Handle relational constraints
        elif etype == 'relational':
            var, lim, loc, slices = self._parse_bounds_relational(
                tag, expr, objects, search_vars)
            success = var is not None and loc is not None
            if success:
                op = np.minimum if loc == 1 else np.maximum
                if slices:
                    self._bounds[var][loc][slices] = op(
                        self._bounds[var][loc][slices], lim)
                else:
                    self._bounds[var][loc] = op(self._bounds[var][loc], lim)
            return success

        # Cannot parse as box constraint
        else:
            return False

    def _parse_bounds_relational(self, tag, expr, objects, search_vars):
        """Parse relational expression to extract bound.

        Returns:
            tuple: (var_name, limit_value, location, slices)
        """
        left, right = expr.args
        _, op = expr.etype

        left_pvar = left.is_pvariable_expression() and left.args[0] in search_vars
        right_pvar = right.is_pvariable_expression() and right.args[0] in search_vars

        # Both sides are pvariables or unsupported operator
        if (left_pvar and right_pvar) or op not in {'<=', '<', '>=', '>'}:
            raise_warning(
                f'{tag} does not have a structure of '
                f'<action or state fluent> <op> <rhs>, where '
                f'<op> is one of {{<=, <, >=, >}} and '
                f'<rhs> is a deterministic function of non-fluents only, '
                f'and will be ignored.\n' +
                print_stack_trace(expr), 'red')
            return None, 0.0, None, []

        # Neither side is pvariable
        elif not left_pvar and not right_pvar:
            return None, 0.0, None, []

        # One side is pvariable, other is constant
        else:
            if left_pvar:
                var, args = left.args
                const_expr = right
            else:
                var, args = right.args
                const_expr = left

            if args is None:
                args = []

            # Check if RHS is non-fluent
            if not self.model.is_non_fluent_expression(const_expr):
                raise_warning(
                    f'{tag} contains a fluent expression '
                    f'(a nondeterministic operation or fluent variable) '
                    f'on both sides of an (in)equality, and will be ignored.\n' +
                    print_stack_trace(const_expr), 'red')
                return None, 0.0, None, []

            # Evaluate constant expression
            # For JAX, we evaluate using init_values as substitution
            const = self._evaluate_constant_expr(const_expr)
            eps, loc = self._get_op_code(op, left_pvar)
            lim = const + eps

            # Construct slices for indexed access
            slices = []
            for arg in args:
                if self.model.is_literal(arg):
                    arg = self.model.strip_literal(arg)
                    slices.append(self.model.object_to_index[arg])
                else:
                    slices.append(slice(None))
            slices = tuple(slices) if slices else []

            return var, lim, loc, slices

    def _evaluate_constant_expr(self, expr):
        """Evaluate a constant (non-fluent) expression.

        This is a simplified evaluator for constant expressions.
        For complex expressions, returns conservative bounds.
        """
        etype, op = expr.etype

        # Literal value
        if etype == 'constant':
            return expr.args

        # Arithmetic operations
        elif etype == 'arithmetic':
            if op == '+':
                return sum(self._evaluate_constant_expr(arg) for arg in expr.args)
            elif op == '-':
                if len(expr.args) == 1:
                    # Unary negation
                    return -self._evaluate_constant_expr(expr.args[0])
                else:
                    # Binary subtraction
                    left, right = expr.args
                    return self._evaluate_constant_expr(left) - self._evaluate_constant_expr(right)
            elif op == '*':
                result = 1.0
                for arg in expr.args:
                    result *= self._evaluate_constant_expr(arg)
                return result
            elif op == '/':
                left, right = expr.args
                return self._evaluate_constant_expr(left) / self._evaluate_constant_expr(right)

        # Non-fluent variable
        elif etype == 'pvar' and expr.args[0] in self.model.non_fluents:
            var_name = expr.args[0]
            args = expr.args[1] if len(expr.args) > 1 else None

            if var_name in self.init_values:
                value = self.init_values[var_name]
                if args:
                    # Indexed access
                    indices = []
                    for arg in args:
                        if self.model.is_literal(arg):
                            obj = self.model.strip_literal(arg)
                            indices.append(self.model.object_to_index[obj])
                        else:
                            return 0.0  # Conservative default
                    return value[tuple(indices)]
                return value

        # Default: return conservative value
        return 0.0

    def _get_op_code(self, op, is_right):
        """Get epsilon and location for relational operator.

        Args:
            op: Relational operator (<=, <, >=, >)
            is_right: Whether variable is on right side

        Returns:
            tuple: (epsilon, location) where location is 0 for lower, 1 for upper
        """
        eps = 0.0
        if is_right:
            if op in ['<=', '<']:
                loc = 1
                if op == '<':
                    eps = -self.epsilon
            elif op in ['>=', '>']:
                loc = 0
                if op == '>':
                    eps = self.epsilon
        else:
            if op in ['<=', '<']:
                loc = 0
                if op == '<':
                    eps = self.epsilon
            elif op in ['>=', '>']:
                loc = 1
                if op == '>':
                    eps = -self.epsilon
        return eps, loc

    @property
    def bounds(self):
        """Get computed bounds dictionary."""
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        """Set bounds dictionary."""
        self._bounds = value

    @property
    def is_box_preconditions(self):
        """Get list indicating which preconditions are box constraints."""
        return self._is_box_precond

    @property
    def is_box_invariants(self):
        """Get list indicating which invariants are box constraints."""
        return self._is_box_invariant
