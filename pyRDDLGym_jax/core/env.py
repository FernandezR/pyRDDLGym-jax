# ***********************************************************************
# Pure JAX Environment for pyRDDLGym
#
# A JAX-native, JIT-compilable environment interface for RDDL domains
# that supports vectorization, batching, and efficient gradient-based
# planning and reinforcement learning.
#
# Key features:
# - Fully JIT-compilable reset() and step() functions
# - No side effects (no logging, no visualization during rollouts)
# - Support for vmap to run multiple parallel environments
# - Pure functional interface using JAX pytrees
# - Static shapes for maximum performance
#
# ***********************************************************************

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.parser.reader import RDDLReader

from pyRDDLGym_jax.core.compiler import JaxRDDLCompiler
from pyRDDLGym_jax.core.constraints import JaxRDDLConstraints
from pyRDDLGym_jax.core import spaces
from flax.struct import dataclass

# Type aliases for clarity
Action = Dict[str, jnp.ndarray]
Observation = Dict[str, jnp.ndarray]
State = Dict[str, jnp.ndarray]
PRNGKey = jnp.ndarray
Params = Any  # Model parameters (pytree)

@dataclass
class EnvState:
    """State of the environment (all fields needed for stepping).

    This is a JAX pytree that contains all the information needed
    to step the environment forward. It's designed to be immutable
    and purely functional.

    Attributes:
        obs: Current observation (may differ from state in POMDPs)
        state: Current internal state of the environment
        subs: Current substitution dictionary (internal state variables)
        key: PRNG key for stochastic sampling
        model_params: Model parameters for stateful operations
        timestep: Current timestep in the episode
        done: Whether the episode has terminated
        reward: Cumulative reward (optional, for tracking)
    """
    obs: Observation
    state: State
    subs: Dict[str, jnp.ndarray]
    key: PRNGKey
    model_params: Params
    timestep: jnp.ndarray
    done: jnp.ndarray
    reward: jnp.ndarray

@dataclass
class TimeStep:
    """A single timestep returned by the environment.

    This follows the dm_env convention but adapted for JAX.

    Attributes:
        observation: The observation at this timestep
        reward: The reward received
        done: Whether the episode terminated naturally (terminal state)
        truncated: Whether the episode was truncated (time limit)
        info: Additional information (currently empty dict)
    """
    observation: Observation
    reward: jnp.ndarray
    done: jnp.ndarray
    truncated: jnp.ndarray
    info: Dict[str, Any]


class JaxRDDLEnv:
    """A pure JAX environment for RDDL domains.

    This class provides a JIT-compilable, functional interface to RDDL
    environments. Unlike the standard RDDLEnv which inherits from gym.Env,
    this environment:

    1. Uses pure functions for reset() and step()
    2. Maintains state explicitly in EnvState objects
    3. Supports vmap for parallel environments
    4. Has no side effects (no logging, visualization during rollouts)
    5. Is fully differentiable for gradient-based planning
    6. Provides action precondition checking (optional, not enforced by default)

    Example usage:
        ```python
        # Create environment
        env = JaxRDDLEnv("domain.rddl", "instance.rddl")

        # Initialize
        key = jax.random.PRNGKey(42)
        env_state, timestep = env.reset(key)

        # Step the environment
        actions = env.sample_random_action(key, env_state)
        env_state, timestep = env.step(env_state, actions)

        # Check if actions satisfy preconditions (optional)
        is_valid = env.check_action_preconditions(env_state, actions)

        # Vectorized rollout (multiple parallel environments)
        keys = jax.random.split(key, num_envs)
        vmapped_reset = jax.vmap(env.reset)
        env_states, timesteps = vmapped_reset(keys)
        ```

    Note:
        For visualization and logging, use the wrapper classes or
        evaluate policies outside of JIT-compiled loops.

        Action preconditions are compiled and can be checked using
        check_action_preconditions(), but are NOT automatically enforced
        during step() for performance reasons. If you need enforcement,
        check preconditions before calling step().
    """

    def __init__(self,
                 domain: Union[str, RDDLLiftedModel],
                 instance: Optional[str] = None,
                 logger: Optional[Any] = None,
                 vectorized: bool = True,
                 python_functions: Optional[Dict[str, Callable]] = None,
                 **compiler_kwargs) -> None:
        """Initialize the JAX environment.

        Args:
            domain: Either a path to RDDL domain file or a RDDLLiftedModel
            instance: Path to RDDL instance file (if domain is a path)
            logger: Optional logger for compilation information
            vectorized: Whether to use vectorized (array) representation for actions/observations.
                       If True (default), actions and observations are JAX arrays.
                       If False, actions and observations are dictionaries of scalars (grounded).
            python_functions: External Python functions callable from RDDL
            **compiler_kwargs: Additional arguments for the JAX compiler
        """
        # Store vectorization flag
        self.vectorized = vectorized
        # Parse and compile the RDDL model
        if isinstance(domain, RDDLLiftedModel):
            self.model = domain
        else:
            reader = RDDLReader(domain, instance)
            domain_text = reader.rddltxt
            parser = RDDLParser(lexer=None, verbose=False)
            parser.build()
            rddl = parser.parse(domain_text)
            self.model = RDDLLiftedModel(rddl)

        # Store basic environment info
        self.horizon = self.model.horizon
        self.discount = self.model.discount
        self.max_allowed_actions = self.model.max_allowed_actions
        self._is_pomdp = bool(self.model.observ_fluents)

        # Compile to JAX
        self.compiler = JaxRDDLCompiler(
            self.model,
            logger=logger,
            python_functions=python_functions or {},
            **compiler_kwargs
        )
        self.compiler.compile(log_jax_expr=True, heading='JAX ENVIRONMENT MODEL')

        # Store compiled components
        self.init_values = self.compiler.init_values
        self.levels = self.compiler.levels
        self.traced = self.compiler.traced
        self.model_params = self.compiler.model_params

        # JIT compile the core functions
        self._jit_reward = jax.jit(self.compiler.reward)
        self._jit_cpfs = {cpf: jax.jit(expr)
                          for cpf, expr in self.compiler.cpfs.items()}
        self._jit_terminals = [jax.jit(term)
                               for term in self.compiler.terminations]
        self._jit_invariants = [jax.jit(inv)
                                for inv in self.compiler.invariants]
        self._jit_preconditions = [jax.jit(precond)
                                   for precond in self.compiler.preconditions]

        # Store noop actions (vectorized form)
        self.noop_actions = {
            var: values for (var, values) in self.init_values.items()
            if self.model.variable_types[var] == 'action-fluent'
        }

        # Get action and state fluent names
        self.action_fluents = list(self.model.action_fluents)
        self.state_fluents = list(self.model.state_fluents)
        if self._is_pomdp:
            self.observ_fluents = list(self.model.observ_fluents)

        # Compute bounds and shapes using JaxRDDLConstraints
        constraints = JaxRDDLConstraints(
            self.model,
            self.init_values,
            vectorized=self.vectorized
        )
        self._bounds = constraints.bounds
        self._shapes = {var: np.shape(values[0])
                        for (var, values) in self._bounds.items()}

        # Build observation space
        if self._is_pomdp:
            obs_ranges = self.model.observ_ranges
        else:
            obs_ranges = self.model.state_ranges
        if not self.vectorized:
            obs_ranges = self.model.ground_vars_with_value(obs_ranges)
        self.observation_space = self._build_space(obs_ranges, is_action=False)

        # Build action space
        action_ranges = self.model.action_ranges
        if not self.vectorized:
            action_ranges = self.model.ground_vars_with_value(action_ranges)
        self.action_space = self._build_space(action_ranges, is_action=True)

        # Pre-compile the step and reset functions for maximum performance
        self._jit_step = jax.jit(self._step_impl)
        self._jit_reset = jax.jit(self._reset_impl)

        # Pre-compute action bounds for JIT-compatible get_available_actions
        self._action_bounds = {}
        for action_name, space in self.action_space.items():
            if isinstance(space, spaces.Box) and space.dtype in [jnp.int32, jnp.int64]:
                # Extract bounds as concrete Python ints (done at init time, outside JIT)
                low_val = space.low.flatten()[0] if hasattr(space.low, 'flatten') else space.low
                high_val = space.high.flatten()[0] if hasattr(space.high, 'flatten') else space.high
                # Convert to Python int to ensure it's concrete
                low = int(np.asarray(low_val))
                high = int(np.asarray(high_val))
                shape = space.shape
                num_agents = shape[0] if len(shape) > 0 else 1
                self._action_bounds[action_name] = {
                    'low': low,
                    'high': high,
                    'num_agents': num_agents,
                    'num_values': high - low + 1
                }
            elif isinstance(space, spaces.Discrete):
                self._action_bounds[action_name] = {
                    'low': 0,
                    'high': space.n - 1,
                    'num_agents': 1,
                    'num_values': space.n
                }

        # Visualizer (not used during JIT-compiled rollouts, only for rendering outside JIT)
        self._visualizer = None
        self._movie_generator = None
        self._movie_per_episode = False
        self._movies = 0

    @property
    def is_pomdp(self) -> bool:
        """Whether this environment is a POMDP (has observations)."""
        return self._is_pomdp

    def _build_space(self, ranges: Dict[str, str], is_action: bool) -> spaces.Dict:
        """Build a JAX space from RDDL variable ranges.

        Args:
            ranges: Dictionary mapping variable names to their types/ranges
            is_action: Whether building action space (affects vectorization logic)

        Returns:
            JAX Dict space containing all variables with appropriate bounds
        """
        result = {}

        for var, prange in ranges.items():
            # Get the shape from _shapes
            shape = self._shapes[var]

            # Enumerated type (object type)
            if prange in self.model.type_to_objects:
                num_objects = len(self.model.type_to_objects[prange])
                if self.vectorized:
                    # Vectorized: use Box for array of discrete values
                    result[var] = spaces.Box(
                        low=0,
                        high=num_objects - 1,
                        shape=shape,
                        dtype=jnp.int32
                    )
                else:
                    # Grounded: use Discrete for scalar value
                    result[var] = spaces.Discrete(num_objects, dtype=jnp.int32)

            # Real values
            elif prange == 'real':
                # Get bounds from constraints
                low, high = self._bounds[var]
                result[var] = spaces.Box(
                    low=low,
                    high=high,
                    shape=shape,
                    dtype=jnp.float32
                )

            # Boolean values
            elif prange == 'bool':
                if self.vectorized:
                    # Vectorized: use Box for array of booleans (0/1)
                    result[var] = spaces.Box(
                        low=0,
                        high=1,
                        shape=shape,
                        dtype=jnp.int32
                    )
                else:
                    # Grounded: use Discrete for scalar boolean
                    result[var] = spaces.Discrete(2, dtype=jnp.int32)

            # Integer values
            elif prange == 'int':
                # Get bounds from constraints
                low, high = self._bounds[var]
                # Clip to int32 range
                low = jnp.maximum(low, jnp.iinfo(jnp.int32).min)
                high = jnp.minimum(high, jnp.iinfo(jnp.int32).max)

                if self.vectorized:
                    result[var] = spaces.Box(
                        low=low,
                        high=high,
                        shape=shape,
                        dtype=jnp.int32
                    )
                else:
                    # For grounded, use a large discrete space
                    result[var] = spaces.Discrete(
                        int(high - low + 1),
                        dtype=jnp.int32
                    )
            else:
                raise ValueError(
                    f'Range <{prange}> of variable <{var}> is not valid, '
                    f'must be an enumerated or primitive type.'
                )

        return spaces.Dict(result)

    def _flatten_space(self, dict_space: spaces.Dict) -> Tuple[spaces.Box, Callable, Callable]:
        """Flatten a Dict space into a 1D vector Box space.

        This is useful for neural networks that expect flat input/output vectors
        instead of structured dictionaries.

        Args:
            dict_space: Dictionary space to flatten

        Returns:
            flat_space: 1D Box space representing the flattened space
        """
        # Collect information about each variable
        var_infos = []
        total_size = 0

        for var_name in sorted(dict_space.keys()):
            space = dict_space[var_name]
            size = int(jnp.prod(jnp.array(space.shape)))
            var_infos.append({
                'name': var_name,
                'shape': space.shape,
                'dtype': space.dtype,
                'size': size,
                'start': total_size,
                'end': total_size + size,
                'low': space.low if isinstance(space, spaces.Box) else 0,
                'high': space.high if isinstance(space, spaces.Box) else (space.n - 1 if isinstance(space, spaces.Discrete) else 1)
            })
            total_size += size

        # Determine bounds for the flat space
        flat_low = jnp.full(total_size, -1e10, dtype=jnp.float32)
        flat_high = jnp.full(total_size, 1e10, dtype=jnp.float32)

        for info in var_infos:
            start, end = info['start'], info['end']
            low, high = info['low'], info['high']

            # Handle scalar bounds
            if jnp.isscalar(low) or (isinstance(low, jnp.ndarray) and low.shape == ()):
                flat_low = flat_low.at[start:end].set(float(low))
            else:
                flat_low = flat_low.at[start:end].set(jnp.ravel(low).astype(jnp.float32))

            if jnp.isscalar(high) or (isinstance(high, jnp.ndarray) and high.shape == ()):
                flat_high = flat_high.at[start:end].set(float(high))
            else:
                flat_high = flat_high.at[start:end].set(jnp.ravel(high).astype(jnp.float32))

        # Create flat space
        flat_space = spaces.Box(
            low=flat_low,
            high=flat_high,
            shape=(total_size,),
            dtype=jnp.float32
        )

        return flat_space

    def flatten_observation_space(self) -> Tuple[spaces.Box, Callable, Callable]:
        """Get flattened observation space with flatten/unflatten functions.

        Returns:
            flat_space: 1D Box space for observations
            flatten_fn: Function to flatten observations
            unflatten_fn: Function to unflatten observations
        """
        return self._flatten_space(self.observation_space)

    def flatten_action_space(self) -> Tuple[spaces.Box, Callable, Callable]:
        """Get flattened action space with flatten/unflatten functions.

        Returns:
            flat_space: 1D Box space for actions
            flatten_fn: Function to flatten actions
            unflatten_fn: Function to unflatten actions
        """
        return self._flatten_space(self.action_space)

    def reset(self, key: PRNGKey) -> Tuple[EnvState, TimeStep]:
        """Reset the environment to initial state.

        This is a pure function that returns a new environment state.
        It can be JIT-compiled and vmapped for parallel environments.

        Args:
            key: JAX PRNG key for stochastic initialization

        Returns:
            env_state: Initial environment state
            timestep: Initial timestep (with initial observation)
        """
        return self._jit_reset(key)

    def _reset_impl(self, key: PRNGKey) -> Tuple[EnvState, TimeStep]:
        """Internal implementation of reset (to be JIT-compiled)."""
        # Initialize substitution dict with initial values
        subs = self.init_values.copy()

        # Extract state
        state = {
            var: subs[var] for var in self.state_fluents
        }

        # Extract observation (for POMDPs) or use state (for MDPs)
        if self._is_pomdp:
            obs = {var: subs[var] for var in self.observ_fluents}
        else:
            obs = state

        # Check if initially terminal
        done = self._check_terminals(subs, self.model_params, key)

        # Initialize reward (scalar or vector depending on reward-vector requirement)
        # We need to check the reward structure from a dummy evaluation
        dummy_reward, _, _, _ = self._jit_reward(subs, self.model_params, key)
        initial_reward = jnp.zeros_like(dummy_reward)

        # Create environment state
        env_state = EnvState(
            obs=obs,
            state=state,
            subs=subs,
            key=key,
            model_params=self.model_params,
            timestep=jnp.array(0, dtype=jnp.int32),
            done=done,
            reward=initial_reward
        )

        # Create initial timestep
        timestep = TimeStep(
            observation=obs,
            reward=initial_reward,
            done=done,
            truncated=jnp.array(False, dtype=bool),
            info={}
        )

        return env_state, timestep

    def step(self, env_state: EnvState, actions: Action) -> Tuple[EnvState, TimeStep]:
        """Step the environment forward one timestep.

        This is a pure function that takes the current state and actions,
        and returns the next state and timestep. It can be JIT-compiled
        and vmapped for parallel environments.

        Args:
            env_state: Current environment state
            actions: Action dictionary in vectorized format (use prepare_actions_for_sim()
                    to convert grounded actions to vectorized format before calling step)

        Returns:
            env_state: Next environment state
            timestep: Timestep information (observation, reward, done, etc.)

        Note:
            Actions must be in vectorized format. If you have grounded actions
            (e.g., {'move___a1': 0, 'move___a2': 2}), call prepare_actions_for_sim()
            first to convert them to vectorized format (e.g., {'move': [0, 2]}).
        """
        return self._jit_step(env_state, actions)

    def _step_impl(self, env_state: EnvState, actions: Action) -> Tuple[EnvState, TimeStep]:
        """Internal implementation of step (to be JIT-compiled)."""
        subs = env_state.subs.copy()
        subs.update(actions)

        key = env_state.key
        model_params = env_state.model_params

        # Evaluate CPFs in topological order
        for cpfs_at_level in self.levels.values():
            for cpf in cpfs_at_level:
                cpf_fn = self._jit_cpfs[cpf]
                value, key, error, model_params = cpf_fn(subs, model_params, key)
                subs[cpf] = value

        # Compute reward
        reward, key, error, model_params = self._jit_reward(subs, model_params, key)

        # Update state variables (state' -> state)
        for state_var, next_state_var in self.model.next_state.items():
            subs[state_var] = subs[next_state_var]

        # Extract new state
        state = {var: subs[var] for var in self.state_fluents}

        # Extract observation
        if self._is_pomdp:
            obs = {var: subs[var] for var in self.observ_fluents}
        else:
            obs = state

        # Check termination conditions
        done = self._check_terminals(subs, model_params, key)

        # Check truncation (horizon limit)
        new_timestep = env_state.timestep + 1
        truncated = new_timestep >= self.horizon

        # Create new environment state
        new_env_state = EnvState(
            obs=obs,
            state=state,
            subs=subs,
            key=key,
            model_params=model_params,
            timestep=new_timestep,
            done=done | truncated,
            reward=env_state.reward + reward
        )

        # Create timestep
        timestep = TimeStep(
            observation=obs,
            reward=reward,
            done=done,
            truncated=truncated,
            info={}
        )

        return new_env_state, timestep

    def _check_terminals(self, subs: Dict, model_params: Params, key: PRNGKey) -> jnp.ndarray:
        """Check if any termination condition is satisfied."""
        is_terminal = jnp.array(False, dtype=bool)
        for terminal_fn in self._jit_terminals:
            result, key, error, model_params = terminal_fn(subs, model_params, key)
            is_terminal = is_terminal | result
        return is_terminal

    def _check_preconditions(self, subs: Dict, model_params: Params, key: PRNGKey) -> jnp.ndarray:
        """Check if all action preconditions are satisfied.

        Returns True if all preconditions are satisfied, False otherwise.
        """
        all_satisfied = jnp.array(True, dtype=bool)
        for precond_fn in self._jit_preconditions:
            result, key, error, model_params = precond_fn(subs, model_params, key)
            all_satisfied = all_satisfied & result
        return all_satisfied

    def check_action_preconditions(self, env_state: EnvState, actions: Action) -> bool:
        """Check if the given actions satisfy all preconditions.

        This is a utility function that can be used to validate actions
        before stepping the environment. Note that this creates a copy
        of the substitution dict and is not automatically called during step().

        Args:
            env_state: Current environment state
            actions: Actions to validate

        Returns:
            satisfied: True if all preconditions are satisfied
        """
        subs = env_state.subs.copy()
        subs.update(actions)
        return self._check_preconditions(subs, env_state.model_params, env_state.key)

    def get_available_actions(self, env_state: EnvState) -> Dict[str, Any]:
        """Get available (valid) actions at the current state.

        Returns a dictionary mapping action names to binary masks indicating
        which action values satisfy preconditions.

        **This method is JIT-compatible** and can be used within JIT-compiled functions.

        Args:
            env_state: Current environment state

        Returns:
            Dictionary mapping action names to validity masks:
            - For vectorized: {'action_name': jnp.array([[agent0_mask], [agent1_mask], ...])}
              where each mask is an array of 0/1 indicating validity of each action value
            - For non-vectorized: {'action_name': {'action_name___agent': jnp.array([mask])}}
              where mask is an array of 0/1 for each possible action value

        Example:
            ```python
            # Vectorized environment with 2 agents, 5 possible actions each
            masks = env.get_available_actions(env_state)
            # Returns: {'move': jnp.array([[1,0,1,0,1], [1,1,0,1,1]])}

            # Non-vectorized environment
            masks = env.get_available_actions(env_state)
            # Returns: {'move': {'move___a1': jnp.array([1,0,1,0,1]), 'move___a2': jnp.array([1,1,0,0,1])}}

            # Can be JIT compiled
            jit_get_actions = jax.jit(env.get_available_actions)
            masks = jit_get_actions(env_state)
            ```

        Note:
            For non-vectorized mode, this returns a nested dict structure which is not
            ideal for JIT but still works. For best JIT performance, use vectorized mode.
        """
        if not self.vectorized:
            # Non-vectorized: check each grounded action independently
            result = {}

            for action_name, space in self.action_space.items():
                # Parse base action name from grounded name (e.g., 'move___a1' -> 'move')
                base_name = action_name.split('___')[0] if '___' in action_name else action_name

                if base_name not in result:
                    result[base_name] = {}

                # Get bounds from pre-computed values
                if action_name not in self._action_bounds:
                    continue

                bounds = self._action_bounds[action_name]
                low = bounds['low']
                high = bounds['high']
                possible_values = list(range(low, high + 1))

                valid_mask = []

                # Test each action value for this grounded action
                for value in possible_values:
                    # Create test action with this value for this grounded action
                    test_action = {action_name: jnp.array(value, dtype=jnp.int32)}

                    # Prepare actions (convert to vectorized format)
                    test_action_vec = self.prepare_actions_for_sim(test_action)

                    # Check if preconditions are satisfied
                    is_valid = self.check_action_preconditions(env_state, test_action_vec)
                    valid_mask.append(1 if is_valid else 0)

                result[base_name][action_name] = jnp.array(valid_mask, dtype=jnp.int32)

            return result

        # Vectorized: check valid actions per agent using vmap (JIT-compatible)
        return self._get_available_actions_vectorized(env_state)

    def _get_available_actions_vectorized(self, env_state: EnvState) -> Dict[str, jnp.ndarray]:
        """JIT-compilable version for vectorized environments.

        This internal method uses pre-computed bounds and pure JAX operations.
        """
        result = {}

        for action_name in self.action_fluents:
            if action_name not in self._action_bounds:
                continue

            bounds = self._action_bounds[action_name]
            num_agents = bounds['num_agents']
            low = bounds['low']
            high = bounds['high']

            # Create possible values as JAX array
            possible_values = jnp.arange(low, high + 1, dtype=jnp.int32)

            # Pre-create base substitution dict (shared across all checks)
            base_subs = env_state.subs
            model_params = env_state.model_params
            key = env_state.key

            # Define function to check if a single (agent, action_value) is valid
            def check_single_action(agent_idx, action_value):
                # Create test action: this agent takes action_value, others take noop (0)
                test_action_values = jnp.zeros(num_agents, dtype=jnp.int32)
                test_action_values = test_action_values.at[agent_idx].set(action_value)

                # Update only the action in the base subs
                test_subs = {**base_subs, action_name: test_action_values}

                # Check preconditions and convert to int
                is_valid = self._check_preconditions(test_subs, model_params, key)
                return jnp.where(is_valid, 1, 0)

            # Vmap over agents, then vmap over action values
            # First vmap over action values for a single agent
            check_agent_actions = jax.vmap(
                lambda action_value, agent_idx: check_single_action(agent_idx, action_value),
                in_axes=(0, None)
            )

            # Then vmap over agents
            check_all = jax.vmap(
                lambda agent_idx: check_agent_actions(possible_values, agent_idx),
                in_axes=0
            )

            # Execute vmapped computation
            agent_indices = jnp.arange(num_agents, dtype=jnp.int32)
            valid_mask = check_all(agent_indices)

            # Store as JAX array
            result[action_name] = valid_mask

        return result

    def sample_random_action(self, key: PRNGKey, env_state: Optional[EnvState] = None) -> Action:
        """Sample a random action from the action space.

        This is useful for testing and as a baseline policy.

        Args:
            key: JAX PRNG key
            env_state: Optional environment state (unused, for API consistency)

        Returns:
            actions: Random action dictionary
        """
        actions = {}
        for action_var in self.action_fluents:
            action_shape = jnp.shape(self.noop_actions[action_var])
            action_range = self.model.action_ranges[action_var]

            if action_range == 'bool':
                actions[action_var] = random.bernoulli(key, shape=action_shape)
                key, _ = random.split(key)
            elif action_range == 'real':
                # Sample from [-1, 1] as a default
                actions[action_var] = random.uniform(key, shape=action_shape,
                                                     minval=-1.0, maxval=1.0)
                key, _ = random.split(key)
            elif action_range == 'int':
                # Sample integers in a reasonable range
                actions[action_var] = random.randint(key, shape=action_shape,
                                                     minval=0, maxval=10)
                key, _ = random.split(key)
            else:
                # Enumerated type - sample from objects
                num_objects = len(self.model.type_to_objects[action_range])
                actions[action_var] = random.randint(key, shape=action_shape,
                                                     minval=0, maxval=num_objects)
                key, _ = random.split(key)

        return actions

    def get_noop_action(self) -> Action:
        """Get the no-op (default) action.

        Returns:
            actions: No-op action dictionary
        """
        return self.noop_actions.copy()

    def prepare_actions_for_sim(self, actions: Dict[str, Any]) -> Action:
        """Prepare action dictionary for the vectorized format required by the simulator.

        This function allows flexible action specification and converts it to the
        internal JAX array format. It supports:
        - Grounded action names (e.g., "move-north___a0" for specific agent)
        - Lifted action names (e.g., "move-north" for all agents)
        - Scalar values for grounded actions
        - Array values for lifted actions
        - Automatic type conversion and validation

        Args:
            actions: Dictionary mapping action names to values. Can use either:
                    - Grounded names: {"move-north___a0": True, "move-east___a1": True}
                    - Lifted names: {"move-north": jnp.array([True, False])}
                    - Mixed: {"move-north___a0": True, "move-east": jnp.array([False, True])}

        Returns:
            sim_actions: Action dictionary with all actions in vectorized JAX array format

        Raises:
            ValueError: If action names are invalid, types don't match, or shapes are wrong

        Example:
            ```python
            # Grounded actions (for specific agents/objects)
            actions = {
                "move-north___a0": True,  # Agent 0 moves north
                "move-east___a1": True    # Agent 1 moves east
            }
            sim_actions = env.prepare_actions_for_sim(actions)

            # Lifted actions (for all agents at once)
            actions = {
                "move-north": jnp.array([True, False]),  # Agent 0 north, agent 1 no
                "move-east": jnp.array([False, True])    # Agent 0 no, agent 1 east
            }
            sim_actions = env.prepare_actions_for_sim(actions)

            # Use with step
            env_state, timestep = env.step(env_state, sim_actions)
            ```
        """
        # Start with a copy of noop actions
        sim_actions = {action: jnp.copy(value)
                      for (action, value) in self.noop_actions.items()}

        for (action_name, value) in actions.items():
            # Parse grounded action name (e.g., "move-north___a0__x1")
            objects = []
            ground_action = action_name

            if action_name not in sim_actions:
                # Try to parse as grounded action
                action_name, objects = self.model.parse_grounded(action_name)

            # Check that the action is valid
            if action_name not in sim_actions:
                raise ValueError(
                    f'<{action_name}> is not a valid action-fluent, '
                    f'must be one of {set(sim_actions.keys())}.')

            # Get action type
            ptype = self.model.action_ranges[action_name]

            # Convert value to JAX array
            if not isinstance(value, jnp.ndarray):
                value = jnp.asarray(value)

            # Boolean type conversion
            if ptype == 'bool':
                value = jnp.asarray(value, dtype=bool)

            # Grounded assignment (specific object/agent)
            if objects:
                if jnp.ndim(value) > 0:
                    raise ValueError(
                        f'Grounded value specification of action-fluent <{ground_action}> '
                        f'received an array where a scalar value is required.')

                # Handle enumerated types - convert object name to index
                if ptype not in {'bool', 'int', 'real'}:
                    if isinstance(value.item(), str):
                        value = jnp.array(self.model.object_to_index.get(value.item(), value.item()))

                # Get the tensor and indices
                tensor = sim_actions[action_name]
                indices = self.model.object_indices(objects)

                if len(indices) != jnp.ndim(tensor):
                    raise ValueError(
                        f'Grounded action-fluent name <{ground_action}> '
                        f'requires {jnp.ndim(tensor)} parameters, got {len(indices)}.')

                # Update the tensor at the specified indices
                sim_actions[action_name] = tensor.at[tuple(indices)].set(value)

            # Vectorized assignment (all objects at once)
            else:
                tensor = sim_actions[action_name]

                # Check shape compatibility
                if jnp.shape(value) != jnp.shape(tensor):
                    raise ValueError(
                        f'Value array for action-fluent <{action_name}> must be of shape '
                        f'{jnp.shape(tensor)}, got array of shape {jnp.shape(value)}.')

                # Handle enumerated types - convert object names to indices
                if ptype not in {'bool', 'int', 'real'}:
                    if isinstance(value, (list, tuple)) or (isinstance(value, jnp.ndarray) and value.dtype == object):
                        # Convert array of object names to indices
                        value_list = np.asarray(value).flatten().tolist()
                        indices = [self.model.object_to_index.get(v, v) for v in value_list]
                        value = jnp.array(indices).reshape(jnp.shape(tensor))

                # Check dtype compatibility
                expected_dtype = jnp.asarray(tensor).dtype
                if ptype == 'bool':
                    value = jnp.asarray(value, dtype=bool)
                elif ptype == 'int' or ptype not in {'bool', 'int', 'real'}:
                    value = jnp.asarray(value, dtype=jnp.int32)
                elif ptype == 'real':
                    value = jnp.asarray(value, dtype=jnp.float32)

                sim_actions[action_name] = value

        # Validate enumerated type ranges
        for (action_name, value) in sim_actions.items():
            ptype = self.model.action_ranges[action_name]
            if ptype not in {'bool', 'int', 'real'}:
                max_index = len(self.model.type_to_objects[ptype]) - 1
                value_arr = jnp.asarray(value)
                if not jnp.all((value_arr >= 0) & (value_arr <= max_index)):
                    raise ValueError(
                        f'Values of action-fluent <{action_name}> of type <{ptype}> '
                        f'are not valid, must be in the range [0, {max_index}].')

        return sim_actions

    def get_observation_spec(self) -> Dict[str, Tuple[Tuple[int, ...], jnp.dtype]]:
        """Get the observation space specification.

        Returns:
            spec: Dictionary mapping observation variable names to (shape, dtype) tuples
        """
        spec = {}
        if self._is_pomdp:
            fluents = self.observ_fluents
        else:
            fluents = self.state_fluents

        for var in fluents:
            value = self.init_values[var]
            spec[var] = (jnp.shape(value), jnp.asarray(value).dtype)

        return spec

    def get_action_spec(self) -> Dict[str, Tuple[Tuple[int, ...], jnp.dtype]]:
        """Get the action space specification.

        Returns:
            spec: Dictionary mapping action variable names to (shape, dtype) tuples
        """
        spec = {}
        for var in self.action_fluents:
            value = self.noop_actions[var]
            spec[var] = (jnp.shape(value), jnp.asarray(value).dtype)

        return spec

    # ***********************************************************************
    # VISUALIZATION METHODS
    # ***********************************************************************

    def set_visualizer(self, visualizer: Any, movie_gen: Optional[Any] = None,
                      movie_per_episode: bool = False) -> None:
        """Set the visualizer for rendering states.

        The visualizer should have a render(state) method that takes a state
        dictionary and returns an image (PIL Image or similar).

        Note: Visualization is done outside JIT-compiled functions and does
        not affect the performance of step() and reset().

        Args:
            visualizer: A visualizer object with a render(state) method.
                       Typically an instance from pyRDDLGym.core.visualizer.*
            movie_gen: Optional MovieGenerator for saving frames and creating animations.
                      Use pyRDDLGym.core.visualizer.movie.MovieGenerator
            movie_per_episode: If True, saves a separate movie file per episode.
                             If False, accumulates all frames into a single movie.

        Example:
            ```python
            from pyRDDLGym.core.visualizer.chart import ChartVisualizer
            from pyRDDLGym.core.visualizer.movie import MovieGenerator

            env = JaxRDDLEnv("domain.rddl", "instance.rddl")
            viz = ChartVisualizer(env.model)
            movie_gen = MovieGenerator("/path/to/save", "myenv", max_frames=100)
            env.set_visualizer(viz, movie_gen=movie_gen)

            # Now you can render states
            env_state, timestep = env.reset(key)
            image = env.render(env_state)

            # Save movie when done
            env.save_movie()
            ```
        """
        self._visualizer = visualizer
        self._movie_generator = movie_gen
        self._movie_per_episode = movie_per_episode
        self._movies = 0

    def render(self, env_state: EnvState, save_frame: bool = True) -> Any:
        """Render the current environment state.

        This method extracts the state from the EnvState and calls the
        visualizer's render method. It's designed to be called outside
        JIT-compiled loops for visualization purposes.

        Args:
            env_state: The current environment state from step() or reset()
            save_frame: If True and movie_generator is set, saves the frame

        Returns:
            image: The rendered image (typically a PIL Image), or None if
                   no visualizer is set

        Example:
            ```python
            env_state, timestep = env.reset(key)
            image = env.render(env_state)

            # Save or display the image
            if image is not None:
                image.save('state.png')
            ```
        """
        if self._visualizer is None:
            return None

        # Extract state dictionary from EnvState
        state = env_state.state

        # Always convert to grounded format for visualizers
        # (JAX env internally uses vectorized representation)
        state = self.model.ground_vars_with_values(state)

        # Convert JAX arrays to NumPy arrays for visualizer compatibility
        state = {k: (np.asarray(v) if hasattr(v, '__array__') else v)
                 for k, v in state.items()}

        # Call visualizer's render method
        image = self._visualizer.render(state)

        # Save frame to movie generator if enabled
        if save_frame and self._movie_generator is not None and image is not None:
            self._movie_generator.save_frame(image)

        return image

    def save_movie(self, file_name: Optional[str] = None) -> None:
        """Save the accumulated frames as a movie (GIF or MP4).

        This should be called after an episode or set of episodes to generate
        the final movie file from all the frames that were captured during
        rendering.

        Args:
            file_name: Optional name for the movie file. If None, uses the
                      movie_generator's default name with episode number.

        Example:
            ```python
            from pyRDDLGym.core.visualizer.movie import MovieGenerator

            # Setup
            env.set_visualizer(viz, movie_gen=MovieGenerator(...))

            # Run episode
            env_state, _ = env.reset(key)
            for _ in range(horizon):
                image = env.render(env_state)  # Frames are saved automatically
                env_state, _ = env.step(env_state, actions)

            # Generate movie file
            env.save_movie()
            ```
        """
        if self._movie_generator is None:
            warnings.warn("No movie generator set. Call set_visualizer with movie_gen parameter.")
            return

        if file_name is None:
            file_name = self._movie_generator.env_name + '_' + str(self._movies)

        self._movie_generator.save_animation(file_name)
        self._movies += 1

    def reset_movie(self) -> None:
        """Reset the movie generator by clearing all saved frames.

        Useful when starting a new episode with movie_per_episode=False
        and you want to start fresh.
        """
        if self._movie_generator is not None:
            self._movie_generator.writer.reset()


# ***********************************************************************
# VECTORIZED ENVIRONMENT WRAPPER
# ***********************************************************************

class VectorizedJaxRDDLEnv:
    """Wrapper for running multiple JAX environments in parallel.

    This class provides a convenient interface for vectorized rollouts
    using vmap. It's particularly useful for:
    - Parallel policy evaluation
    - Batch gradient computation
    - Monte Carlo sampling

    Example:
        ```python
        # Create vectorized environment
        env = JaxRDDLEnv("domain.rddl", "instance.rddl")
        vec_env = VectorizedJaxRDDLEnv(env, num_envs=100)

        # Parallel reset
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 100)
        env_states, timesteps = vec_env.reset(keys)

        # Parallel step
        actions = vec_env.sample_random_actions(keys, env_states)
        env_states, timesteps = vec_env.step(env_states, actions)
        ```
    """

    def __init__(self, env: JaxRDDLEnv, num_envs: int):
        """Initialize vectorized environment.

        Args:
            env: Base JAX environment to vectorize
            num_envs: Number of parallel environments
        """
        self.env = env
        self.num_envs = num_envs

        # Create vmapped versions of reset and step
        self.reset = jax.vmap(env.reset)
        self.step = jax.vmap(env.step)
        self.sample_random_actions = jax.vmap(env.sample_random_action)

    def get_noop_actions(self) -> Action:
        """Get batched no-op actions for all environments.

        Returns:
            actions: Batched no-op actions with shape (num_envs, ...)
        """
        noop = self.env.get_noop_action()
        return jax.tree_map(lambda x: jnp.tile(x[None, ...], (self.num_envs,) + (1,) * len(x.shape)),
                           noop)


# ***********************************************************************
# UTILITY FUNCTIONS
# ***********************************************************************

def rollout(env: JaxRDDLEnv,
            policy_fn: Callable[[PRNGKey, EnvState], Action],
            key: PRNGKey,
            max_steps: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Execute a single episode rollout.

    This function can be JIT-compiled for fast rollouts.

    Args:
        env: The JAX environment
        policy_fn: Policy function mapping (key, env_state) -> actions
        key: PRNG key
        max_steps: Maximum number of steps (defaults to env.horizon)

    Returns:
        total_reward: Total accumulated reward
        num_steps: Number of steps taken
        info: Additional information dictionary
    """
    if max_steps is None:
        max_steps = env.horizon

    # Reset environment
    key, reset_key = random.split(key)
    env_state, timestep = env.reset(reset_key)

    total_reward = jnp.array(0.0)

    def step_fn(carry, _):
        env_state, total_reward, key, done = carry

        # Sample action from policy
        key, action_key = random.split(key)
        actions = policy_fn(action_key, env_state)

        # Step environment (but don't update if already done)
        new_env_state, timestep = env.step(env_state, actions)

        # Accumulate reward only if not done
        reward_to_add = jnp.where(done, 0.0, timestep.reward)
        new_total_reward = total_reward + reward_to_add

        # Check if done
        new_done = done | timestep.done | timestep.truncated

        return (new_env_state, new_total_reward, key, new_done), timestep

    (final_env_state, total_reward, _, _), timesteps = jax.lax.scan(
        step_fn,
        (env_state, total_reward, key, jnp.array(False)),
        None,
        length=max_steps
    )

    num_steps = final_env_state.timestep

    return total_reward, num_steps, {'timesteps': timesteps}


def parallel_rollout(env: JaxRDDLEnv,
                     policy_fn: Callable[[PRNGKey, EnvState], Action],
                     keys: jnp.ndarray,
                     max_steps: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Execute multiple parallel episode rollouts.

    Args:
        env: The JAX environment
        policy_fn: Policy function mapping (key, env_state) -> actions
        keys: Array of PRNG keys, one per rollout
        max_steps: Maximum number of steps per episode

    Returns:
        total_rewards: Array of total rewards, shape (num_rollouts,)
        num_steps: Array of episode lengths, shape (num_rollouts,)
    """
    vmapped_rollout = jax.vmap(lambda k: rollout(env, policy_fn, k, max_steps))
    total_rewards, num_steps, _ = vmapped_rollout(keys)
    return total_rewards, num_steps
