"""
JAX-compatible space classes for pyRDDLGym-jax.

Provides jittable classes for action and observation spaces that work
with JAX's functional programming paradigm. These spaces are similar
to Gymnasium spaces but designed to work with JAX arrays and JIT compilation.

Based on JAXMARL/Gymnax space implementations.
"""
from typing import Tuple, Union, Sequence
from collections import OrderedDict
import chex
import jax
import jax.numpy as jnp


class Space(object):
    """
    Abstract base class for JAX-compatible spaces.

    All spaces must implement sample() and contains() methods
    that are JIT-compilable.
    """

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample a random element from the space.

        Args:
            rng: JAX PRNG key

        Returns:
            Sampled element as a JAX array
        """
        raise NotImplementedError

    def contains(self, x: jnp.ndarray) -> bool:
        """Check if x is a valid element of the space.

        Args:
            x: Element to check

        Returns:
            True if x is in the space, False otherwise
        """
        raise NotImplementedError


class Discrete(Space):
    """
    Discrete space representing a finite set of categorical choices.

    TODO: For now this is a 1d space. Make composable for multi-discrete.
    Example:
        space = Discrete(4)  # Represents {0, 1, 2, 3}
        sample = space.sample(key)  # Random int in [0, 4)
    """

    def __init__(self, num_categories: int, dtype=jnp.int32):
        """Initialize discrete space.

        Args:
            num_categories: Number of possible values (non-negative)
            dtype: Data type for samples (default: jnp.int32)
        """
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random value uniformly from {0, 1, ..., n-1}."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jnp.ndarray) -> bool:
        """Check if x is in [0, n)."""
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        info = [str(self.n)]
        if self.dtype != jnp.int64:
            info.append(f"dtype={self.dtype}")

        return f"Discrete({', '.join(info)})"


class MultiDiscrete(Space):
    """
    Multi-dimensional discrete space.

    Represents multiple discrete choices, one per dimension.

    Example:
        # 3 dimensions: first has 2 choices, second has 3, third has 2
        space = MultiDiscrete([2, 3, 2])
        sample = space.sample(key)  # e.g., [0, 2, 1]
    """

    def __init__(self, num_categories: Sequence[int], dtype=jnp.int32):
        """Initialize multi-discrete space.

        Args:
            num_categories: Number of choices per dimension
            dtype: Data type for samples (default: jnp.int32)
        """
        self.num_categories = jnp.array(num_categories)
        self.shape = (len(num_categories),)
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from each categorical choice."""
        return jax.random.randint(
            rng,
            shape=self.shape,
            minval=0,
            maxval=self.num_categories,
            dtype=self.dtype
        )

    def contains(self, x: jnp.ndarray) -> bool:
        """Check if all elements are within their respective ranges."""
        range_cond = jnp.logical_and(x >= 0, x < self.num_categories)
        return jnp.all(range_cond)

class Box(Space):
    """
    Continuous (or discrete) box-shaped space in R^n.

    Represents all points in a rectangular region with specified bounds.

    TODO: Add unboundedness - sampling from other distributions, etc.

    Example:
        # 2D continuous space in [0, 1] x [0, 1]
        space = Box(low=0.0, high=1.0, shape=(2,))

        # 3D discrete space
        space = Box(low=0, high=10, shape=(3,), dtype=jnp.int32)
    """

    def __init__(
        self,
        low: Union[float, jnp.ndarray],
        high: Union[float, jnp.ndarray],
        shape: Tuple[int, ...],
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize box space.

        Args:
            low: Lower bound (scalar or array matching shape)
            high: Upper bound (scalar or array matching shape)
            shape: Shape of the space
            dtype: Data type for samples
        """
        self.low = low if isinstance(low, jnp.ndarray) else jnp.array(low)
        self.high = high if isinstance(high, jnp.ndarray) else jnp.array(high)
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample uniformly from the box."""
        return jax.random.uniform(
            rng, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: jnp.ndarray) -> bool:
        """Check if x is within the box bounds."""
        return jnp.all(jnp.logical_and(x >= self.low, x <= self.high))

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        info = [f"low={self.low}, high={self.high}, shape={self.shape}"]
        if self.dtype != jnp.float32:
            info.append(f"dtype={self.dtype}")

        return f"Box({', '.join(info)})"

class Dict(Space):
    """
    Dictionary space representing a product of spaces.

    Useful for structured observation/action spaces with multiple components.

    Example:
        space = Dict({
            'position': Box(0, 1, shape=(2,)),
            'velocity': Box(-1, 1, shape=(2,)),
            'button': Discrete(2)
        })
    """

    def __init__(self, spaces: dict):
        """Initialize dictionary space.

        Args:
            spaces: Dictionary mapping keys to Space objects
        """
        self.spaces = OrderedDict(spaces)
        self.shape = ()  # Dict doesn't have a single shape
        self.dtype = None  # Dict doesn't have a single dtype

    def sample(self, rng: chex.PRNGKey) -> dict:
        """Sample from each subspace."""
        samples = {}
        for key, space in self.spaces.items():
            rng, subkey = jax.random.split(rng)
            samples[key] = space.sample(subkey)
        return OrderedDict(samples)

    def contains(self, x: dict) -> bool:
        """Check if x has valid values for all subspaces."""
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(
            self.spaces[key].contains(x[key])
            for key in self.spaces.keys()
        )

    def __getitem__(self, key):
        """Access subspace by key."""
        return self.spaces[key]

    @property
    def num_spaces(self):
        """Get number of subspaces."""
        return len(self.spaces)

    def keys(self):
        """Get all subspace keys."""
        return self.spaces.keys()

    def values(self):
        """Get all subspace objects."""
        return self.spaces.values()

    def items(self):
        """Get all (key, space) pairs."""
        return self.spaces.items()

class Tuple(Space):
    """Minimal jittable class for tuple (product) of jittable spaces."""
    def __init__(self, spaces: Union[tuple, list]):
        self.spaces = spaces

    def sample(self, rng: chex.PRNGKey) -> Tuple[chex.Array]:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return tuple(
            [
                space.sample(key_split[i])
                for i, space in enumerate(self.spaces)
            ]
        )

    def contains(self, x: jnp.int_) -> bool:
        """Check whether dimensions of object are within subspace."""
        # type_cond = isinstance(x, tuple)
        # num_space_cond = len(x) != len(self.spaces)
        # Check for each space individually
        out_of_space = 0
        for space in self.spaces:
            out_of_space += 1 - space.contains(x)
        return out_of_space == 0

    @property
    def num_spaces(self):
        """Get number of subspaces."""
        return len(self.spaces)
