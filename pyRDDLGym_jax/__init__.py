__version__ = '2.8'

# Import the registration function for easy access
from pyRDDLGym_jax.registration import make

# Import core pure JAX environment classes
from pyRDDLGym_jax.core.env import (
    JaxRDDLEnv,
    VectorizedJaxRDDLEnv,
    EnvState,
    TimeStep,
    rollout,
    parallel_rollout
)

__all__ = [
    'make',
    'JaxRDDLEnv',
    'VectorizedJaxRDDLEnv',
    'EnvState',
    'TimeStep',
    'rollout',
    'parallel_rollout',
]
