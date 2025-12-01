"""
Registration utilities for creating JAX RDDL environments.

This module provides the make() function for creating JaxRDDLEnv instances
from domain/instance identifiers or file paths, similar to pyRDDLGym.make().
"""

import importlib
import os
from typing import Optional, Type, Union

from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym_jax.core.env import JaxRDDLEnv


VALID_EXT = '.rddl'
REPO_MANAGER_MODULE = 'rddlrepository'
REPO_MANAGER_CLASS = 'RDDLRepoManager'


def make(domain: Union[str, RDDLLiftedModel],
         instance: Optional[str] = None,
         base_class: Type[JaxRDDLEnv] = JaxRDDLEnv,
         **env_kwargs) -> JaxRDDLEnv:
    """Creates a new JaxRDDLEnv from domain and instance identifier or file paths.

    There are currently three modes of execution:
    1. If the domain and instance arguments are both paths to RDDL files, then this
       creates a JaxRDDLEnv from the RDDL files.
    2. Otherwise if the domain and instance arguments are strings, then this
       creates a JaxRDDLEnv from rddlrepository. This requires rddlrepository to be
       installed and available on the Python path, i.e. pip install rddlrepository.
    3. Otherwise if the domain is an instance of RDDLLiftedModel, uses this model
       as the basis for the environment.

    Args:
        domain: The domain identifier, model instance, or path to domain RDDL
        instance: The instance identifier, or path to instance RDDL
        base_class: A subclass of JaxRDDLEnv to instantiate (default: JaxRDDLEnv)
        **env_kwargs: Other arguments to pass to the JaxRDDLEnv constructor

    Returns:
        env: A JaxRDDLEnv instance

    Raises:
        ValueError: If domain/instance paths are invalid
        ImportError: If rddlrepository is needed but not installed

    Example:
        ```python
        # From rddlrepository
        env = make("CartPole_Continuous_gym", "0")

        # From file paths
        env = make("path/to/domain.rddl", "path/to/instance.rddl")

        # From precompiled model
        model = RDDLLiftedModel(...)
        env = make(model, None)
        ```
    """

    # Check if the domain is a precompiled model
    if isinstance(domain, RDDLLiftedModel):
        env = base_class(domain=domain, instance=instance, **env_kwargs)
        return env

    # Check if arguments are file paths
    domain_is_file = os.path.isfile(domain)
    instance_is_file = os.path.isfile(instance)
    if domain_is_file != instance_is_file:
        raise ValueError(
            f'Domain and instance must either be both valid file paths or neither, '
            f'but got domain={domain} (is_file={domain_is_file}) and '
            f'instance={instance} (is_file={instance_is_file}).'
        )

    # If they are files, check they are RDDL files
    if domain_is_file:
        domain_is_rddl = str(os.path.splitext(domain)[1]).lower() == VALID_EXT
        instance_is_rddl = str(os.path.splitext(instance)[1]).lower() == VALID_EXT
        if not domain_is_rddl or not instance_is_rddl:
            raise ValueError(
                f'Domain and instance paths {domain} and {instance} '
                f'are not valid RDDL ({VALID_EXT}) files.'
            )

        # Extract environment
        env = base_class(domain=domain, instance=instance, **env_kwargs)
        return env

    # Check the repository exists
    spec = importlib.util.find_spec(REPO_MANAGER_MODULE)
    if spec is None:
        raise ImportError(
            'rddlrepository is not installed: '
            'can be installed with \'pip install rddlrepository\'.'
        )

    # Load the repository manager
    module = importlib.import_module(REPO_MANAGER_MODULE)
    manager = getattr(module, REPO_MANAGER_CLASS)()
    info = manager.get_problem(domain)

    # Extract environment
    domain_path = info.get_domain()
    instance_path = info.get_instance(instance)
    env = base_class(domain=domain_path, instance=instance_path, **env_kwargs)

    # Note: Visualization is not automatically set for JaxRDDLEnv
    # since it's designed for JIT-compiled rollouts without side effects.
    # Users can add visualization outside of the JIT-compiled loop if needed.

    return env
