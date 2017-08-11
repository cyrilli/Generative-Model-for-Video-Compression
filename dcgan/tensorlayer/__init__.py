"""
Deep learning and Reinforcement learning library for Researchers and Engineers
"""
# from __future__ import absolute_import


try:
    install_instr = "Please make sure you install a recent enough version of TensorFlow."
    import tensorflow
except ImportError:
    raise ImportError("__init__.py : Could not import TensorFlow." + install_instr)

from . import activation
act = activation
# from . import init


__version__ = "1.4.5"

global_flag = {}
global_dict = {}
