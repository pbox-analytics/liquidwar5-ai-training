"""
PyTorch GPU-accelerated Liquid War simulator.

Reimplements the core Liquid War 5 game engine as batched tensor
operations, enabling hundreds of games to run simultaneously on GPU.
"""

from simulator.engine import LiquidWarEngine
