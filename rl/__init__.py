"""Reinforcement-learning training for Liquid War 5 (PPO self-play).

Replaces the genetic-algorithm parameter search with a learned neural
cursor policy trained by PPO against the batched GPU engine
(simulator.engine.LiquidWarEngine).
"""
