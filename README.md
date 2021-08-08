"""
Created on Wed Feb 20,2021
"""
Code for 'Adaptive Belief Discretization for POMDP Planning'

For Pseudocode, directly goto TreeSearchPOMDP.py

^^ Note that it implements a clustering function, but its job is only to keep track of Indentical belief nodes that have been stored in memory as different objects.

For others, you need to install dependencies first: PyPOMDP and POMDPy libraries (Check paper for Github link). The RS & Tiger simulator used from 'github.com/pemami4911/POMDPy' library.

SamplingSearchPOMDP (For RandomPODMP env, it generates, solvers and outputs for later use at the same time, also uses above mentiond trick)
testRockSample (For RS environment)

*sh scripts for running multiple experiments.
