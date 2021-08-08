Code for Paper 'Adaptive Belief Discretization for POMDP Planning' (https://arxiv.org/abs/2104.07276)
------------------------------------
For Pseudocode, directly goto TreeSearchPOMDP.py
^^ Note that it implements a clustering function, but its job is only to keep track of Identical belief nodes that have been stored in memory as different objects. Hence unrelated to the main idea of Paper.
------------------------------------
I've borrowed RS simulator from 'github.com/pemami4911/POMDPy' library, while code from 'github.com/namoshizun/PyPOMDP' is used to save my generated RandomPOMDPs in the POMDP format.
-----------------------------------
Description:

For RandomPODMP Env: SamplingSearchPOMDP.py (It generates, solves and outputs a fixed RandomPOMDP for later use/comparision)

For RockSample Env: testRockSample.py (It uses SSRockSample.py which implements Tree 'Node' class, 'Expand' & 'Cluster' method; latter is used only for memory managment, as explained above). RS is the most important environment for comparing different algorithm IMHO.
-----------------------------------
Above both scripts use AlgoPOMDPSparse.py as Base Class (implements NearestNeighbour method). Furthermore, I call these methods 'Sampling Search POMDP', since the underlying POMDP env. is usually specified as a simulator (e.g. RockSample), and there is no explicit likelihood function available to update the beliefs. So I update the posterior by it's MC-estimate using the generator (for an example, refer to the function 'belief_update_simple' in util.py)
-----------------------------------
*.sh scripts are for running multiple experiments.
