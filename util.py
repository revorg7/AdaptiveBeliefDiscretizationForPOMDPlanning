#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:11:51 2021



1. Reject Sampling doesn't work if likelihood P(obs|theta) is completely  zero, gets stuck in while loop
2. Search comment "INFORMATION LEAK" below
"""
from examples.rock_sample.grid_position import GridPosition
from examples.rock_sample.rock_state import RockState
from examples.rock_sample.rock_action import RockAction
from examples.rock_sample.rock_observation import RockObservation
from collections import defaultdict
import numpy as np

  #Particle filter / Reject sampling update of posterior
  #^^ Could have used it inside Bayesfilter() fn as well
def belief_update(m,bel,old_pos,action,obs,n_particles=1000):
  if type(action) == int:
    action = RockAction(action)
  if type(obs) == int:
    if obs == 1:
      obs = RockObservation(True,False)
    else:
      obs = RockObservation(False,False)
  if type(old_pos) == tuple:
      old_pos = GridPosition(old_pos[0],old_pos[1])


  particles = []
  while len(particles) < n_particles:
    old_state = RockState(old_pos, np.random.binomial(n=1,p=bel)) #Sampling the rock_states from prior
    result, is_legal = m.model.generate_step(old_state, action)
    if obs == result.observation:
      particles.append(result.next_state.rock_states)

  #obs-marginal
  marginal = len(particles)*1.0/n_particles
  #state-posterior
  normalizers = np.count_nonzero(particles,axis=0)
  post = []
  for norm in normalizers:
    post.append(norm/n_particles)
  return post,marginal

def belief_update_simple(m,bel,old_pos,act,n_samples=20):
  if type(old_pos) == tuple:
      old_pos = GridPosition(old_pos[0],old_pos[1])
  idx = act-5
  rock_no = idx
  
  #Sampling from generative model
  out_samples = [] #Just for debugging
  joint = defaultdict(int) #Obs-state
  marginal = defaultdict(int) #Obs
  for i in range(n_samples):
    old_state = RockState(old_pos, np.random.binomial(n=1,p=bel)) #Sampling the rock_states from prior
#    obs = m.model.actual_rock_states[rock_no] and (rock_no not in m.model.unique_rocks_sampled) #<< INFORMATION LEAK
    obs = old_state.rock_states[rock_no] and (rock_no not in m.model.unique_rocks_sampled)
    dist = old_state.position.euclidean_distance(m.model.rock_positions[rock_no])
    correct = np.random.binomial(1.0, m.model.get_sensor_correctness_probability(dist))
    if not correct:
      obs = int ( not obs)

    joint[obs,old_state.rock_states[idx]] += 1
    marginal[obs] += 1
    out_samples.append((old_state,obs))

  #Calculating observation-wise sample info
  posterior = []
  z_samples = []
  r_samples = []
  z_marginal = []
  visible_states = []
  for o in marginal.keys():
    z_samples.append( o ) 
    visible_states.append( (old_pos.i,old_pos.j) ) # sn.visible_state
    r_samples.append( 0.0 ) #Avg R for given obs
    z_marginal.append( float(marginal[o])/n_samples )
    
#    post = list(bel) #Deepcopy
#    post[idx] = (float(joint[o,1]) / n_samples) / z_marginal[-1]
    #SAVING SOME MEMORY HERE
    val = (float(joint[o,1]) / n_samples) / z_marginal[-1]
    if bel[idx] != val:
      post = list(bel)
      post[idx] = val
    else:
      post = bel
    posterior.append( post )

  return posterior,z_samples,z_marginal,r_samples,visible_states,out_samples

#@jit(nopython=True, fastmath=True) #NEED TO REDEFINE NODE CLASS TO MAKE THIS BACKWARD INDUCTION WORK
def Eval_VI_NoCluster(alg,A,gamma,H,V):
  #Intialization
#  gamma = alg.gamma
#  A = alg.A
#  V = alg._init_leafs(None)
    
  #Backward Induction
  total_skipped = 0
  final_action = 0
  for depth in range(H-2,-1,-1):

    parents = alg.nodes_at_level[depth]
    child_nodes = alg.nodes_at_level[depth+1]
#      for par in parents:
#        print(par.bel.items())
#      pdb.set_trace()

    already_calculated = {}
    temp_V = [0.0]*len(parents)
    Q = np.zeros((len(parents),A))
    for i,parent in enumerate(parents):
      cond = False
      for k in already_calculated:
        if parents[k].check_equality(parent):
          temp_V[i] = already_calculated[k]
          cond = True
          break
      if cond:
        total_skipped += 1
        continue
      for j,child in enumerate(child_nodes):
#          if depth==1 and parent.act == 1 and child.visible_state == (3,1):
#            pdb.set_trace()
#          if depth==1 and parent.act == 1 and child.act==4:
#            pdb.set_trace()
        if child.parentname == parent.myname:
#            if depth==2 and parent.act == 2 and parent.parent.act== 2 and child.act==4:
#              pdb.set_trace()
          act = child.act
          Q[i,act] += child.prob * ( child.r + gamma * V[j] )
#            if p_values[depth][act,i,j] != child.prob and reward != child.r:
#              pdb.set_trace()
      temp_V[i] = np.max(Q[i])
      already_calculated[i] = temp_V[i]
      if depth==0:
#          final_action = np.argmax(Q[i])
#          print("Root Q-vals are ",Q[i])
        winner = np.argwhere( Q[i] == np.amax(Q[i]) ) #Randomizing
        final_action = np.random.choice(winner[0])

#      print("\n Already calculated at depth ",depth)
#      for k in already_calculated:
#        print(id(parents[k]))

    V = temp_V

#    print("Total Skipped at Eval ",total_skipped)
  return V,final_action
