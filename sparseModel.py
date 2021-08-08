# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:38:39 2021


"""
from scipy.sparse import dok_matrix
import numpy as np
from collections import Counter
import pdb

class outsample:
  def __init__(self,o,sp,r):
    self.o = o
    self.sp = sp
    self.r = r

# I keep tr_dist, and obs_dist as Dense, would be useful in sampling
class sparseModel:
  def __init__(self,SAHZDisc_tuple,tr=None,obs_dist=None,reward=None,sparsity=0.8,bel=None):
    self.S,self.A,self.H,self.Z,self.disc = SAHZDisc_tuple
    self.tr_dist = tr
    self.obs_dist = obs_dist
    self.bel = bel
    self.reward = reward
    
    S,A,Z = self.S,self.A,self.Z
    if bel == None:
      self.bel = dok_matrix((1, S), dtype=np.float32)
#      length = max(1,int(S*sparsity))
      length = S #Dont take init_bel as sparse
      states = np.random.choice(range(S),length)
      for s in states:
        self.bel[0,s] += np.random.randint(0,high=1000)
      self.bel = self.bel/np.sum(self.bel)

    if obs_dist == None:
      obs_dist = np.zeros((S, Z), dtype=np.float32)
      length = max(1,int(Z*sparsity))
      for s in range(S):
        states = np.random.choice(range(Z),length)
        for z in states:
          obs_dist[s,z] += np.random.randint(0,high=1000)
      np.nan_to_num(obs_dist,copy=False)
      self.obs_dist = obs_dist/np.expand_dims(np.sum(obs_dist,axis=1),axis=1)

    if tr == None:
      tr_dist = np.zeros((S,A,S))
      length = max(1,int(S*sparsity))
      for s in range(S):
        for a in range(A):
          states = np.random.choice(range(S),length)
          for sn in states:
            tr_dist[s,a,sn] += np.random.randint(0,high=1000)
      np.nan_to_num(tr_dist,copy=False)
      norm = np.sum(tr_dist,axis=2)
      broadcast = np.expand_dims(norm,axis=2)
      self.tr_dist = tr_dist/broadcast
    
    if reward == None:
      reward = np.zeros((S,A,S))
      length = max(1,int(S*sparsity))
      for s in range(S):
        for a in range(A):
          states = np.random.choice(range(S),length)
          for sn in states:
            reward[s,a,sn] = np.random.rand()
      self.reward = reward

  def generate_step(self,s,a):
    sn = np.random.choice(range(self.S),p=self.tr_dist[s,a])
    o = np.random.choice(range(self.Z),p=self.obs_dist[sn])
    r = self.reward[s,a,sn]
    return outsample(o,sn,r)


def to_pomdp_file(model, output_path=None,
                  discount_factor=0.95):
    """
    The .pomdp file format is specified at:
    http://www.pomdp.org/code/pomdp-file-spec.html

    Args:
        model : suitable
        output_path (str): The path of the output file to write in. Optional.
                           Default None.
        discount_factor (float): The discount factor
    Returns:
        (list, list, list): The list of states, actions, observations that
           are ordered in the same way as they are in the .pomdp file.
    """
    # Preamble
    try:
        all_states = range(model.S)
        all_actions = range(model.A)
        all_observations = range(model.Z)
        if model.disc != None:
            discount_factor = model.disc
    except NotImplementedError:
        print("S, A, O must be enumerable for a given agent to convert to .pomdp format")

    content = "discount: %f\n" % discount_factor
    content += "values: reward\n" # We only consider reward, not cost.

    list_of_states = " ".join(str(s) for s in all_states)
    assert len(list_of_states.split(" ")) == len(all_states),\
        "states must be convertable to strings without blank spaces"
    content += "states: %s\n" % list_of_states

    list_of_actions = " ".join(str(a) for a in all_actions)
    assert len(list_of_actions.split(" ")) == len(all_actions),\
        "actions must be convertable to strings without blank spaces"
    content += "actions: %s\n" % list_of_actions

    list_of_observations = " ".join(str(o) for o in all_observations)
    assert len(list_of_observations.split(" ")) == len(all_observations),\
        "observations must be convertable to strings without blank spaces"
    content += "observations: %s\n" % list_of_observations

    # Starting belief state - they need to be normalized
    total_belief = sum(model.bel[0,s] for s in all_states)
    content += "start: %s\n" % (" ".join(["%f" % (model.bel[0,s]/total_belief)
                                          for s in all_states]))

    # State transition probabilities - they need to be normalized
    for s in all_states:
        for a in all_actions:
            probs = []
            for s_next in all_states:
                prob = model.tr_dist[s, a, s_next]
                probs.append(prob)
            total_prob = sum(probs)
            for i, s_next in enumerate(all_states):
                prob_norm = probs[i] / total_prob
                content += 'T: %s : %s : %s %f\n' % (a, s, s_next, prob_norm)

    #My POMDP model has different format of obs_dist, so need to find next-state probs
#    vals = [y for (x,y) in model.bel.keys()]
#    normalizer = sum(model.bel.values())
#    pr = [p/normalizer for p in model.bel.values()]
      
#    sn_probs = np.zeros((model.A,model.S))
#    for a in all_actions:
#        s_samples = np.random.choice(all_states,10000,p=model.bel )
#        sn_samples = []
#        for s in s_samples:
#            sn_samples .append( np.random.choice(all_states,1,p=model.tr_dist[s,a] )[0] )
#        d = Counter(sn_samples)
#        norm = len(sn_samples)
#        for sn in all_states:
#            sn_probs[a,sn] = d[sn]/norm    
    #EDIT: THE MODEL SPECIFICATION SHOULDNT DEPEND ON BELIEF, FOR MY CASE, OBS-DIST IS SIMPLY ACTION-INDEPEDENT
    for s_next in all_states:
        for a in all_actions:
            #Creating next_state samples for all observations, they indirectly give obs marginal
            probs = []
            for o in all_observations:
#                prob = agent.observation_model.probability(o, s_next, a)
                prob = model.obs_dist[s_next,o] # ITS INDEPENDENT OF THE ACTION ACTUALLY, JUST NEXT-STATE DEPENDENT
                probs.append(prob)
            total_prob = sum(probs)
            if total_prob <= 0.0:
                print("No observation is probable under state={} action={}"\
                .format(s_next, a))
            for i, o in enumerate(all_observations):
                if total_prob <= 0.0:
                    prob_norm = 0.0
                else:
                    prob_norm = probs[i] / total_prob
                content += 'O: %s : %s : %s %f\n' % (a, s_next, o, prob_norm)

    # Immediate rewards
    for s in all_states:
        for a in all_actions:
            for s_next in all_states:
                # We will take the argmax reward, which works for deterministic rewards.
#                r = agent.reward_model.sample(s, a, s_next)
#                content += 'R : %s : %s : %s : *  %f\n' % (a, s, s_next, r)
                content += 'R:' + ' %s : %s : %s\n' % (a,s,s_next)
                content += ('%f ' % model.reward[s,a,s_next])* (model.Z - 1)
                content += '%f\n' % model.reward[s,a,s_next]

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(content)
    return all_states, all_actions, all_observations

if __name__ == "__main__":


    S=4
    A=2
    Z=3
    H = 3
    Disc = 0.99999
    m = sparseModel((S,A,H,Z,Disc))
    to_pomdp_file(m,output_path='myModel.POMDP')
