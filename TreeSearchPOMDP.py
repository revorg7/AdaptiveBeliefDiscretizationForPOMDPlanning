# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 11:44:06 2020


This is the POMDP version of TreeSearch.py, Assume Discrete Observation (in Node class, when sample next-samples)

ESSENTIALLY, THE MAIN THING I NEED TO DO IS CONVERT THE BAYES-FILTER UPDATE FOR COUNTING DISTRIBUTION CASE
EDIT: BAYES-FILTER IS USED TO CALC PROB IF OBS ARE DISCRETE AND NEED TO IMPLEMENT ITS MORE GENERAL VERSION
IF NEED TO USE IT AS A SAMPLER
"""


from algoPOMDP import AlgoPOMDP as Algo
import pdb
import numpy as np
from collections import defaultdict


class Node:
  def __init__(self,tree,parent,level,a,obs,prob,bel):
    self.tree = tree
    self.parent = parent    #Used in two ways, either denotes single parent or denotes list of parents (for RepNode)
    self.depth = level
    self.bel = bel/np.sum(bel)
    self.obs = obs
    self.prob = prob
    self.act = a #Also dual use, either denotes single action by a parent, or list of actions by each parent

  def Bayesfilter(self,a,obs_samples):
    prior = np.dot(self.tree.tr_dist , self.bel)
    prior = prior/np.sum(prior,axis=0) #Even the marginal doesn't sum to 1, due to float error
    posterior = []
    z_marginal = []
    for obs in obs_samples:
      post = self.tree.obs_dist[:,obs] * prior[:,a]
      z_marginal.append(sum(post))
      post = post/z_marginal[-1] # Dividing by Normalization constant Eta
      posterior.append(post)
    return posterior,z_marginal
    
  #During BI, the prob. of child node will be the total no. of beliefs that it represents
  #Full parameter represents full expansion, for all possible observation
  #EDIT: The "obs-post small and remove" trick of BAMDP is not applicable
  #since marginal is over next-state and not obs, so in bayes-filter, no matter what is obs
  #It will give some high post to some of the states, since sum of post = 1
  #This gives hint for sampling based Bayes-filter, need to sample from marginal only
  #high-prob next-states, while still keeping all possible obs,then elimination of child nodes
  #will simply happen due to obs_dist[sampled_state,obs]*marginal[sampled_state] being small
  def Expand(self,pol=None,full=False):
    lis_A = range(self.tree.A)
    if self.depth > self.tree.H:
      return []
    
    if pol is not None:
        lis_A = [ np.argmax( np.dot(pol,self.bel) ) ] #Selecting the action with best alpha-vectors, #THIS HELPS DO BOTH POL EVAL and VI

    children = []
    z_samples = range(self.tree.Z)
    for a in lis_A:  #JUST THIS METHOD WILL CHANGE IN POLICY EVAL ALGO VS VI ALGORITHM
#      normalizer = np.sum(self.bel)
#      if normalizer == 0:
#        z_samples = np.random.choice(range(S),self.tree.n_samples,p=[1.0/S]*S)
#      else:
#        z_samples = np.random.choice(range(S),self.tree.n_samples,p=self.bel/normalizer)

      posteriors,z_marginals = self.Bayesfilter(a,z_samples) #Dont need this actually, update_memory() method should handle it
      for post,obs,prob in zip(posteriors,z_samples,z_marginals):
        if not full and prob < 0.1: #Certain prob. threshold
          continue
        nn = self.tree.update_memory(self.bel,post,self.depth)  #ACTUALLY INCORRECT TO SAY MEM IS UPDATED INSIDE ALGPOMDP CLASS
        children.append( Node(self.tree,self,self.depth+1,a,obs,prob,nn) ) # NN IS NOT NORMALIZED, HENCE DONE AT NODE INITIALIZATION

    return children


class PolicyEval(Algo):
  def __init__(self,init_bel,SAHZDisc_tuple,tr,obdist,finite_mem_params,samples=10):
    self.S,self.A,self.H,self.Z,self.gamma = SAHZDisc_tuple
    self.counter,self.tolerance = finite_mem_params
    self.counter, self.delta = finite_mem_params #Tuple of lists (atleast delta is a list)
    self.init_bel,self.n_samples = init_bel,samples
    self.tr_dist,self.obs_dist = tr,obdist
    super().__init__((self.S,self.H),self.counter,self.tolerance)
    self.nodes_at_level = defaultdict(list)


  def Expand(self,pol=None):

    #Main tree expansion logic
    self.nodes_at_level[0].append( Node(self,None,0,None,None,None,self.init_bel) )
    for depth in range(self.H-1):
      curr_nodes = self.nodes_at_level[depth]
      for curr_node in curr_nodes:
        children = curr_node.Expand(pol)
        self.nodes_at_level[depth+1] += children


      RepNodes = self.Cluster(depth+1,self.nodes_at_level[depth+1])
      self.nodes_at_level[depth+1] = []    #Flush out nodes with single parent
      self.nodes_at_level[depth+1] = RepNodes

  #MANAGES IMP INFO TO BE STORED, that will be needed later for tree evaluation (backward induction)
  def Cluster(self,depth,nodes):
    d = []
    p = []
    a = []
    pr = []
    for node in nodes:
      d.append(node.bel) #chk what info needs to be stored
      p.append(node.parent)
      a.append(node.act)
      pr.append(node.prob)
    p = np.array(p)
    d = np.array(d)
    a = np.array(a)
    pr= np.array(pr)

    RepNodes = [] #Representative nodes
    clusters,idxs = self.clustering(depth,d)
    for idx,cluster in enumerate(clusters):
      all_nodes = np.where(np.array(idxs) == idx)[0] #All nodes belonging to a particular cluster
      parents = p[all_nodes] #Collecting parent info from those idx
      actions = a[all_nodes]
      probs = pr[all_nodes]
      mean = np.mean(d[all_nodes],axis=0)
      RepNodes.append(Node(nodes[0].tree,parents,depth,actions,None,probs,mean)) #ALTHOUGH THE INFO NEEDED FOR TREE EVALUATION IS FROM PARENT TO CHILD, BUT THAT IS LOST WHILE CLUSTERING THE CHILDREN GLOBALLY AT NEXT DEPTH
    print('no of Nodes & RepNodes at depth ',depth,' : ',len(nodes),'&',len(RepNodes))
    return RepNodes

  #NEEDS TO BE MODIFIED TO COUNT PER ACTION
  def _calc_pvals(self):
    p_values = []
    for depth in range(self.H-1):

      child_nodes = self.nodes_at_level[depth+1]
      parents = self.nodes_at_level[depth]
      counts = np.zeros( ( self.A, len(parents),len(child_nodes) ) ) #Actually just need the normalization constant
      for i,parent in enumerate(parents):
        for j,child in enumerate(child_nodes):
          for k,parent_representation in enumerate(child.parent):
            if id(parent_representation) == id(parent): #will not work if there are multiple equiv. parents (happens due to belief integerization after clustering)
             # on a 2nd thought, it will not be a problem if we are using pickle.dumps() as comparator
              counts[child.act[k],i,j] += child.prob[k]

      normalizer = np.sum(counts,axis=2)    #ACTUALLY REDUDANT AS DOING CHILD.PROB[K] ABOVE
      p_values .append ( counts / np.expand_dims(normalizer,axis=2) )

    return p_values

  #Option = 1, following given policy, option=0, taking max-action
  def _init_leafs(self,Rfunc,pol,option=None):
      
    children = self.nodes_at_level[self.H-1]
    V = []
    for child in children:
      marginal = np.dot(child.bel,Rfunc)
      marginal = marginal/np.sum(marginal,axis=0)
      if option:
        V.append(marginal[np.argmax( np.dot(pol,child.bel) )]) #Following given policy thereafter
      else:
        V.append(max(marginal)) #Taking max over actions
    return V

  #CANT DO VI WITH THIS: Need to store action information in child node (taken by parent node)
  #^^EDIT: Rectified, now stores the actions that lead to the RepNode by various parents
  #^^EDIT: Action info only needed for VI, not PI as implemented here
  def Eval(self,Rfunc,pol):

      
    #Intialization
    V = self._init_leafs(Rfunc,pol)
    p_values = self._calc_pvals()
    
    #Backward Induction
    for depth in range(self.H-2,-1,-1):

      parents = self.nodes_at_level[depth]
      child_nodes = self.nodes_at_level[depth+1]
      
      condition = np.zeros((len(parents) , len(child_nodes)))   #Since their is risk to multi-counting the same parent
                                                                  #^^EDIT: Shouldn't be for PI, only VI

      temp_V = [0.0]*len(parents)
      for i,parent in enumerate(parents):
        marginal = np.dot(parent.bel,Rfunc)
        marginal = marginal/np.sum(marginal,axis=0) #Even the marginal doesn't sum to 1, due to float error
        act = np.argmax( np.dot(pol,parent.bel) )
        for j,child in enumerate(child_nodes):
          if not condition[i,j]:
            temp_V[i] += p_values[depth][act,i,j] * ( marginal[act] + self.gamma * V[j] ) #IF I DO REVERSE SEARCH HERE, THEN RISK COUNTING SAME PARENT MULTIPLE TIMES
            condition[i,j] += 1
          else:
            print('ERROR, should not occur')
            pass

      V = temp_V
      
    return V

  #VI Eval
  def Eval_VI(self,Rfunc):

      
    #Intialization
    V = self._init_leafs(Rfunc,pol)
    p_values = self._calc_pvals()
    
    #Backward Induction
    final_action = 0
    for depth in range(self.H-2,-1,-1):

      parents = self.nodes_at_level[depth]
      child_nodes = self.nodes_at_level[depth+1]

      #Since their is risk to multi-counting
      condition = np.zeros( ( len(parents),len(child_nodes),self.A ) )

      temp_V = [0.0]*len(parents)
      Q = np.zeros((len(parents),self.A))
      for i,parent in enumerate(parents):
        marginal = np.dot(parent.bel,Rfunc)
        marginal = marginal/np.sum(marginal,axis=0) #Even the marginal doesn't sum to 1, due to float error

        for j,child in enumerate(child_nodes):
          for k,parent_rep in enumerate(child.parent):
            if id(parent_rep) == id(parent):
              act = child.act[k]
              if not condition[i,j,act]:
                #IF I DO REVERSE SEARCH HERE, THEN RISK COUNTING SAME PARENT MULTIPLE TIMES
                #Consider same (parent,child) but child represents two nodes, generated from same parent, for same action, but slightly different obs
                #^^above situation is taken care by p_values, so need to avoid multi-counting
                # But still need cond[i,j,act] to have act, to not skip (parent,child) pair arising from different actions
                Q[i,act] += p_values[depth][act,i,j] * ( marginal[act] + self.gamma * V[j] )
                condition[i,j,act] += 1
              else:
                print('Can possibly occur, see comments in code\n')
                pass
        temp_V[i] = np.max(Q[i])
        if depth==0:
          final_action = np.argmax(Q[i])

      V = temp_V

    return V,final_action


if __name__ == "__main__":

    S=10
    A=4
    Z=10
    H = 3
    Disc = 0.99999
    counter_per_sa = 10
    Rfn = np.random.random_sample((S,A))
    init_bel = np.random.randint(0,high=counter_per_sa,size=(S,))
    pol = np.random.randint(0,A,size=(A,S)) #Can take Q-MDP policy here, Maintaing alpha-vectors essentially

    tr = np.random.randint(0,high=1000,size=(S,A,S))
    z = np.sum(tr,axis=2)
    tr=tr/z.reshape(S,A,1) #Adding extra axis to make it ready for broadcasting division

    obs_dist = np.random.randint(0,high=1000,size=(S,Z))
    z = np.sum(obs_dist,axis=1)
    z_broadcast = z[:,np.newaxis]
    obs_dist = obs_dist/z_broadcast
    
    print("Doing Pol Eval")
    alg = PolicyEval(init_bel,(S,A,H,Z,Disc),tr,obs_dist, ([counter_per_sa]*H,[0.001]*H) )
    alg.Expand(pol)
    #Runtime-warning is coming here, due to calc_pvals havning Nan due not taking all actions
    V1 = alg.Eval(Rfn,pol)
    V2,bst_act = alg.Eval_VI(Rfn)
    print(V1,V2) #Eval_VI should work even for fixed Pol, sanity chk should give same Val

    print("\n Doing VI")
    alg = PolicyEval(init_bel,(S,A,H,Z,Disc),tr,obs_dist, ([counter_per_sa]*H,[0.05]*H) )
    alg.Expand()
    V,bst_act = alg.Eval_VI(Rfn)
    print(V,bst_act)
    print("I THINK THE SLOWNESS IS STILL DUE TO THE MEMORY, IE, STORING ALL PARENT NODES, NEED TO JUST KEEP THE ESSENTIAL INFO NOW")
    print("Profiling shows clustering takes most time, followed by np.mean (inside Cluster method) and update_memory")
