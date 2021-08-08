# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 01:36:01 2021



Only difference to previous version of planner:
Search for these comments inside BayesFilter():
1. sn would be a concatenated tuple of grid-position + true underlying rock-states
2. therefore this joint will be much sparse, redering any belief clustering unnecessary
3. Actually o should be gridpos + checkrock output together while sn be the sampled rock_states
4. Actually this will not work, since check=5 just checks rock-0, and I can only update its posterior
5. NEWEST: Overall the current BayesFilter is bad, cause underlying obs-likelihood model is distance , c.f comment below: "Doesnt help much cause with..."
6. ^^ Not so bad actualy, print(posterior) for actions-5,6 for see that some posteriors indeed are diff. from 0-1, just need to sample a lot due to bad likelihood model
7. See the comment above check_equality as well
"""


from algoPOMDPsparse import AlgoPOMDP as Algo
from sparseModel import sparseModel
import numpy as np
from collections import defaultdict
from examples.rock_sample.grid_position import GridPosition
from util import belief_update_simple
from numba import errors
import warnings

np.seterr(divide='ignore', invalid='ignore') #Ignore warnings
warnings.simplefilter('ignore', category=errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=errors.NumbaPendingDeprecationWarning)


class Node:
  def __init__(self,tree,parent,level,a,obs,vs,prob,r,bel):
    #Bel should be sparse with shape (1,S)
    self.tree = tree
    self.parent = parent    #Used in two ways, either denotes single parent or denotes list of parents (for RepNode)
    self.depth = level
    self.obs = obs
    self.visible_state = vs
    self.prob = prob
    self.r = r
#    self.n_childs = 0.0 #normalizer to directly calculate p_values << EDIT: Gets much complicated if done here, cause need to keep track on child_nodes per action
    self.act = a #Also dual use, either denotes single action by a parent, or list of actions by each parent
    self.bel = bel
    self.unique_rocks_sampled = [] #Can't leave it to the simulator, need to manage here
    self.myname = id(self)
    self.parentname = id(parent)

  # self.r == other.r and self.act == other.act , need not be considered since the Node substitued by other is retained, only its V-function of subtree is not computed twice. Draw graph to undestand.
  def check_equality(self,other):
    if self.visible_state == other.visible_state and self.bel == other.bel:
      return True
    else:
#      count = 0
#      for i in range(len(self.bel)):
#        if other.bel[i] == self.bel[i]:
#          count+=1
#      if count == len(self.bel) and self.visible_state == other.visible_state:
#        pdb.set_trace()
      return False
    
  def BayesfilterSimple(self,act):
    if act < 4:
      s = np.random.binomial(n=1,p=self.bel) #Don't need more than 1 sample
      combined_state = self.tree.model.combine_state(self.visible_state,s)
      self.tree.model.model.unique_rocks_sampled = self.unique_rocks_sampled
      out_sample = self.tree.model.generate_step(combined_state,act)
#      return [list(self.bel)],[out_sample.o],[1.0],[out_sample.r],[out_sample.o[:2]],[out_sample]
      #NO NEED TO COPY BELIEF IF IT DOESNT CHNAGE
      return [self.bel],[out_sample.o],[1.0],[out_sample.r],[out_sample.o[:2]],[out_sample]

    elif act > 4:
      self.tree.model.model.unique_rocks_sampled = self.unique_rocks_sampled
      return belief_update_simple(self.tree.model,self.bel,self.visible_state,act,self.tree.n_samples)

    else:
      #Sampling from generative model
      rewards = []
      out_samples = [] #<<Just for debugging
      for i in range(self.tree.n_samples):
        combined_state = self.tree.model.combine_state(self.visible_state,np.random.binomial(n=1,p=self.bel))
        self.tree.model.model.unique_rocks_sampled = self.unique_rocks_sampled
        out_sample  = self.tree.model.generate_step(combined_state,act)
        rewards.append(out_sample.r)
        out_samples.append(out_sample)

      out_sample = out_samples[-1] # << Representative sample for 'sample' action
      avg_r = sum(rewards)/len(rewards)
#      return [list(self.bel)],[out_sample.o],[1.0],[avg_r],[out_sample.o[:2]],out_samples
      return [self.bel],[out_sample.o],[1.0],[avg_r],[out_sample.o[:2]],out_samples
#      return self.Bayesfilter(act) << Cant do simply this cause it gives absoluetely incorrect posteriors, better remove it
  
  def Expand(self,pol=None,full=False):
    lis_A = range(self.tree.A)
    if self.depth > self.tree.H:
      return []
    
    if pol is not None:
        lis_A = [ np.argmax( self.bel.dot(pol) ) ] #Selecting the action with best alpha-vectors, #THIS HELPS DO BOTH POL EVAL and VI

    children = []
    for a in lis_A:  #JUST THIS METHOD WILL CHANGE IN POLICY EVAL ALGO VS VI ALGORITHM
      #out_samples is just debug info
#      if self.tree.nodes_at_level[0][0].visible_state == (0,4) and (a==6 or a==4 or a == 1):
#        pdb.set_trace()
      posteriors,z_samples,z_marginals,r_samples,visible_states,out_samples = self.BayesfilterSimple(a)
      if (np.array(z_marginals)<0.03).all():
        print("A : ", a, " is not exapnded further")
      for post,obs,prob,r,vs in zip(posteriors,z_samples,z_marginals,r_samples,visible_states):
        if not full and prob < 0.03: #Certain prob. threshold
          continue
        nn = self.tree.update_memory(post,self.depth)  #ACTUALLY INCORRECT TO SAY MEM IS UPDATED INSIDE ALGPOMDP CLASS
        #RS specific
        node = Node(self.tree,self,self.depth+1,a,obs,vs,prob,r,nn)
#        node.unique_rocks_sampled = list(self.unique_rocks_sampled) # NO NEED TO COPY IF DOESNT CHANGE
        node.unique_rocks_sampled = self.unique_rocks_sampled
        if a == 4 and abs(r) != 0.0:
          node.unique_rocks_sampled = list(self.unique_rocks_sampled)
          rock_no = self.tree.model.model.get_cell_type(GridPosition(vs[0],vs[1]))
          if rock_no != -1:
            node.unique_rocks_sampled.append(rock_no)
        children.append( node ) # NN IS NOT NORMALIZED, HENCE DONE AT NODE INITIALIZATION

    return children


class PolicyEval(Algo):
  def __init__(self,init_bel,SAHZDisc_tuple,model,finite_mem_params,samples=10):
    self.S,self.A,self.H,self.Z,self.gamma = SAHZDisc_tuple
    self.counter,self.tolerance = finite_mem_params
    self.counter, self.delta = finite_mem_params #Tuple of lists (atleast delta is a list)
    self.init_bel,self.n_samples = init_bel,samples
    self.model = model
    super().__init__((self.S,self.H),self.counter,self.tolerance)
    self.nodes_at_level = defaultdict(list)
    self.nodes_at_level[0].append( Node(self,None,0,None,None,None,None,None,init_bel) )

  def update(self,bel,vs=None):
    self.init_bel = bel
    self.nodes_at_level = defaultdict(list)
    node = Node(self,None,0,None,None,vs,None,None,bel)
    node.unique_rocks_sampled = list(self.model.model.unique_rocks_sampled)
    self.nodes_at_level[0].append( node )
    

  def Expand(self,pol=None,Cluster=True):

    #Main tree expansion logic
    total_skipped = 0
    for depth in range(self.H-1):
      curr_nodes = self.nodes_at_level[depth]
#      print("Total no. of Nodes at depth ",depth," is ",len(curr_nodes))
      already_calculated = {} #Skipped nodes will be replaced during expansion (by same process), all works due to same ordering of list self.nodes_at_level[depth]
      for i,curr_node in enumerate(curr_nodes):
        cond = False
        for k in already_calculated:
          if curr_nodes[k].check_equality(curr_node):
            cond = True
            break
        if cond:
          total_skipped += 1
          continue
#        if self.nodes_at_level[0][0].visible_state == (0,4):
#          pdb.set_trace()
        children = curr_node.Expand(pol) #<<Sometimes here comes problem, need to chk
        self.nodes_at_level[depth+1] += children
        already_calculated[i] = True

#      print("\n Already calculated at depth ",depth)
#      for k in already_calculated:
#        print(id(curr_nodes[k]))

      if Cluster:
        RepNodes = self.Cluster(depth+1,self.nodes_at_level[depth+1])
        self.nodes_at_level[depth+1] = []    #Flush out nodes with single parent
        self.nodes_at_level[depth+1] = RepNodes

#    print("Total no. of Nodes at depth ",self.H-1," is ",len(self.nodes_at_level[self.H-1]))    
#    print("Total skipped at Expand ",total_skipped)

  #MANAGES IMP INFO TO BE STORED, that will be needed later for tree evaluation (backward induction)
  def Cluster(self,depth,nodes):
    d = []
    p = []
    a = []
    pr = []
    r = []
    for node in nodes:
      d.append(np.asarray(node.bel.todense()).reshape(-1)) #For compatiblity with algoPOMDP class
      p.append(node.parent)
      a.append(node.act)
      pr.append(node.prob)
      r.append(node.r)
    p = np.array(p)
    d = np.array(d)
    a = np.array(a)
    pr= np.array(pr)
    r = np.array(r)

    RepNodes = [] #Representative nodes
    clusters,idxs = self.clusteringNN(depth,d)
    for idx,cluster in enumerate(clusters):
      all_nodes = np.where(np.array(idxs) == idx)[0] #All nodes belonging to a particular cluster
      parents = p[all_nodes] #Collecting parent info from those idx
      actions = a[all_nodes]
      probs = pr[all_nodes]
      rewards = r[all_nodes]
      mean = np.mean(d[all_nodes],axis=0)
      RepNodes.append(Node(nodes[0].tree,parents,depth,actions,None,probs,rewards,mean)) #ALTHOUGH THE INFO NEEDED FOR TREE EVALUATION IS FROM PARENT TO CHILD, BUT THAT IS LOST WHILE CLUSTERING THE CHILDREN GLOBALLY AT NEXT DEPTH
    print('no of Nodes & RepNodes at depth ',depth,' : ',len(nodes),'&',len(RepNodes))
    return RepNodes

  #MODIFIED TO COUNT PER ACTION
  def _calc_pvals(self):
    p_values = []
    r_values = []
    for depth in range(self.H-1):

      child_nodes = self.nodes_at_level[depth+1]
      parents = self.nodes_at_level[depth]
      counts = np.zeros( ( self.A, len(parents),len(child_nodes) ) ) #Actually just need the normalization constant
      avg_r = np.zeros( ( self.A, len(parents),len(child_nodes) ) )
      counts_r = np.zeros( ( self.A, len(parents),len(child_nodes) ) )
      for i,parent in enumerate(parents):
        for j,child in enumerate(child_nodes):
          for k,parent_representation in enumerate(child.parent):
            if id(parent_representation) == id(parent): #will not work if there are multiple equiv. parents (happens due to belief integerization after clustering)
             # on a 2nd thought, it will not be a problem if we are using pickle.dumps() as comparator
              counts[child.act[k],i,j] += child.prob[k]
              avg_r[child.act[k],i,j] += child.r[k]
              counts_r[child.act[k],i,j] += 1
    
      normalizer = np.sum(counts,axis=2)    #ACTUALLY REDUDANT AS DOING CHILD.PROB[K] ABOVE
      p_values .append ( counts / np.expand_dims(normalizer,axis=2) )
      normalizer = np.sum(counts_r,axis=2)    #ACTUALLY REDUDANT AS DOING CHILD.PROB[K] ABOVE
      r_values .append ( avg_r / np.expand_dims(normalizer,axis=2) )

    return p_values,r_values

  #No Cluster
  def _calc_pvals_NoCluster(self):
    p_values = []
    r_values = []
    for depth in range(self.H-1):

      child_nodes = self.nodes_at_level[depth+1]
      parents = self.nodes_at_level[depth]
      counts = defaultdict(float)
      avg_r = defaultdict(float)
      for i,parent in enumerate(parents):
        for j,child in enumerate(child_nodes):
          if id(child.parent) == id(parent):
            counts[child.act,i,j] += child.prob
            avg_r[child.act,i,j] += child.r

      p_values .append ( counts  )
      r_values .append ( avg_r )

    return p_values,r_values

  #Option = 1, following given policy, option=0, taking max-action
  def _init_leafs(self,Rfunc,pol=None,option=None):
    children = self.nodes_at_level[self.H-1]
    V = []
    for child in children:
      if Rfunc == None:
        V.append(0.0)
        continue
      marginal = child.bel.dot(Rfunc)
      marginal = marginal/np.sum(marginal,axis=0)
      if option:
        V.append(marginal[np.argmax( child.bel.dot(pol) )]) #Following given policy thereafter
      else:
        V.append(max(marginal)) #Taking max over actions
    return V

  #VI Eval
  def Eval_VI(self,Rfunc):

      
    #Intialization
    V = self._init_leafs(Rfunc)
    p_values,r_values = self._calc_pvals()
    
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
        if Rfunc!=None:
          marginal = parent.bel.dot(Rfunc)
          marginal = marginal/np.sum(marginal,axis=0) #Even the marginal doesn't sum to 1, due to float error

        for j,child in enumerate(child_nodes):
          for k,parent_rep in enumerate(child.parent):
            if id(parent_rep) == id(parent):
              act = child.act[k]
              if Rfunc != None:
                reward = marginal[act]
              else:
                reward = r_values[depth][act,i,j]
              if not condition[i,j,act]:
                #IF I DO REVERSE SEARCH HERE, THEN RISK COUNTING SAME PARENT MULTIPLE TIMES
                #Consider same (parent,child) but child represents two nodes, generated from same parent, for same action, but slightly different obs
                #^^above situation is taken care by p_values, so need to avoid multi-counting
                # But still need cond[i,j,act] to have act, to not skip (parent,child) pair arising from different actions
                Q[i,act] += p_values[depth][act,i,j] * ( reward + self.gamma * V[j] )
                condition[i,j,act] += 1
              else:
#                print('Can possibly occur, see comments in code\n')
                pass
        temp_V[i] = np.max(Q[i])
        if depth==0:
          final_action = np.argmax(Q[i])

      V = temp_V

    return V,final_action

  def Eval_VI_NoCluster(self):
    #Intialization
    V = self._init_leafs(None)
    
    #Backward Induction
    total_skipped = 0
    final_action = 0
    for depth in range(self.H-2,-1,-1):

      parents = self.nodes_at_level[depth]
      child_nodes = self.nodes_at_level[depth+1]
#      for par in parents:
#        print(par.bel.items())
#      pdb.set_trace()

      already_calculated = {}
      temp_V = [0.0]*len(parents)
      Q = np.zeros((len(parents),self.A))
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
            Q[i,act] += child.prob * ( child.r + self.gamma * V[j] )
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

if __name__ == "__main__":

    S=10
    A=4
    Z=10
    H = 4
    Disc = 0.99999
    counter_per_sa = 5

    model = sparseModel((S,A,H,Z,Disc))

#    print("\n Doing VI")
    init_bel = model.bel
    alg = PolicyEval(init_bel,(S,A,H,Z,Disc),model, ([counter_per_sa]*H,[0.05]*H) )
    alg.Expand(Cluster=False)
    print("\n In Eval \n")
    V,bst_act = alg.Eval_VI_NoCluster()
    print(V,bst_act)
