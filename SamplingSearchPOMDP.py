# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 00:19:44 2021


https://stackoverflow.com/questions/2540059/scipy-sparse-arrays
for sparse-vector 'a', need to be careful, np.dot(a,b) doesnt work, need a.dot(b). Check API on Scipy
Also need to take care that Posterior.shape = (1,S) now instead of previous (S,)
^^https://stackoverflow.com/questions/3337301/numpy-matrix-to-array
"""

from algoPOMDPsparse import AlgoPOMDP as Algo
from sparseModel import sparseModel
import numpy as np
from collections import defaultdict
from scipy.sparse import dok_matrix
from sparseModel import to_pomdp_file
import pdb,sys

np.seterr(divide='ignore', invalid='ignore') #Ignore warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class Node:
  def __init__(self,tree,parent,level,a,obs,prob,r,bel):
    #Bel should be sparse with shape (1,S)
    self.tree = tree
    self.parent = parent    #Used in two ways, either denotes single parent or denotes list of parents (for RepNode)
    self.depth = level
    self.obs = obs
    self.prob = prob
    self.r = r
    self.act = a #Also dual use, either denotes single action by a parent, or list of actions by each parent
    self.myname = id(self)

    if type(bel)==type(dok_matrix((1, self.tree.S), dtype=np.float32)):
      self.bel = bel/sum(bel.values())
    else:
      self.bel = dok_matrix((1, self.tree.S), dtype=np.float32)
      norm = sum(bel)
      for idx in np.nonzero(bel)[0]:
        self.bel[0,idx] = bel[idx]/norm

  def Bayesfilter(self,act):
    #Generating samples from currrent belief for MC estimation
    normalizer = sum(self.bel.values())
    if normalizer == 0:
      s_samples = np.random.choice(range(S),self.tree.n_samples,p=[1.0/S]*S)
    else:
      vals = [y for (x,y) in self.bel.keys()]
      pr = [p/normalizer for p in self.bel.values()]
      s_samples = np.random.choice(vals,self.tree.n_samples,p=pr )
    #Sampling from generative model
    out_samples = []
    for s in s_samples:
      out_samples.append(self.tree.model.generate_step(s,act))

    #Collecting samples info in right format
    joint = defaultdict(int)
    ob = defaultdict(int)
    sp_given_o = defaultdict(list)
    r_given_o = defaultdict(list)
    for out_sample in out_samples:
      o,sn = out_sample.o,out_sample.sp
      ob[o] += 1
      joint[(o,sn)] += 1
      sp_given_o[o].append(sn)
      r_given_o[o].append(out_sample.r)

    #Normalizer count should be same
    normalizer_joint = sum(joint.values())
    normalizer_o = sum(joint.values())
    
    #Calculating observation-wise sample info
    posterior = []
    z_samples = []
    r_samples = []
    z_marginal = []
    for o in ob.keys():
      z_samples.append(o)
      r_samples.append( sum(r_given_o[o]) / len(r_given_o[o]) ) #Avg R for given obs
      z_marginal.append(ob[o]/normalizer_o)
      sparse_posterior = dok_matrix((1, self.tree.S), dtype=np.float32)
      for sn in set(sp_given_o[o]):
        sparse_posterior[0,sn] = (joint[(o,sn)] / normalizer_joint) / z_marginal[-1]
      posterior.append( sparse_posterior )
    return posterior,z_samples,z_marginal,r_samples
    

  def Expand(self,pol=None,full=False):
    lis_A = range(self.tree.A)
    if self.depth > self.tree.H:
      return []
    
    if pol is not None:
        lis_A = [ np.argmax( self.bel.dot(pol) ) ] #Selecting the action with best alpha-vectors, #THIS HELPS DO BOTH POL EVAL and VI

    children = []
    for a in lis_A:  #JUST THIS METHOD WILL CHANGE IN POLICY EVAL ALGO VS VI ALGORITHM
      posteriors,z_samples,z_marginals,r_samples = self.Bayesfilter(a)
      for post,obs,prob,r in zip(posteriors,z_samples,z_marginals,r_samples):
        if not full and prob < 0.1: #Certain prob. threshold
          continue
        post = np.asarray(post.todense()).reshape(-1)
        nn = self.tree.update_memory(post,self.depth)  #ACTUALLY INCORRECT TO SAY MEM IS UPDATED INSIDE ALGPOMDP CLASS
        new_bel = dok_matrix((1, self.tree.S), dtype=np.float32)
        for idx in np.nonzero(nn)[0]:
          new_bel[0,idx] = nn[idx]
        children.append( Node(self.tree,self,self.depth+1,a,obs,prob,r,new_bel) ) # NN IS NOT NORMALIZED, HENCE DONE AT NODE INITIALIZATION

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


  def update(self,bel):
    self.nodes_at_level = defaultdict(list)
    if type(bel)==type(dok_matrix((1, self.S), dtype=np.float32)):
      self.init_bel = bel/sum(bel.values())
    else:
      self.init_bel = dok_matrix((1, self.S), dtype=np.float32)
      norm = sum(bel)
      for idx in np.nonzero(bel)[0]:
        self.init_bel[0,idx] = bel[idx]/norm

  def Expand(self,pol=None):

    #Main tree expansion logic
    self.nodes_at_level[0].append( Node(self,None,0,None,None,None,None,self.init_bel) )
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
#      counts = np.zeros( ( self.A, len(parents),len(child_nodes) ) ) #Actually just need the normalization constant
#      avg_r = np.zeros( ( self.A, len(parents),len(child_nodes) ) ) #Incorrect, there doesn't need to be averaging, anyways, keeping the variable name same
#      counts_r = np.zeros( ( self.A, len(parents),len(child_nodes) ) )
      counts = defaultdict(float)
      avg_r = defaultdict(float)
      for i,parent in enumerate(parents):
        for j,child in enumerate(child_nodes):
          for k,parent_representation in enumerate(child.parent):
            if id(parent_representation) == id(parent): #will not work if there are multiple equiv. parents (happens due to belief integerization after clustering)
             # on a 2nd thought, it will not be a problem if we are using pickle.dumps() as comparator
              counts[child.act[k],i,j] += child.prob[k]
              avg_r[child.act[k],i,j] += child.r[k]
 
#      normalizer = np.sum(counts,axis=2)    #ACTUALLY REDUDANT AS DOING CHILD.PROB[K] ABOVE
#      p_values .append ( counts / np.expand_dims(normalizer,axis=2) )
#      if (np.abs(p_values[-1] - counts) > 0.001).any():
#        print(np.where(p_values[-1] != counts))
#        print("Thr is problem") 
      p_values.append ( counts )
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

  #CANT DO VI WITH THIS: Need to store action information in child node (taken by parent node)
  #^^EDIT: Rectified, now stores the actions that lead to the RepNode by various parents
  #^^EDIT: Action info only needed for VI, not PI as implemented here
  def Eval(self,Rfunc,pol):

      
    #Intialization
    V = self._init_leafs(Rfunc,pol)
    p_values,r_values = self._calc_pvals()
    
    #Backward Induction
    for depth in range(self.H-2,-1,-1):

      parents = self.nodes_at_level[depth]
      child_nodes = self.nodes_at_level[depth+1]
      
      condition = np.zeros((len(parents) , len(child_nodes)))   #Since their is risk to multi-counting the same parent
                                                                  #^^EDIT: Shouldn't be for PI, only VI

      temp_V = [0.0]*len(parents)
      for i,parent in enumerate(parents):

        act = np.argmax( parent.bel.dot(pol) )
        reward = 0.0
        if Rfunc!=None:
          marginal = parent.bel.dot(Rfunc)
          marginal = marginal/np.sum(marginal,axis=0) #Even the marginal doesn't sum to 1, due to float error
          reward = marginal[act]

        for j,child in enumerate(child_nodes):
          if Rfunc==None:
            reward = r_values[depth][act,i,j]
          if not condition[i,j]:
            temp_V[i] += p_values[depth][act,i,j] * ( reward + self.gamma * V[j] ) #IF I DO REVERSE SEARCH HERE, THEN RISK COUNTING SAME PARENT MULTIPLE TIMES
            condition[i,j] += 1
          else:
            print('ERROR, should not occur')
            pass

      V = temp_V
      
    return V

  #VI Eval
  def Eval_VI(self,Rfunc):

      
    #Intialization
    V = self._init_leafs(Rfunc)
#    p_values,r_values = self._calc_pvals()
    
    #Backward Induction
    final_action = 0
    for depth in range(self.H-2,-1,-1):

      parents = self.nodes_at_level[depth]
      child_nodes = self.nodes_at_level[depth+1]


      temp_V = [0.0]*len(parents)
      Q = np.zeros((len(parents),self.A))
      for i,parent in enumerate(parents):
        if Rfunc!=None:
          marginal = parent.bel.dot(Rfunc)
          marginal = marginal/np.sum(marginal,axis=0) #Even the marginal doesn't sum to 1, due to float error

        for j,child in enumerate(child_nodes):
          for k,parent_rep in enumerate(child.parent):
            if parent_rep.myname == parent.myname:
              act = child.act[k]
              if Rfunc != None:
                reward = marginal[act]
              else:
                reward = child.r[k]
              Q[i,act] += child.prob[k] * (child.r[k] + self.gamma*V[j])
        temp_V[i] = np.max(Q[i])
        if depth==0:
          final_action = np.argmax(Q[i])

      V = temp_V

    return V,final_action


def Bayesfilter(m,bel,a,obs_samples):
  dense_bel = np.zeros(m.S)
  for k in bel.keys():
    dense_bel[k[1]] = bel[k]
  bel = dense_bel/sum(dense_bel)
  prior = np.dot(m.tr_dist , bel)
  prior = prior/np.sum(prior,axis=0) #Even the marginal doesn't sum to 1, due to float error
  posterior = []
  z_marginal = []
  for obs in obs_samples:
    post = m.obs_dist[:,obs] * prior[:,a]
    z_marginal.append(sum(post))
    post = post/z_marginal[-1] # Dividing by Normalization constant Eta
    posterior.append(post)

  return posterior,z_marginal


def run(m,alg,n_steps=90):
  alg.update(alg.init_bel)
  discount = alg.gamma
  discounted_reward = 0.0
  state = np.random.choice(range(alg.S))
  for i in range(n_steps):
    alg.Expand()
    V,bst_act = alg.Eval_VI(None)
    out_sample = m.generate_step(state,bst_act)
    discounted_reward += discount * out_sample.r

    # Updating
    discount *= alg.gamma
    state = out_sample.sp
    posteriors,z_marginal = Bayesfilter(m,alg.init_bel,bst_act,[out_sample.o])
    alg.update(posteriors[0])

  return discounted_reward

if __name__ == "__main__":

    S= int(sys.argv[1])
    A= int(sys.argv[2])
    Z= int(S)
    sp = float(sys.argv[3])
    path = './RandomPOMDPCorrectBI/'
#    S= 30
#    A= 4
#    Z= int(S)
#    sp = 0.8
    H = 4
    Disc = 0.95
    runs = 50
    model = sparseModel((S,A,H,Z,Disc),sparsity=sp)
    name = str(S) + '-' + str(A) + '-' + str(Z) + '-' + str(int(sp*10) ) + '.POMDP'
    to_pomdp_file(model,output_path = path+name)
    
    
    counter_per_sa = 5
    Rfn = np.random.random_sample((S,A))
    pol = np.random.randint(0,A,size=(A,S)) #Can take Q-MDP policy here, Maintaing alpha-vectors essentially


#    print("Doing Pol Eval")
#    alg = PolicyEval(init_bel,(S,A,H,Z,Disc),model, ([counter_per_sa]*H,[0.001]*H) )
#    alg.Expand(pol)
#    #Runtime-warning is coming here, due to calc_pvals havning Nan due not taking all actions
#    V1 = alg.Eval(Rfn,pol)
#    V2,bst_act = alg.Eval_VI(Rfn)
#    print(V1,V2) #Eval_VI should work even for fixed Pol, sanity chk should give same Val

#    print("\n Doing VI")
#    init_bel = model.bel
#    alg = PolicyEval(init_bel,(S,A,H,Z,Disc),model, ([counter_per_sa]*H,[0.05]*H) )
#    alg.Expand()
#    V,bst_act = alg.Eval_VI(None)
#    print(V,bst_act)

    init_bel = model.bel
    alg = PolicyEval(init_bel,(S,A,H,Z,Disc),model, ([counter_per_sa]*H,[0.001]*H) )
    avg = []
    for i in range(runs):
      avg .append( run(model,alg) )
    print(name,avg)
    f = open(path+ name + '.result', 'w')
    f.write(str(sum(avg)/len(avg)) + '\n')
    f.close()
