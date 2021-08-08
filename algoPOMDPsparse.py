# -*- coding: utf-8 -*-


import numpy as np
from fastdist import fastdist
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from numba import jit

@jit(nopython=True, fastmath=True)
def NN(post,depth,S):
  discretization_lvl = 1.0/depth
  nn = []
  for i in range(S):
    val = post[i]
    if val == 0:
      nn.append(val)
      continue
    step = val//discretization_lvl
    floor,ceil = max(0.0,discretization_lvl*step),min(1.0,discretization_lvl*(step+1))
    cordinate = floor if (val-floor) < (ceil-val) else ceil
    nn.append( cordinate )
  return nn


class AlgoPOMDP:
  def __init__(self, SH_tuple,counter=[],tolerance=[]): #Tolerance should be ideally an increasing sequence
    self.S,self.H = SH_tuple
    self.counter = counter #THIS SHOULD BE ADAPTED TO DEPTH, NOT BE CONSTANT
    self.memory = defaultdict(list) #Just to store valid grid points at certain depth
    self.delta = np.zeros((self.H,))
    self.delta = tolerance

  #THIS ONE REMEDIES THE PROBLEM STATED IN PREVIOUS NN() FUNCTION
  def NN_simple(self,post,depth,order=1):
    "The possible cordinates of NN on a uniform grid would be the points surrounding the true posterior"
    discretization_lvl = 1.0/self.counter[depth]
    nn = []
    for i in range(self.S):
      step = post[i]//discretization_lvl
      floor,ceil = max(0.0,discretization_lvl*step),min(1.0,discretization_lvl*(step+1))
      cordinate = floor if (post[i]-floor) < (ceil-post[i]) else ceil
      nn.append( cordinate )


#    if np.count_nonzero(np.array(nn)==np.array(bst_cordinates)) != len(bst_cordinates) :
#      print('Throw Error\n',nn,bst_cordinates)
    return nn


  #Although counting from 1 to M is one possible parameter-space, other could be counting 1 to M in exponential gaps
  #The best-parameter space would be the higest entropy parameter-space, given a fixed memory
  def update_memory(self,true_post,level):
#    nn,dist = self.NN(prior,true_post,level)
#    nn = self.NN_simple(true_post,level)
    nn = NN(true_post,self.counter[level],self.S)
    return nn

  def clusteringNN(self,t,beliefs,metric_order=1):
    neigh = NearestNeighbors(radius=self.delta[t],p=metric_order)
    size = len(beliefs)
    neigh.fit(beliefs)
    groups = neigh.radius_neighbors()[1]
    lengths = []
    for r in groups:
      lengths.append(r.shape[0])
    idxs=sorted(range(size),key=lengths.__getitem__,reverse=True) #Sort by id in descending order
    nodes = []
    clusters = [-1]*size
    number=0
    for idx in idxs:
      if clusters[idx]!=-1:  #Incorrect, maybe I am skipping some nodes just because they are singular
        continue
      clusters[idx] = number
      lis = [beliefs[idx]]
      for bel_id in groups[idx]:
        if clusters[bel_id]!=-1:
          continue
        clusters[bel_id] = number
        lis.append(beliefs[bel_id])
      nodes.append(lis)
      number+=1
    
#    for node in nodes:
#      vals = cdist(node,node[0].reshape(1,-1),'minkowski',p=1)
#      if max(vals)>self.delta[t]:
#        print("Error, should not occur")
    return nodes,clusters

  #Can be any algorithm, I use a global one, running once per level. In general clustering can be a full optimization problem in itself
  #Here I use my own devised algorithm (i.e, add a point to a cluster if after adding, all points remain with delta distrance from the new median, try to put the point in all possible clusters, if not make new one for itself)
  def clustering(self,t,beliefs,metric_order=1):
    indexes = [0]    #Need to track original idx
    delta = self.delta[t]
    nodes = [[beliefs[0]]]
    for belief in beliefs[1:]:
      condition = False
      for idx,node in enumerate(nodes):
        condition = self.calculate_membership(belief,node,delta,metric_order) #condition = True if new-member added
#        pdb.set_trace()
        if condition:
            node.append(belief)
            nodes[idx] = node
            indexes.append(idx)
            break
      if not condition:
          indexes.append(len(nodes))
          nodes.append([belief])
          
    return nodes,indexes
#Can still try RadiusNeighborsTransformer sklearn. Not useful on 2nd thought, cause i'd need running fit(X), then querying will take time too
  # Follows logic explained above
  def calculate_membership(self,belief,node,delta,order):
#    Ok even for p=2, mean is not equidistant
    cond = True
    lis = node + [belief]
    mean = np.mean(lis,axis=0)
#    val1 = cdist(lis,mean.reshape(1,-1),'minkowski',p=order)
    val = fastdist.vector_to_matrix_distance(mean,np.array(lis),fastdist.minkowski)
#    d = val[0]
#    for v in val[1:]:
#      if v != d:
#        print(val)  #https://math.stackexchange.com/questions/1296628/circumcenter-of-tetrahedron-in-4d
#        print('Test\n')
    if np.max(val) > delta:
      cond = False

    return cond            

  #Getting valid candidate mean-point
  def mean(self,lis):
    #Conceptully, don't need to find the discreted-mean, can use the direct mean with delta-level clustering
    #Since epsilon or 1/m level-discretization or approx. inference should be suffcient, if not, it helps reduce the no. of clusters
    #But for making delta-level clustering, no need to find inner-cover over discretized belief-space, simple mean is ok
    mean = np.mean(lis,axis=0)
    return mean

if __name__ == "__main__":

    S=2
    H = 5
    counter_per_sa = 5
    n_beliefs = 10
    alg = AlgoPOMDP((S,H),[counter_per_sa]*H,[0.7]*H)
        
    belief = np.random.choice(range(1000),size=(n_beliefs,S))
    z = np.sum(belief,axis=1)
    z_broadcast = z[:,np.newaxis]
    beliefs = belief/z_broadcast
    print(beliefs.shape,sum(beliefs[np.random.choice(n_beliefs)]))
    nodes,idxs = alg.clustering(0,beliefs)
    for node in nodes:
        print('\n new-node with length ',len(node))

    print('\n Total Cluster Length: ',len(nodes))
    print('\n idx are: ',idxs)

    #Testing the other one
    nodes,idxs = alg.clusteringNN(0,beliefs)
    for node in nodes:
        print('\n new-node with length ',len(node))

    print('\n Total Cluster Length: ',len(nodes))
    print('\n idx are: ',idxs)
    
    ##Testing NN inference
    alg.NN_simple(beliefs[-2],0)
    NN(beliefs[-2],alg.counter[0],alg.S)
