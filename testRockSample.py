# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:11:04 2021



POMDPy's overiding of true state by current observation is basically setting the belief to 1 or 0 based on current obs

V IMP: THE REASON ROCK-1 WAS SKIPPED WAS BECAUSE EXIT-REWARD ON WHILE RIGHT BORDER TOO HIGH
^^ THE ALGO WORKS FINE
"""

from examples.rock_sample import RockModel
from examples.rock_sample.grid_position import GridPosition
from examples.rock_sample.rock_state import RockState
from examples.rock_sample.rock_action import RockAction
from SSRockSample import PolicyEval
from sparseModel import outsample
import numpy as np
import sys


class ModelWrapper:

  def __init__(self,model):
    self.model = model
    self.N = model.n_rows #Assuming square grid
    self.K = model.n_rocks
    self.state = self.make_state()
    
    #Testing
    legal_actions = model.get_legal_actions(self.state)
    for act in legal_actions:
      res,is_legal = model.generate_step(self.state,act)
#    print(RockAction(5+self.K-1).rock_no)  

  def make_state(self,x=None,y=None):
    if x==None:
      position = GridPosition()
    else:
      position = GridPosition(x,y)
    
    rock_states = list(np.random.randint(0,2,size=self.K)) #Random observation vector
    return RockState(position,rock_states)

  def combine_state(self,visible_state,rock_states):
    if visible_state == None:
      x = self.state.position.i
      y = self.state.position.j
    else:
      x = visible_state[0]
      y = visible_state[1]
    return RockState(GridPosition(x,y), rock_states)

  def generate_step(self,state,a):
    #During running too, I hv to replace the simulator's overriden state with true=state from previous step
    #As I do in planning automatically by feeding the combined_state from planner
    #Constrast this with gym pomdp. which only has action input, and self.state is correctly taken care by it
    result, cond = self.model.generate_step(state,RockAction(a))
    #Old comment: ^^ Think I need modify the generate_step in RS class to handle the changed true state of underlying rocks (sampled as observation in belief filter)
#    self.model.update(result) #<< Maybe not good to do here (but hv to do it for correct make_reward() output during planning)
     #^^^^ NEWEST: EVEN IN SIMULATION, THIS CREATES A LOT OF PROBLEM, ONLY UPDATE IN SIMULATION FOR SAMPLE-ACTION   
    
    #Concatenated tuple will be used as they can be hashed inside dictionaries as keys
    o = None
    pos = result.next_state.position
    if result.observation.is_good is not False:
#      o = (pos.i,pos.j) + tuple(map(int,result.next_state.rock_states)) #next-state is overidden by current observation, so its actually current observation
      o = (pos.i,pos.j) + tuple(result.next_state.rock_states)
    else:
      o = (pos.i,pos.j,None)
    #Getting old rock_states because simulator loses that info
#    rock_states = list(state.rock_states)
#    rock_no = self.model.get_cell_type(pos)
#    if self.model.actual_rock_states[rock_no] and (rock_no not in self.model.unique_rocks_sampled):
#      rock_states[rock_no] = True

    sp = tuple(state.rock_states) # Need the next 'rock_states', not the previous one
    #^^ Current-state is also modifiyed in make_reward() inside generate_step()
    r = result.reward

    return outsample(o,sp,r)

  def update_reward(self,step_result):
    rock_no = self.model.get_cell_type(step_result.next_state.position)
    if rock_no < 0:
      sys.exit(-1)
    if self.model.actual_rock_states[rock_no] and (rock_no not in self.model.unique_rocks_sampled):
      step_result.reward = 10.0
      self.model.unique_rocks_sampled.append(rock_no)
    else:
      step_result.reward = -10.0

def run(m,alg):

  state = m.model.sample_an_init_state()
  alg.update(alg.init_bel,(state.position.i,state.position.j))
  prev_rocks_sampled = list(m.model.unique_rocks_sampled)
#  print(m.model.actual_rock_states)
  
  reward = 0
  discounted_reward = 0
  discount = 1.0
  act_seq = []
  for i in range(m.model.max_steps):
    # Need to take care alg.init_bel and alg.vs is initialized (vs taken care by modelwrapper combined_state fn)
    alg.Expand(Cluster=False)
    V,bst_act = alg.Eval_VI_NoCluster()
#    print('bst act ',bst_act)
    action = RockAction(bst_act)
    #Negating the effect of simulation on unique_rocks_sampled variable
    m.model.unique_rocks_sampled = prev_rocks_sampled

    # Simulating            
    step_result, is_legal = m.model.generate_step(state, action)
    if bst_act == 4:
      m.update_reward(step_result)
    reward += step_result.reward #<<Incorrect reward generated due to modified rock_sample.make_reward(), need to calc the values myself
#    if bst_act == 4:
#      print(discount,step_result.reward)
#      pdb.set_trace()
    discounted_reward += discount * step_result.reward

    # Updating
    discount *= m.model.discount
    state = step_result.next_state

    #Accouting for the effect of real action now
    m.model.update(step_result)
    prev_rocks_sampled = list(m.model.unique_rocks_sampled)
    
#    if state.position.i == 0 and state.position.j == 4:
#      pdb.set_trace()
#    if not step_result.is_terminal or not is_legal:
    bel = list(alg.init_bel)
    if not step_result.is_terminal and bst_act >=5:
      rock_no = bst_act - 5
      obs = m.model.actual_rock_states[rock_no] and (rock_no not in m.model.unique_rocks_sampled)
      dist = state.position.euclidean_distance(m.model.rock_positions[rock_no])
      likelihood = m.model.get_sensor_correctness_probability(dist)
      if not obs:
        val0 = likelihood * (1-bel[rock_no]) #The formula's are correct chk notes, use fact that
        val1 = (1-likelihood) * bel[rock_no] # P(correct) is both P(y=1|state=1) and P(y=0|state=0)
        bel[rock_no] = val1/(val0+val1)
      else:
        val1 = likelihood * bel[rock_no]
        val0 = (1-likelihood) * (1-bel[rock_no])
        bel[rock_no] = val1/(val0+val1)

    vs = (state.position.i,state.position.j)
    alg.update(bel,vs)
      
    act_seq.append(bst_act)

    if step_result.is_terminal or not is_legal:
#      print('Terminated after episode step ' + str(i + 1))
      break

  return ([discounted_reward,act_seq,m.model.actual_rock_states])
#  print(act_seq,discounted_reward,reward)
#  m.model.draw_env()
#  print(m.model.actual_rock_states)

if __name__ == "__main__":

  writepath = './Results/'
  args = {'env': 'RockSample', 'solver': 'POMCP', 'seed': 123, 'use_tf': False, 'discount': 0.95, 'n_epochs': 10, 'max_steps': 50, 'save': False, 'test': 10, 'epsilon_start': 1.0, 'epsilon_minimum': 0.1, 'epsilon_decay': 0.99, 'epsilon_decay_step': 20, 'n_sims': 500, 'timeout': 3600, 'preferred_actions': True, 'ucb_coefficient': 3.0, 'n_start_states': 2000, 'min_particle_count': 1000, 'max_particle_count': 2000, 'max_depth': 100, 'action_selection_timeout': 60}
  model = RockModel(args)
  model.reset_for_epoch() #Ignore the int values, only underlying truth value is imp
#  model.actual_rock_states[0] = 1
#  model.actual_rock_states[1] = 1
#  model.actual_rock_states[2] = 1
#  model.actual_rock_states[3] = 1
#  model.actual_rock_states[4] = 1
#  model.actual_rock_states[5] = 0
#  model.actual_rock_states[6] = 0
#  model.actual_rock_states[7] = 1
#  model.draw_env()
  m = ModelWrapper(model)
#  m.generate_step()

  S= model.n_rocks
  A= 5 + model.n_rocks
  Z= None
  H = 4
  Disc = 0.99 # 0.99 Disc is fine, even better than .95, THE REASON ROCK-1 WAS SKIPPED WAS BECAUSE EXIT-REWARD ON WHILE RIGHT BORDER TOO HIGH              
  counter_per_sa = 5
  tolerance = [5,5,4,3,2,2]
#  tolerance = [4]*H
  epochs = 12

#  init_bel = np.random.rand(S)
#  alg = PolicyEval(init_bel,(S,A,H,Z,Disc),m, ([counter_per_sa]*H,[0.05]*H) )
#  alg.Expand(Cluster=False)
#  print("\n In Eval \n")
#  V,bst_act = alg.Eval_VI_NoCluster()
#  print(V,bst_act)

  init_bel = [0.5]*S
  alg = PolicyEval(init_bel,(S,A,H,Z,Disc),m, (tolerance,[0.05]*H) )
#  alg = PolicyEval(init_bel,(S,A,H,Z,Disc),m, ([counter_per_sa]*H,[0.05]*H) )
  
  data = []
  for i in range(epochs):
    model.reset_for_epoch()
    data.append( str(run(m,alg)) + '\n' )

  name = writepath + str(model.n_rocks) + '-' + str(tolerance) + '-' + sys.argv[1]
  file = open(name , 'w')
  file.writelines(data)
  file.close()
