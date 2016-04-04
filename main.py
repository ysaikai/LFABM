'''
An agent-based model of local food systems

Based on WoflSheep model creted by Project Mesa.
https://github.com/projectmesa/mesa/tree/master/examples/WolfSheep
'''

'''
Notes

[Activation]
1. Buyers learn the prices, choose a seller, and buy one unit
2. Sellers check the cash balance and die or post prices

[Unique identifier]
Many examples use pos as identifier (x-y position is unique on each grid).
Here, scalar bid & sid are created to make them independent of spaces
because it's just nuatural to use a sequence of integers Plus, some models
may not use a spatial grid.

[Initial parameters]
They are scattered across this script. It may be cleaner to specify
all the initial parameter values in an external text file.
'''

import numpy as np
import random
from collections import defaultdict

from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation
# from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector


class Trade(Model):
  height = 20
  width = 20
  ini_buyers = 100
  ini_sellers = 50
  verbose = False # Print-monitoring

  def __init__(self, height=20, width=20, ini_buyers=ini_buyers, ini_sellers=ini_sellers):
    self.height = height
    self.width = width
    self.ini_buyers = ini_buyers
    self.ini_sellers = ini_sellers

    self.schedule = RandomActivationByBreed(self)
    # self.schedule = BaseScheduler(self)
    self.grid = MultiGrid(self.height, self.width, torus=True)
    self.datacollector = DataCollector(
      {"Sellers": lambda m: m.schedule.get_type_count(Seller),
      "Buyers": lambda m: m.schedule.get_type_count(Buyer)})

    # '''
    # Generate a matching matrix
    #   (for now) just random 0 & 1
    #   Sellers are on rows. Buyers on columns.
    #   Wal-Mart has 1s for every buyers (correct?)
    # '''
    # self.match = np.random.randint(2, size=(ini_sellers-1, ini_buyers))
    # self.match = np.append(match, [np.ones(ini_buyers,dtype=np.int)], axis=0)

    '''
    Price
      To let buyers access the prices, define as a Trade class attribute
      instead of an individual seller's attribute.
      (arbitrary) range from 0 to 2
    '''
    self.prices = 2 * np.random.rand(ini_sellers)

    '''Create buyers'''
    for i in range(self.ini_buyers):
      '''
      What happens if two pos coincide? Since it manages to run, I guess,
      the grid module doesn't rule out such a senario. For now, leave it
      as it is.
      '''
      x = random.randrange(self.width)
      y = random.randrange(self.height)

      '''income > max(price) to make every sellers affordable'''
      income = 10 * np.random.rand() + max(self.prices)
      a = np.random.rand() # a coefficient on trust
      '''
      Trust
        a vector of trust levels in the sellers
        (arbitrary) uniform[0,2]
        Wal-Mart has 0
      '''
      trust = 2 * np.random.rand(ini_sellers - 1)
      trust = np.append(trust,0) # 0 trust in Wal-Mart
      b = 0.02 * np.random.rand() # a coefficient on distance

      buyer = Buyer(i, self.grid, (x, y), True, a, trust, income, b)
      self.grid.place_agent(buyer, (x, y))
      self.schedule.add(buyer)

    '''Create sellers'''
    self.sellers = [] # a list of seller objects
    for i in range(self.ini_sellers):
      # the same concern of coincident positions as above
      x = random.randrange(self.width)
      y = random.randrange(self.height)

      cash = 100 # initial cash balance
      # relative to ini_buyers, implying the required market share
      costs = 0.1 * ini_buyers
      price = self.prices[i]
      w = False
      if i == self.ini_sellers - 1: w = True # the last is Wal-Mart

      seller = Seller(i, self.grid, (x, y), True, cash, costs, price, w)
      '''
      To have instant access to seller attributes, create a list of seller
      objects. If it turns out a waste of memory (esp. with a big simulation)
      I guess we may loop the scheduler or the grid to access a specific
      seller. But, then, it would be a waste of computation, I suppose...
      '''
      self.sellers.append(seller)
      self.grid.place_agent(seller, (x, y))
      self.schedule.add(seller)

    self.running = True

  def step(self):
    for obj in self.sellers:
      obj.sales = 0 # initialize the adjacent sales

    self.schedule.step()
    self.datacollector.collect(self)
    if self.verbose:
      print([self.schedule.time,
        self.schedule.get_type_count(Seller),
        self.schedule.get_type_count(Buyer)])

# Not yet worked on
  def run_model(self, step_count=200):
    if self.verbose:
      print('Initial number sellers: ',
        self.schedule.get_type_count(Seller))
      print('Initial number buyers: ',
        self.schedule.get_type_count(Buyer))

    for i in range(step_count):
      self.step()

    if self.verbose:
      print('')
      print('Final number sellers: ',
        self.schedule.get_type_count(Seller))
      print('Final number buyers: ',
        self.schedule.get_type_count(Buyer))


class Buyer(Agent):
  '''
  bid: buyer unique id
  a: a coefficient on trust
  trust: a vector of trust levels in the producers
  income: wealth level (for now, just set high enough)
  b: a coefficient on distance, i.e. local_affinity
  '''
  def __init__(self, bid, grid, pos, moore, a, trust, income, b):
    self.bid = bid
    self.grid = grid
    self.pos = pos
    self.moore = moore
    self.a = a
    self.trust = trust
    self.income = income
    self.b = b

  def step(self, model):
    '''Buyer's optimization problem is to choose the best seller'''
    def util(i):
      '''
      utility = a*trust - b*d - p
      model.prices: the price vector with sid as its indices
      model.sellers[sid]: a seller object, containing attribute pos=[x][y]
      to calculate the distance from her
      '''
      a = self.a
      trust = self.trust[i]
      pos = model.sellers[i].pos
      d = abs(pos[0] - self.pos[0]) + abs(pos[1] - self.pos[1])
      b = self.b
      p = model.prices[i]

      return a*trust - b*d - p

    choice = max(range(model.ini_sellers), key=util)
    model.sellers[choice].sales += 1


class Seller(Agent):
  '''
  sid: seller unique id
  cash: liquidity level
  costs: fixed costs, working as the threshold of breakeven
  w: boolean for Wal-Mart
  '''
  def __init__(self, sid, grid, pos, moore, cash, costs, price, w=False):
    self.sid = sid
    self.grid = grid
    self.pos = pos
    self.moore = moore
    self.cash = cash
    self.costs = costs
    self.price = price
    self.w = w
    self.sales = 0 # the number of costomers at the adjacent period

  def step(self, model):
    # Wal-Mart is immortal with unchanged cash balances
    if self.w==False:
      '''The cash balance changes by #sales - costs (#sales = #buyers)'''
      self.cash += self.sales - self.costs

    # Insolvency
    if self.cash < 0:
      model.grid._remove_agent(self.pos, self)
      model.schedule.remove(self)

    # Post a new price
    else:
      # For now, it is fixed and do nothing
      model.prices[self.sid] = model.prices[self.sid]


# Haven't been touched yet
#
class RandomActivationByBreed(RandomActivation):
  '''
  A scheduler which activates each type of agent once per step, in random
  order, with the order reshuffled every step.

  This is equivalent to the NetLogo 'ask breed...' and is generally the
  default behavior for an ABM.

  Assumes that all agents have a step(model) method.
  '''
  agent_types = defaultdict(list)

  def __init__(self, model):
    super().__init__(model)
    self.agent_types = defaultdict(list)

  def add(self, agent):
    '''
    Add an Agent object to the schedule

    Args:
      agent: An Agent to be added to the schedule.
    '''

    self.agents.append(agent)
    agent_class = type(agent)
    self.agent_types[agent_class].append(agent)

  def remove(self, agent):
    '''
    Remove all instances of a given agent from the schedule.
    '''

    while agent in self.agents:
      self.agents.remove(agent)

    agent_class = type(agent)
    while agent in self.agent_types[agent_class]:
      self.agent_types[agent_class].remove(agent)

  def step(self, by_breed=True):
    '''
    Executes the step of each agent breed, one at a time, in random order.

    Args:
      by_breed: If True, run all agents of a single breed before running
            the next one.
    '''
    if by_breed:
      for agent_class in  self.agent_types:
        self.step_breed(agent_class)
      self.steps += 1
      self.time += 1
    else:
      super().step()

  def step_breed(self, breed):
    '''
    Shuffle order and run all agents of a given breed.

    Args:
      breed: Class object of the breed to run.
    '''
    agents = self.agent_types[breed]
    random.shuffle(agents)
    for agent in agents:
      agent.step(self.model)

  def get_type_count(self, a_type):
    '''
    Returns the current number of agents of certain breed in the queue.
    '''
    return len(self.agent_types[a_type])
