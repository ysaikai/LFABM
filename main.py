'''
An agent-based model of local food systems

Based on WoflSheep model created by Project Mesa.
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
because it's just natural to use a sequence of integers. Plus, some models
may not use a spatial grid.

[Initial parameters]
They are scattered across the scripts. It may be cleaner to specify
all the initial parameter values in an external text file.

[Entry]
At each period, there's a random entry of a new seller
Price: the average of the existing prices (so far, including Wal-Mart)
Calculate the profitability of each cell at each period, as if it entered
Set a threshold, Generate weights for those above it, Enter accordingly

Profitability
  The profitability (pi) for each cell
  2-dimensional tuple is reduced into 1-dimensional.
  [0] - (0,0), [1] - (0,1),..., [width] - (1,0),... and so on.
'''

import numpy as np
import random
from collections import defaultdict

from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

import debug
import csa


class Trade(Model):
  verbose = False # Print-monitoring
  '''
  Parameters
  '''
  height = 20
  width = 20
  ini_buyers = 200
  ini_sellers = 50
  ini_cash = 100
  num_w = 1 # Number of Wal-Mart
  trust_w = 0.5
  costs = 0.05 * ini_buyers
  entryOn = 0  # Toggle Entry on and off (for quicker running)
  mktresearch = False
  csa = 0
  csa_length = 26 # CSA contract length

  '''
  Debugging
  '''
  sellerDebug = 1  # Toggle for seller variable information
  buyerDebug = 0   # Toggle for buyer variable information


  def __init__(self, height=height, width=width, ini_buyers=ini_buyers, ini_sellers=ini_sellers):
    self.height = height
    self.width = width
    self.ini_buyers = ini_buyers
    self.ini_sellers = ini_sellers

    self.schedule = RandomActivationByType(self)
    self.grid = MultiGrid(self.height, self.width, torus=True)
    self.datacollector = DataCollector(
      {"Sellers": lambda m: m.schedule.get_type_count(Seller),
      "Buyers": lambda m: m.schedule.get_type_count(Buyer)})

    '''Initialization'''
    self.cnt = 0 # To count steps
    prices = {}
    for i in range(ini_sellers):
      # prices[i] = 2
      prices[i] = np.random.rand() + 1 # 1.0 - 2.0
    min_price = min(prices.values())
    for i in range(self.num_w):
      prices[i] = min_price*0.9
    self.prices = prices
    self.buyers = {} # Dictionary of buyer instances
    self.sellers = {} # Dictionary of seller instances
    self.pi = [0] * (height * width) # Profitability

    '''Create buyers'''
    for i in range(self.ini_buyers):
      # It seems coincidence in the same cell is allowed
      x = np.random.randint(self.width)
      y = np.random.randint(self.height)

      a = 1
      trust = {}
      for j in range(ini_sellers):
        trust[j] = np.random.rand()
      for j in range(self.num_w):
        trust[j] = self.trust_w
      b = 1

      buyer = Buyer(i, self.grid, (x, y), True, a, trust, b)
      self.buyers[i] = buyer # Dictionary key is an integer
      self.grid.place_agent(buyer, (x, y))
      self.schedule.add(buyer)

    '''Create sellers'''
    for i in range(self.ini_sellers):
      x = np.random.randint(self.width)
      y = np.random.randint(self.height)

      cash = self.ini_cash
      costs = self.costs
      price = self.prices[i]
      w = False
      if i < self.num_w: w = True

      seller = Seller(i, self.grid, (x, y), True, cash, costs, price, w)
      self.sellers[i] = seller # a dictionary key is an integer
      self.grid.place_agent(seller, (x, y))
      self.schedule.add(seller)

    self.running = True

  def step(self):
    self.cnt += 1

    '''Initialize the profitability'''
    self.pi = [0] * (self.height * self.width)

    '''initialize the adjacent sales'''
    for obj in self.sellers.values():
      obj.sales = 0

    '''Add customer list'''
    for obj in self.sellers.values():
      obj.customers[self.cnt] = []

    self.schedule.step()
    self.datacollector.collect(self)
    if self.verbose:
      print([self.schedule.time,
        self.schedule.get_type_count(Seller),
        self.schedule.get_type_count(Buyer)])

    '''
    Entry
      Determine the most profitable position and whether to enter
      Threshold: the fixed costs
    '''
    if (self.entryOn and self.mktresearch):
      opt = max(self.pi)
      opt_pos = self.pi.index(opt)

      if opt >= self.costs:
        x = opt_pos // self.width
        y = opt_pos % self.width
        cash = self.ini_cash
        costs = self.costs
        sid = max([seller.sid for seller in self.sellers.values()]) + 1
        price = np.mean([seller.price for seller in self.sellers.values()])
        w = False
        seller = Seller(sid, self.grid, (x, y), True, cash, costs, price, w)
        self.sellers[sid] = seller
        self.grid.place_agent(seller, (x, y))
        self.schedule.add(seller)
        self.prices[sid] = price

        if self.entryOn:
          print("\n**********\n", "Entry!!", "\n**********")
          print("sid:", sid, ", Cell:(" + str(x) + ", " + str(y) + ")")
        self.mktresearch = False

    '''
    Debugging
    '''
    '''Display trust levels'''
    if self.buyerDebug:
      debug.buyers(self.buyers)
    '''Display seller information'''
    if self.sellerDebug:
      debug.sellers(self.cnt, self.num_w, self.sellers, self.buyers)


class Buyer(Agent):
  '''
  bid: buyer unique id
  a: a coefficient on trust
  trust: a vector of trust levels in the producers
  b: a coefficient on distance, i.e. local_affinity
  '''
  def __init__(self, bid, grid, pos, moore, a, trust, b):
    self.bid = bid
    self.grid = grid
    self.pos = pos
    self.moore = moore
    self.a = a
    self.trust = trust
    self.b = b
    self.csa = False

  def step(self, model):
    def util(i):
      '''
      utility = a*trust - b*d - p
      model.prices: the price vector with sid as its indices
      model.sellers[sid]: a seller object, containing attribute pos=[x][y]
      to calculate the distance from her

      Since utils are used to calculate probability weights, they should be
      positive. So, they are exponentiated.
      '''
      a = self.a
      trust = self.trust[i]
      pos = model.sellers[i].pos
      d = abs(pos[0] - self.pos[0]) + abs(pos[1] - self.pos[1])
      b = self.b
      p = model.sellers[i].price

      return np.exp(a*trust - b*d - p)

    # A bad coding, CSA decision covers too wide. Pardon me for awhile!
    if self.csa == False:
      '''
      Buyer chooses a seller at weighted random. Weights are normalized utils.
      '''
      sid_alive = []
      utils = []
      for sid, seller in model.sellers.items():
        if seller.alive:
          sid_alive.append(sid)
          utils.append(util(sid))

      weights = utils / sum(utils)
      choice = np.random.choice(sid_alive, p=weights)
      model.sellers[choice].sales += 1
      model.sellers[choice].customers[model.cnt].append(self.bid)

      '''
      Update the trust
        Up on each purchase and down without purchase (forgetting)
        Building stops at ub, and forgetting stops at lb
        No update for Wal-Mart
      '''
      lb = 1 # Lower bound
      ub = 10 # Upper bound
      up = 1.05 # Up rate
      down = 0.9 # Down rate

      for sid, seller in model.sellers.items():
        if seller.w == False:
          if sid == choice:
            self.trust[sid] = self.trust[sid] * up
          else:
            self.trust[sid] = self.trust[sid] * down

          if self.trust[sid] > ub:
            self.trust[sid] = ub
          elif self.trust[sid] < lb:
            self.trust[sid] = lb

      '''
      Profitability & Entry
        x - row, y - column (the other way around!?)
        Allow a position already occupied by an existing seller
        Conduct market research every now and then.
      '''
      if (model.entryOn and model.cnt % 8 == 0):
        cash = model.ini_cash
        costs = model.costs
        price = np.mean([seller.price for seller in model.sellers.values()])
        w = False
        sid = max([seller.sid for seller in model.sellers.values()]) + 1

        for j in range(len(model.pi)):
          x = j // model.width
          y = j % model.width
          seller = Seller(sid, model.grid, (x,y), True, cash, costs, price, w)
          model.sellers[sid] = seller
          self.trust[sid] = lb # Set at the lower bound
          sid_alive.append(sid)
          utils.append(util(sid))
          weights = utils / sum(utils)
          choice = np.random.choice(sid_alive, p=weights)
          if choice == sid:
            model.pi[j] += 1
          # remove the dummy seller
          del model.sellers[sid]
          # del self.trust[sid]
          del sid_alive[-1]
          del utils[-1]

        model.mktresearch = True


class Seller(Agent):
  '''
  sid: seller unique id
  cash: liquidity level
  costs: fixed costs, working as the threshold of breakeven
  w: boolean for Wal-Mart
  '''
  underCutPercent = 0.99     # Percent that seller under cuts neighbors price
  priceAdjustDown = 0.95     # Percent they reduce price (in absence of neighbors)
  priceAdjustUp = 1.10       # Percent they increase price (in absence of neighbors)
  obsRadius = 1              # How far seller can observe prices (in cell units)
  idealPremium = 0.50        # Premium above costs that reflects sellers ideal profits

  def __init__(self, sid, grid, pos, moore, cash, costs, price, w):
    self.sid = sid
    self.grid = grid
    self.pos = pos
    self.moore = moore
    self.cash = cash
    self.costs = costs
    self.price = price
    self.w = w
    self.idealProfits = costs*Seller.idealPremium
    self.alive = True
    self.sales = 0 # Number of customers at the adjacent period
    self.csa = False
    self.cnt_csa = 0
    self.csa_list = []
    self.customers = {}

  def step(self, model):
    # Cash balance
    if self.csa == False:
      profit = self.sales*self.price - self.costs
      self.cash += profit

    # Insolvency (Wal-Mart is immortal)
    if (self.w == False and self.cash < 0):
      self.alive = False
      model.grid._remove_agent(self.pos, self)
      model.schedule.remove(self)
      del model.sellers[self.sid]

    if self.alive:
      if model.csa:
        '''Become a CSA farmer'''
        if (self.sales*self.price > self.costs*1.5 and self.csa == 0):
          for customer in self.customers[model.cnt]:
            model.buyers[customer].csa = True
          self.cash += profit*(model.csa_length - 1)
          self.csa = True
          self.cnt_csa = 0
          self.csa_list = self.customers[model.cnt]
          self.alive = False # Temporarily disappears from buyers' eyes

      ''' Price Adjustment Downwards'''
      # React if not walmart and sales were too low (ie didn't cover all the costs)
      if (self.csa == 0 and not self.w and np.random.rand() > self.price*self.sales/self.costs):
        minNeighborPrice = 100000
        for neighbor in self.grid.get_neighbors(self.pos,True,False,Seller.obsRadius):
          if (isinstance(neighbor, Seller) and not neighbor.w and neighbor.price < minNeighborPrice):
            minNeighborPrice = neighbor.price

        if (minNeighborPrice <= self.price):
          # If a lower price nearby they undercut their neighbors
          model.prices[self.sid] = Seller.underCutPercent*minNeighborPrice
          self.price = model.prices[self.sid]
        else:
          # Keep their price the same for now (otherwise it's very unstable)
          #model.prices[self.sid] = Seller.priceAdjustDown*model.prices[self.sid]
          #self.price = model.prices[self.sid]
          model.prices[self.sid] = model.prices[self.sid]

      ''' Price Adjustment Upwards'''
      # React if not walmart and sales were high (but below ideal revenue)
      profits = self.price*self.sales-self.costs
      if (self.csa == 0 and not self.w and np.random.rand() > 1 - profits/self.idealProfits):
        maxNeighborPrice = 0
        for neighbor in self.grid.get_neighbors(self.pos,True,False,Seller.obsRadius):
          if (isinstance(neighbor, Seller) and not neighbor.w and neighbor.price > maxNeighborPrice):
            maxNeighborPrice = neighbor.price

        if (maxNeighborPrice >= self.price):
          # If a not the lowest price nearby they just undercut their neighbors
          model.prices[self.sid] = Seller.underCutPercent*maxNeighborPrice
          self.price = model.prices[self.sid]
        else:
          # Keep their price the same for now (otherwise it's very unstable)
          #model.prices[self.sid] = Seller.priceAdjustUp*model.prices[self.sid]
          #self.price = model.prices[self.sid]
          model.prices[self.sid] = model.prices[self.sid]

    if self.csa:
      self.cnt_csa += 1
    if self.cnt_csa >= model.csa_length:
      for customer in self.csa_list:
        model.buyers[customer].csa = False
      self.csa = False
      self.alive = True


class RandomActivationByType(RandomActivation):
  '''
  Activate every agent once per step. The order is reshuffled at each step.
  by_type: If True, activate all the agents of a certain type first in random
  order, and then do the same for the next type.

  Assumes that all agents have a step(model) method.
  '''
  agent_types = defaultdict(list) # "Buyer" or "Seller"

  def __init__(self, model):
    super().__init__(model)
    self.agent_types = defaultdict(list)

  def add(self, agent):
    '''
    Add an agent to the schedule
      It seems that self.agents are NOT a list of the raw agent instances,
      which are passed to, i.e. each list element semms to be processed
      into a dictionary or the like (but, not sure yet).
      Again, this is a list, whose indices are unrelated with unique id.
      This is why we retain a separate dictionary of seller instances,
      whose keys correspond to their unique id (sid).
      self.agent_types[] seems similar, a list of the certain type,
      while self.agents are a list of all the agents. i.e. self.agent_types[]
      has a hierarchy.
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

  def step(self, by_type=True):
    '''
    Executes the step of each agent, one at a time, in random order.
    '''
    if by_type:
      for agent_class in self.agent_types:
        self.step_type(agent_class)
      self.steps += 1
      self.time += 1
    else:
      super().step()

  def step_type(self, type):
    '''
    Shuffle order and run all agents of a given type.

    Args:
      type: Class object of the type to run.
    '''
    agents = self.agent_types[type]
    random.shuffle(agents)
    for agent in agents:
      agent.step(self.model)

  def get_type_count(self, a_type):
    '''
    Returns the current number of agents of certain type in the queue.
    '''
    return len(self.agent_types[a_type])
