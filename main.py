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
# import csa
# import embeddedness as ebd


class Trade(Model):
  verbose = False # Print-monitoring

  '''Parameters'''
  height = 20
  width = 20
  ini_buyers = 300
  ini_sellers = 50
  ini_cash = 100
  num_w = 1 # Number of Wal-Mart
  trust_w = 0.5
  costs = 0.03 * ini_buyers
  mktresearch = False
  csa = 0
  csa_length = 26 # CSA contract length
  '''
  Entry mode
    0: No entry
    1: Full market research
    2: Whenever Avg cash balance > ini_cash with a random position
  '''
  entry = 0
  entryFrequency = 8

  '''Debugging'''
  sellerDebug = True
  buyerDebug = False
  utilweightDebug = 0


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
    self.cnt = 0 # Period counter
    self.buyers = {} # Dictionary of buyer instances
    self.sellers = {} # Dictionary of seller instances
    self.sid_alive = []
    self.pi = [0] * (height * width) # Profitability

    self.lb = 1 # Lower bound
    self.ub = 10000 # Upper bound (in effect, unbounded)
    self.up = 1.05 # Up rate
    self.down = 0.99 # Down rate

    prices = {}
    for i in range(ini_sellers):
      # prices[i] = 2
      prices[i] = np.random.rand() + 1 # 1.0 - 2.0
    min_price = min(prices.values())
    for i in range(self.num_w):
      prices[i] = min_price*0.9
    self.prices = prices

    e = {} # Embeddedness
    for i in range(ini_sellers):
      e[i] = 0.8*np.random.rand() + 0.2 # 0.2 - 1.0
    for i in range(self.num_w):
      e[i] = 0
    self.e = e

    '''Create buyers'''
    for i in range(self.ini_buyers):
      # It seems coincidence in the same cell is allowed
      x = np.random.randint(self.width)
      y = np.random.randint(self.height)

      α = 2
      trust = {}
      # β = 5*np.random.rand()
      β = 3
      for j in range(ini_sellers):
        trust[j] = np.random.rand()
      for j in range(self.num_w):
        trust[j] = self.trust_w
      γ = 1

      buyer = Buyer(i, self.grid, (x, y), True, α, trust, β, γ)
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
      if i < self.num_w:
        w = True
      e = self.e[i]

      seller = Seller(i, self.grid, (x, y), True, cash, costs, price, w, e)
      self.sellers[i] = seller
      self.grid.place_agent(seller, (x, y))
      self.schedule.add(seller)

    self.running = True

  def step(self):
    '''Initialization'''
    self.cnt += 1
    self.sid_alive = [] # Excluding Wal-Mart

    for sid, seller in self.sellers.items():
      '''Adjacent sales'''
      seller.sales = 0

      '''Customer list'''
      seller.customers[self.cnt] = []

      '''A list of living sellers (excluding Wal-Mart)'''
      if (seller.alive and seller.w == False):
        self.sid_alive.append(sid)
    '''
    Entry=1
      Initialize the profitability vector
    Entry=2
      Calculate the average cash balance (scalar)
    '''
    if self.entry == 1:
      self.pi = [0] * (self.height * self.width)
    elif self.entry == 2:
      total_cash = 0
      cnt_seller = 0
      total_cash = sum([self.sellers[sid].cash for sid in self.sid_alive])
      self.avg_cash = total_cash / len(self.sid_alive)


    '''
    Entry
      Entry=1
        Determine the most profitable position and whether to enter
        Threshold: the fixed costs
      Entry=2
        Enter whenever Avg cash balance > ini_cash
    '''
    entry_on = False

    if (self.entry == 1 and self.mktresearch):
      opt = max(self.pi)
      opt_pos = self.pi.index(opt)

      if opt >= self.costs:
        x = opt_pos // self.width
        y = opt_pos % self.width
        entry_on = True

    elif (self.entry == 2 and self.avg_cash > self.ini_cash):
      x = np.random.randint(self.width)
      y = np.random.randint(self.height)
      entry_on = True

    if entry_on:
      cash = self.ini_cash
      costs = self.costs
      w = False
      price = np.mean([self.sellers[sid].price for sid in self.sid_alive])
      # e = np.random.choice([self.sellers[sid].e for sid in self.sid_alive])
      e = np.random.rand()
      sid = max([seller.sid for seller in self.sellers.values()]) + 1
      self.sid_alive.append(sid)
      seller = Seller(sid, self.grid, (x, y), True, cash, costs, price, w, e)
      self.sellers[sid] = seller
      self.sellers[sid].customers[self.cnt] = []
      for buyer in self.buyers.values():
        buyer.trust[sid] = self.lb
      self.grid.place_agent(seller, (x, y))
      self.schedule.add(seller)
      self.prices[sid] = price

      if self.entry == 1:
        print("\n**********\n", "Entry!!", "\n**********")
        print("sid:", sid, ", Cell:(" + str(x) + ", " + str(y) + ")")

      self.mktresearch = False


    '''Move'''
    self.schedule.step()
    self.datacollector.collect(self)
    if self.verbose:
      print([self.schedule.time,
        self.schedule.get_type_count(Seller),
        self.schedule.get_type_count(Buyer)])


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
  α: coefficient on trust
  trust: vector of trust levels in the producers
  β: coefficient on embeddedness
  γ: coefficient on distance
  '''
  def __init__(self, bid, grid, pos, moore, α, trust, β, γ):
    self.bid = bid
    self.grid = grid
    self.pos = pos
    self.moore = moore
    self.α = α
    self.trust = trust
    self.β = β
    self.γ = γ
    self.csa = False

  def step(self, model):
    def util(i):
      '''
      utility = α*trust - γ*d - p
      model.prices: the price vector with sid as its indices
      model.sellers[sid]: a seller object, containing attribute pos=[x][y]
      to calculate the distance from her

      Since utils are used to calculate probability weights, they should be
      positive. So, they are exponentiated.
      '''
      trust = self.trust[i]
      e = model.sellers[i].e
      pos = model.sellers[i].pos
      d = abs(pos[0] - self.pos[0]) + abs(pos[1] - self.pos[1])
      p = model.sellers[i].price

      return self.α*trust + self.β*e - self.γ*d - p

    if self.csa == False:
      '''
      Buyer chooses a seller at weighted random.
        1. Calculate raw utils
        2. Scale them into u' = (u-min(u))*κ/(max(u)-min(u)) - δ
          κ: the range, δ: min(u'), so max(u') = min(u') + κ
        3. Exponentiate them
        4. Weights = Relative sizes of them
      '''
      sid_alive = []
      utils = []
      for sid, seller in model.sellers.items():
        if seller.alive:
          sid_alive.append(sid)
          utils.append(util(sid))
      # Transform into an appropriate interval
      Δ = max(utils) - min(utils)
      utils = np.array([(u - min(utils))*15/Δ - 5 for u in utils])
      # Exponentiate
      utils = np.exp(utils)
      weights = utils / np.sum(utils)
      choice = np.random.choice(sid_alive, p=weights)
      model.sellers[choice].sales += 1
      model.sellers[choice].customers[model.cnt].append(self.bid)

      # Debugging
      if model.utilweightDebug:
        debug.util_weight(self.bid, utils, weights)

      '''
      Update the trust
        'up' reflects good experience with a seller. So, without a random
        negative shock (eg misbehavior), it only increases on each purchase.
        'down' reflects gradual fogetting without purchase and interaction.
        The more embedded a seller, the greater and lower chance of up and
        down respectively. Recall e∈(0,1).
        No update for Wal-Mart
      '''
      lb = model.lb # Lower bound
      ub = model.ub # Upper bound
      up = model.up # Up rate
      down = model.down # Down rate

      for sid, seller in model.sellers.items():
        if seller.w == False:
          up_chance = (np.random.rand() < (0.5 + seller.e/2))
          # up_chance = 1
          if sid == choice:
            self.trust[sid] = self.trust[sid] * (up * up_chance)
          else:
            self.trust[sid] = self.trust[sid] * (down * (1-up_chance))

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
      if (model.entry == 1 and model.cnt % model.entryFrequency == 0):
        cash = model.ini_cash
        costs = model.costs
        price = np.mean([seller.price for seller in model.sellers.values()])
        w = False
        e = np.random.rand()
        sid = max([seller.sid for seller in model.sellers.values()]) + 1

        for j in range(len(model.pi)):
          x = j // model.width
          y = j % model.width
          seller = Seller(sid, model.grid, (x,y), True, cash, costs, price, w, e)
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

  κ = 15

  def __init__(self, sid, grid, pos, moore, cash, costs, price, w, e):
    self.sid = sid
    self.grid = grid
    self.pos = pos
    self.moore = moore
    self.cash = cash
    self.costs = costs
    self.price = price
    self.w = w
    self.e = e
    self.idealProfits = costs*Seller.idealPremium
    self.alive = True
    self.sales = 0 # Number of customers at the adjacent period
    self.profits = 0
    self.csa = False
    self.cnt_csa = 0
    self.csa_list = []
    self.customers = {}

  def step(self, model):
    # Cash balance
    if self.csa == False:
      # self.profits = self.sales*(self.price - self.κ*self.e) - self.costs
      self.profits = self.sales*(self.price) - self.costs - self.κ*self.e
      self.cash += self.profits

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
          self.cash += self.profits*(model.csa_length - 1)
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
      if (self.csa == 0 and not self.w and np.random.rand() > 1 - self.profits/self.idealProfits):
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
