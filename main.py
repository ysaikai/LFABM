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

'''

import numpy as np
import random
from collections import defaultdict

from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


class Trade(Model):
  '''
  Parameters
  '''
  height = 20
  width = 20
  ini_buyers = 100
  ini_sellers = 50
  verbose = False # Print-monitoring
  entryOn = False  # Toggle Entry on and off (for quicker running)

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

    '''
    Profitability
      The profitability (pi) for each cell
      2-dimensional tuple is reduced into 1-dimensional.
      [0] - (0,0), [1] - (0,1),..., [width] - (1,0),... and so on.
    '''
    self.pi = [0] * (height * width)

    self.cnt = 0 # a counter for debugging

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
      Wal-Mart has the lowest price, 90% of the local lowest
    '''
    # self.prices = 2 * np.random.rand(ini_sellers - 1)
    # self.prices = [2] * (ini_sellers - 1)
    # self.prices.append() = np.append(self.prices, min(self.prices)*0.9)
    self.prices = {}
    for i in range(ini_sellers - 1):
      self.prices[i] = 2
    self.prices[ini_sellers - 1] = min(self.prices.values())*0.9

    # '''
    # Scoreboard
    #   Each tally of sales, used as a popularity ranking
    # '''
    # self.sb = np.zeros(self.ini_sellers, dtype=np.int)

    '''Create buyers'''
    self.buyers = {} # a dictionary of buyer instances
    for i in range(self.ini_buyers):
      '''
      What happens if two pos coincide? Since it manages to run, I guess,
      the grid module doesn't rule out such a senario. For now, leave it
      as it is.
      '''
      x = random.randrange(self.width)
      y = random.randrange(self.height)

      '''income > max(price) to make every sellers affordable'''
      income = 10 * np.random.rand() + max(self.prices.values())
      # a = np.random.rand() # a coefficient on trust
      a = 1 # (for now) set = 1
      '''
      Trust
        A vector of trust levels in the sellers (1s and 0)
        Trust encapsulates all the 'quality' information about each seller,
        which helps a buyer make a decision. e.g. goods & service quality and
        character.
      '''
      # trust = 2 * np.random.rand(ini_sellers - 1)
      trust = {}
      for j in range(ini_sellers - 1):
        trust[j] = 1
      trust[ini_sellers - 1] = 0 # 0 trust in Wal-Mart
      b = 0.02 * np.random.rand() # a coefficient on distance

      buyer = Buyer(i, self.grid, (x, y), True, a, trust, income, b)
      self.buyers[i] = buyer # a dictionary key is an integer
      self.grid.place_agent(buyer, (x, y))
      self.schedule.add(buyer)

    '''Create sellers'''
    self.sellers = {} # a dictionary of seller instances
    for i in range(self.ini_sellers):
      # the same concern of coincident positions as above
      x = random.randrange(self.width)
      y = random.randrange(self.height)

      cash = 100 # initial cash balance
      # relative to ini_buyers, implying the required market share
      costs = 0.1 * ini_buyers
      price = self.prices[i]
      w = False
      if i == self.ini_sellers - 1:
        w = True # the last is Wal-Mart

      seller = Seller(i, self.grid, (x, y), True, cash, costs, price, w)
      '''
      To have instant access to seller attributes, create a list of seller
      objects. If it turns out a waste of memory (esp. in a big simulation)
      I guess we may loop the scheduler or the grid to access a specific
      seller. But, then, it would be a waste of computation...
      '''
      self.sellers[i] = seller # a dictionary key is an integer
      self.grid.place_agent(seller, (x, y))
      self.schedule.add(seller)

    self.running = True

  def step(self):
    '''Initialize the profitability'''
    self.pi = [0] * (self.height * self.width)

    '''initialize the adjacent sales'''
    for obj in self.sellers.values():
      obj.sales = 0

    self.schedule.step()
    self.datacollector.collect(self)
    if self.verbose:
      print([self.schedule.time,
        self.schedule.get_type_count(Seller),
        self.schedule.get_type_count(Buyer)])

    '''
    Debugging
      Display trust levels
    '''
    print("\nBuyers trust levels Top3 (sid in brackets)")
    for obj in self.buyers.values():
      tmp = list(obj.trust.values())
      t1 = max(tmp)
      sid1 = tmp.index(t1)
      tmp.remove(t1)
      t2 = max(tmp)
      sid2 = tmp.index(t2)
      tmp.remove(t2)
      t3 = max(tmp)
      sid3 = tmp.index(t3)
      print("bid:", obj.bid, "Trust: {t1:.2f}({sid1}), {t2:.2f}({sid2}), {t3:.2f}({sid3})".format(t1=t1,t2=t2,t3=t3,sid1=sid1,sid2=sid2,sid3=sid3))

    '''
    Determine the most profitable position and whether ot enter
      Threshold: 15% market share
    '''
    if (self.entryOn and self.pi != [0] * (self.height * self.width)):
      opt = max(self.pi)
      opt_pos = self.pi.index(opt)

      if opt >= 0.15 * self.ini_buyers:
        x = opt_pos // self.width
        y = opt_pos % self.width
        cash = 100 # initial cash balance
        # relative to ini_buyers, implying the required market share
        costs = 0.1 * self.ini_buyers
        sid = max([seller.sid for seller in self.sellers.values()]) + 1
        price = np.mean([seller.price for seller in self.sellers.values()])
        w = False
        seller = Seller(sid, self.grid, (x, y), True, cash, costs, price, w)
        self.sellers[sid] = seller # a dictionary key is an integer
        self.grid.place_agent(seller, (x, y))
        self.schedule.add(seller)
        self.prices[sid] = price

        print("\n**********\n", "Entry!!", "\n**********")
        print("sid:", sid, ", Cell:(" + str(x) + ", " + str(y) + ")")


    # Debugging
    self.cnt += 1
    print("\nStep: ", self.cnt)
    for obj in self.sellers.values():
      print("sid:", obj.sid, ", Sales:", obj.sales, ", Cash:", obj.cash)


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

    '''
    Update the trust
      Up on each purchase and down without purchase (forgetting)
      Building stops at ub, and forgetting stops at lb
      No update for Wal-Mart
    '''
    lb = 1 # Lower bound
    ub = 3 # Upper bound
    up = 1.3 # Up rate
    down = 0.95 # Down rate

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

    # '''Update the scoreboard'''
    # sb = model.sb
    # # multiply each trust by (1 + normalized score)
    # self.trust = [self.trust[i]*(1 + item/sum(sb)) for i,item in enumerate(sb)]
    # model.sb[choice] += 1 # update the scoreboard

    '''
    Profitability & Entry
      x - row, y - column (the other way around!?)
      Allow a position already occupied by an existing seller
    '''
    if (model.entryOn and model.cnt % 3 == 0):
      cash = 100
      costs = 0.1 * model.ini_buyers
      price = np.mean([seller.price for seller in model.sellers.values()])
      w = False
      sid = max([seller.sid for seller in model.sellers.values()]) + 1

      for j in range(len(model.pi)):
        x = j // model.width
        y = j % model.width
        seller = Seller(sid, model.grid, (x, y), True, cash, costs, price, w)
        model.sellers[sid] = seller
        self.trust[sid] = lb
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


class Seller(Agent):
  '''
  sid: seller unique id
  cash: liquidity level
  costs: fixed costs, working as the threshold of breakeven
  w: boolean for Wal-Mart
  '''
  def __init__(self, sid, grid, pos, moore, cash, costs, price, w):
    self.sid = sid
    self.grid = grid
    self.pos = pos
    self.moore = moore
    self.cash = cash
    self.costs = costs
    self.price = price
    self.w = w
    self.alive = True
    self.sales = 0 # the number of customers at the adjacent period

  def step(self, model):
    '''The cash balance changes by #sales - costs (#sales = #buyers)'''
    self.cash += self.sales*self.price - self.costs

    # Insolvency (Wal-Mart is immortal)
    if (self.w == False and self.cash < 0):
      self.alive = False
      model.grid._remove_agent(self.pos, self)
      model.schedule.remove(self)
      del model.sellers[self.sid]

    # Post a new price
    else:
      # For now, it is fixed and do nothing
      #if self.sales 
      model.prices[self.sid] = model.prices[self.sid]


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
