'''
An agent-based model of local food systems

Based on WoflSheep model creted by Project Mesa.
https://github.com/projectmesa/mesa/tree/master/examples/WolfSheep
'''

'''
Notes
  [Activation]
  1. Buyers learn the prices, choose a seller, and buy one unit
  2. Sellers post prices
'''

import random
from collections import defaultdict

from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


class Trade(Model):
  height = 20
  width = 20
  ini_buyers = 100
  ini_sellers = 50
  verbose = False # Print-monitoring

  def __init__(self, height=20, width=20, ini_buyers=100, ini_sellers=50):
    self.height = height
    self.width = width
    self.ini_buyers = ini_buyers
    self.ini_sellers = ini_sellers

    self.schedule = RandomActivationByBreed(self)
    self.grid = MultiGrid(self.height, self.width, torus=True)
    self.datacollector = DataCollector(
      {"Sellers": lambda m: m.schedule.get_breed_count(Seller),
      "Buyers": lambda m: m.schedule.get_breed_count(Buyer)})


    '''
    Generate a matching matrix
      (for now) just random 0 & 1
      Sellers are on rows. Buyers on columns.
      Wal-Mart has 1s for every buyers (correct?)
    '''
    self.match = np.random.randint(2, size=(ini_sellers-1, ini_buyers))
    self.match = np.append(match, [np.ones(ini_buyers,dtype=np.int)], axis=0)

    '''
    Price
      To let buyers access the prices, define as a class attribute.
      a vector of size ini_sellers
      (arbitrary) range from 1 to 3
    '''
    self.prices = 2 * np.random.rand(ini_sellers) + 1

    # Create buyers:
    for i in range(self.ini_buyers):
      x = random.randrange(self.width)
      y = random.randrange(self.height)
      '''
      Income
        income > max(price) to make every sellers affordable
      '''
      income = 10 * np.random.rand() + max(self.prices)
      buyer = Buyer(i, self.grid, (x, y), True, income)
      self.grid.place_agent(buyer, (x, y))
      self.schedule.add(buyer)

    # Create sellers
    for i in range(self.ini_sellers):
      x = random.randrange(self.width)
      y = random.randrange(self.height)
      cash = 100 # initial cash balance
      '''
      Fixed costs
        relative to ini_buyers, implying the required market share
      '''
      costs = 0.1 * ini_buyers
      price = self.prices[i]
      w = False
      # The last seller is Wal-Mart
      if i == self.ini_sellers - 1:
        w = True
      # i is used as a unique identifier
      seller = Seller(i, self.grid, (x, y), True, cash, costs, price, w)
      self.grid.place_agent(seller, (x, y))
      self.schedule.add(seller)

    self.running = True

# Not yet worked on
  def step(self):
    self.schedule.step()
    self.datacollector.collect(self)
    if self.verbose:
      print([self.schedule.time,
        self.schedule.get_breed_count(Seller),
        self.schedule.get_breed_count(Buyer)])

# Not yet worked on
  def run_model(self, step_count=200):
    if self.verbose:
      print('Initial number sellers: ',
        self.schedule.get_breed_count(Seller))
      print('Initial number buyers: ',
        self.schedule.get_breed_count(Buyer))

    for i in range(step_count):
      self.step()

    if self.verbose:
      print('')
      print('Final number sellers: ',
        self.schedule.get_breed_count(Seller))
      print('Final number buyers: ',
        self.schedule.get_breed_count(Buyer))


class Buyer(Agent):
'''
income: for wealth
'''
  def __init__(self, bid, grid, pos, moore, income):
    self.bid = bid # buyer unique id
    self.grid = grid
    self.pos = pos
    self.moore = moore
    self.income = income

  def step(self, model):
    # Write an optimization problem
    # prices can be accessed thru model.prices with sid as their indices



class Seller(Agent):
'''
sid: seller unique id
cash: liquidity level. analogue of energy.
costs: fixed costs, working as the threshold of breakeven
w: boolean for conventional producer (Wal-Mart), who is immortal.
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

  def step(self, model):
    # (if not w) Cash changes by sales - the fixed costs
    if w==False:
      self.cash = # need to get the sales at the previous period

    # Insolvency
    if self.cash < 0:
      model.grid._remove_agent(self.pos, self)
      model.schedule.remove(self)

    # Post a new price
    else:
      # For now, it is fixed
      # model.prices[self.sid] =



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
  agents_by_breed = defaultdict(list)

  def __init__(self, model):
    super().__init__(model)
    self.agents_by_breed = defaultdict(list)

  def add(self, agent):
    '''
    Add an Agent object to the schedule

    Args:
      agent: An Agent to be added to the schedule.
    '''

    self.agents.append(agent)
    agent_class = type(agent)
    self.agents_by_breed[agent_class].append(agent)

  def remove(self, agent):
    '''
    Remove all instances of a given agent from the schedule.
    '''

    while agent in self.agents:
      self.agents.remove(agent)

    agent_class = type(agent)
    while agent in self.agents_by_breed[agent_class]:
      self.agents_by_breed[agent_class].remove(agent)

  def step(self, by_breed=True):
    '''
    Executes the step of each agent breed, one at a time, in random order.

    Args:
      by_breed: If True, run all agents of a single breed before running
            the next one.
    '''
    if by_breed:
      for agent_class in  self.agents_by_breed:
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
    agents = self.agents_by_breed[breed]
    random.shuffle(agents)
    for agent in agents:
      agent.step(self.model)

  def get_breed_count(self, breed_class):
    '''
    Returns the current number of agents of certain breed in the queue.
    '''
    return len(self.agents_by_breed[breed_class])
