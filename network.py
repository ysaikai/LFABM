import numpy as np

'''
Tie formation
  Random chance to communicate with another customer in the seller network.
  The more embedded the seller, the greater chance to talk and form a new tie.
  If talking to the already connected or himself (sad...), just put 1 again.
'''
def formation(cnt, buyers, sellers):
  for seller in sellers.values():
    if seller.alive:
      customers = seller.customers[cnt]
      for bid in customers:
        if np.random.rand() < seller.e:
          friend = np.random.choice(customers)
          '''Symmetric tie formation'''
          buyers[bid].ties[friend] = 1
          buyers[friend].ties[bid] = 1
