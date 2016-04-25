import numpy as np

'''
Display trust levels
'''
def buyers(arg):
  print("\nBuyers trust levels Top3 (sid in brackets)")
  print("{0:6} {1:9} {2:9} {3:9}".format("bid", "#1", "#2", "#3"))
  for obj in arg.values():
    bid = obj.bid
    tmp = list(obj.trust.values())
    t1 = max(tmp)
    sid1 = tmp.index(t1)
    tmp.remove(t1)
    t2 = max(tmp)
    sid2 = tmp.index(t2)
    tmp.remove(t2)
    t3 = max(tmp)
    sid3 = tmp.index(t3)
    print("{bid:03d} {t1:5.2f}({sid1:02d}) {t2:5.2f}({sid2:02d}) {t3:5.2f}({sid3:02d})".format(bid=bid,t1=t1,t2=t2,t3=t3,sid1=sid1,sid2=sid2,sid3=sid3))


'''
Show seller information
'''
def sellers(cnt, num_w, sellers, buyers):
  print("\nPeriod:", cnt)
  print(len(sellers)-num_w, "local sellers")
  print("{:>4} {:>9} {:>3} {:>5} {:>7} {:>6} {:>8} {:>6} {:>8} {:>4}".format("sid", "Cell", "CSA", "Ebd", "Price", "Sales", "Profits", "Cash", "AvgTst", "Age"))
  for obj in sellers.values():
    sid = obj.sid
    # Calculate the average trust of the customers at the current period
    τ = 0
    avg_τ = 0
    for customer in obj.customers[cnt]:
      τ += buyers[customer].trust[sid]
    if τ > 0:
      avg_τ = τ / len(obj.customers[cnt])
    print("{:>4} {:>9} {:>3} {:>5.2f} {:>5.2f} {:1} {:>6} {:>8.2f} {:>6.0f} {:>8.2f} {:>4}".format(sid, str(obj.pos), obj.csa, obj.e, obj.price, obj.priceDir, obj.sales, obj.profits, obj.cash, avg_τ, obj.age))

'''
exp(util) - weight
'''
def util_weight(bid, utils, weights):
  # print("\nexp() - bid:", bid)
  # print(''.join(["{:.2f} ".format(x) for x in np.sort(utils)]))
  print("\nweights - bid:", bid)
  print(''.join(["{:.2f} ".format(x) for x in np.sort(weights)]))
