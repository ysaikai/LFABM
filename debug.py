import numpy as np

'''
Display trust levels
'''
def buyers(arg):
  print("\nBuyers trust levels (sid in brackets)")
  print("{:6} {:9} {:9} {:9} {:9} {:9}".format("bid", "#1", "#2", "#3", "#4", "#5"))
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
    tmp.remove(t3)
    t4 = max(tmp)
    sid4 = tmp.index(t4)
    tmp.remove(t4)
    t5 = max(tmp)
    sid5 = tmp.index(t5)
    print("{bid:03d} {t1:5.2f}({sid1:02d}) {t2:5.2f}({sid2:02d}) {t3:5.2f}({sid3:02d}) {t4:5.2f}({sid4:02d}) {t5:5.2f}({sid5:02d})".format(bid=bid,t1=t1,t2=t2,t3=t3,t4=t4,t5=t5,sid1=sid1,sid2=sid2,sid3=sid3,sid4=sid4,sid5=sid5))

'''
Network
'''
def network(arg):
  print("\nBuyers network")
  for buyer in arg.values():
    print(buyer.ties)


'''
Show seller information
'''
def sellers(cnt, num_w, sellers, buyers):
  print("\nPeriod:", cnt)
  print(len(sellers)-num_w, "local sellers")
  print("{:>4} {:>9} {:>3} {:>5} {:>7} {:>6} {:>8} {:>6} {:>8} {:>4}".format("sid", "Cell", "CSA", "Ebd", "Price", "Sales", "Profits", "Cash", "AvgTst", "Age"))
  totCSA = 0
  totE = 0
  totPrice = 0
  totSales = 0
  totProfits = 0
  totCash = 0
  totAvgτ = 0
  totAge = 0
  for obj in sellers.values():
    sid = obj.sid
    # Calculate the average trust of the customers at the current period
    τ = 0
    avg_τ = 0
    for customer in obj.customers[cnt]:
      τ += buyers[customer].trust[sid]
    if τ > 0:
      avg_τ = τ / len(obj.customers[cnt])
    # Averaging all seller attributes
    if(sid > 0): # ie is not Walmart
      totCSA += obj.csa
      totE += obj.e
      totPrice += obj.price
      totSales += obj.sales
      totProfits += obj.profits
      totCash += obj.cash
      totAvgτ += avg_τ
      totAge += obj.age

    print("{:>4} {:>9} {:>3} {:>5.2f} {:>5.2f} {:1} {:>6} {:>8.2f} {:>6.0f} {:>8.2f} {:>4}".format(sid, str(obj.pos), obj.csa, obj.e, obj.price, obj.priceDir, obj.sales, obj.profits, obj.cash, avg_τ, obj.age))
  num_sellers = len(sellers.values())
  avgCSA = totCSA/num_sellers
  avgE = totE/num_sellers
  avgPrice = totPrice/num_sellers
  avgSales = totSales/num_sellers
  avgProfits = totProfits/num_sellers
  avgCash = totCash/num_sellers
  avgAvgτ = totAvgτ/num_sellers
  avgAge = totAge/num_sellers
  print("{:<14} {:>3} {:>5.2f} {:>5.2f} {:>8.1f} {:>8.2f} {:>6.0f} {:>8.2f} {:>4.1f}".format("Averages: ", avgCSA, avgE, avgPrice, avgSales, avgProfits, avgCash, avgAvgτ, avgAge))

'''
exp(util) - weight
'''
def util_weight(bid, utils, weights):
  # print("\nexp() - bid:", bid)
  # print(''.join(["{:.2f} ".format(x) for x in np.sort(utils)]))
  print("\nweights - bid:", bid)
  print(''.join(["{:.2f} ".format(x) for x in np.sort(weights)]))
