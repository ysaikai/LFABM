

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
  print("{0:<5} {1:<9} {2:<6} {3:<7} {4:<7} {5:<8} {6:<7}".format("sid", "Cell", "CSA", "Price", "Sales", "Cash", "Trust"))
  for obj in sellers.values():
    sid = obj.sid
    t = 0 # To calculate the cumulative trust
    for buyer in buyers.values():
      t += buyer.trust[sid]
    t = int(t)
    print("{0:<5} {1:<9} {2:<6} {3:<7} {4:<7} {5:<8} {6:<7}".format(sid, str(obj.pos), str(obj.csa), round(obj.price,2), obj.sales, round(obj.cash,2),t))
