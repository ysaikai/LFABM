'''
Notes

[Basic]
A fraction of sellers operates as CSA farmers.
A single period is interpreted as a week.
A unit of CSA sale corresponds to 52 units of sales for a standard farmer.
At the beginning of every 52 periods,
  Shares are paid
  A random regional event that destroys the subsequent t-period sales
    Costs are avoided
    Standard: the regular trust depreciation
    CSA: Besides, members' (the current buyers) trust goes down badly

'''

frac_CSA = 0.5
frac_std = 1 - frac_CSA
len_season = 26 # length of a season
