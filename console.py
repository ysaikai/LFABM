import sys
import main

model = main.Trade()
max_steps = int(sys.argv[1])
model.run_model(max_steps)

# history = model.dc.get_agent_vars_dataframe()
# history.reset_index(inplace=True)
# history[history.Step == (max_steps - 1)].Wealth.hist(bins=range(10))
