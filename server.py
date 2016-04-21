# from main import Seller, Buyer, Trade
import main
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

'''Does this fix the cash balance issue?'''
import importlib
importlib.reload(main)

color_buyer = "#f5f5f5" # very light grey
color_seller = "#339900" # green
color_seller_price = 0 # display seller price in their color
color_WM = "#990000" # Wal-Mart red
modelname = "LFABM"

def LFABM_portrayal(agent):
  if agent is None: return

  portrayal = {"Shape": "circle", "Filled": "true"}
  '''
  The higher layer is placed on top of the lower. So, the lower should have
  a bigger portrayal["r"] (the size of shape) so that it can be seen.
  '''
  # The following shows only the static color reflecting the initial prices.
  # If we want to dynamically change the colors, need to go thru the scheduler.
  if type(agent) is main.Seller:
    if agent.w:
      portrayal["Color"] = color_WM
    elif (color_seller_price):
      val = 255-int(255*(agent.price/2))
      price_color = '#'+str(val)+'FF'+str(val)   # Lighter = low price, darker = high price
      portrayal["Color"] = price_color
    else:
      portrayal["Color"] = color_seller
    portrayal["r"] = 0.5
    portrayal["Layer"] = 2

  elif type(agent) is main.Buyer:
    portrayal["Color"] = color_buyer
    portrayal["r"] = 0.9
    portrayal["Layer"] = 1

  return portrayal

canvas_element = CanvasGrid(LFABM_portrayal, 20, 20, 500, 500)
# chart_element = ChartModule([
#   {"Label": "Sellers", "Color": color_seller},
#   {"Label": "Wal-Mart", "Color": color_WM},
#   {"Label": "Buyers", "Color": color_buyer}])
chart_element = ChartModule([
  {"Label": "Sellers", "Color": color_seller}])

server = ModularServer(main.Trade, [canvas_element, chart_element], modelname)
server.launch()
