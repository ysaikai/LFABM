from main import Seller, Buyer, Trade
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

color_buyer = "#f5f5f5" # very light grey
color_seller = "#00FF00"
price_color_on = 1 # display seller price in their color
color_WM = "#FF0000"
modelname = "LFABM"

def LFABM_portrayal(agent):
  if (agent is None or type(agent) is Buyer): return

  portrayal = {"Shape": "circle", "Filled": "true"}
  '''
  The higher layer is placed on top of the lower. So, the lower should have
  a bigger portrayal["r"] (the size of shape) so that it can be seen.
  '''
  # if type(agent) is Seller:
  portrayal["Color"] = color_seller
  if agent.w:
    portrayal["Color"] = color_WM
  elif (price_color_on):
    val = 255-int(255*(agent.price/3))
    # Lighter = low price, darker = high price
    price_color = '#'+str(val)+'FF'+str(val)
    portrayal["Color"] = price_color

  portrayal["r"] = 0.5
  portrayal["Layer"] = 2

  # elif type(agent) is Buyer:
  #   portrayal["Color"] = color_buyer
  #   portrayal["r"] = 0.9
  #   portrayal["Layer"] = 1

  return portrayal

canvas_element = CanvasGrid(LFABM_portrayal, 20, 20, 500, 500)
chart_element = ChartModule([
  {"Label": "Sellers", "Color": color_seller}])

server = ModularServer(Trade, [canvas_element, chart_element], modelname)
server.launch()
