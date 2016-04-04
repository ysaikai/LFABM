from main import Seller, Buyer, Trade
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

color_buyer = "#666666" # grey
color_seller = "#339900" # green
color_WM = "#990000" # Wal-Mart red
modelname = "LFABM"

def LFABM_portrayal(agent):
  if agent is None: return

  portrayal = {"Shape": "circle", "Filled": "true"}
  '''
  The higher layer is placed on top of the lower. So, the lower should have
  a smaller portrayal["r"] (the size of shape) so that it can be seen.
  '''
  if type(agent) is Buyer:
    portrayal["Color"] = color_buyer
    portrayal["r"] = 0.8
    portrayal["Layer"] = 1

  elif type(agent) is Seller:
    if agent.w:
      portrayal["Color"] = color_WM
    else:
      if agent.price < 0.5:
        portrayal["Color"] = "#b3ffb3"   # Light green for low price
      elif agent.price < 1:
        portrayal["Color"] = "#33ff33"
      elif agent.price < 1.5:
        portrayal["Color"] = "#00cc00"
      else:
        portrayal["Color"] = "#004d00"  # Dark green for highe price
      #portrayal["Color"] = color_seller
    portrayal["r"] = 0.5
    portrayal["Layer"] = 2

  return portrayal

canvas_element = CanvasGrid(LFABM_portrayal, 20, 20, 500, 500)
chart_element = ChartModule([
  {"Label": "Sellers", "Color": color_seller},
  {"Label": "Wal-Mart", "Color": color_WM},
  {"Label": "Buyers", "Color": color_buyer}])

server = ModularServer(Trade, [canvas_element, chart_element], modelname)
server.launch()
