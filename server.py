from main import Seller, Buyer, Trade
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule

def LFABM_portrayal(agent):
    if agent is None:
        return

    portrayal = {"Shape": "circle", "Filled": "true"}

    if type(agent) is Buyer:
        portrayal["Color"] = "#666666"
        portrayal["r"] = 0.8
        portrayal["Layer"] = 1

    elif type(agent) is Seller:
        portrayal["Color"] = "#AA0000"
        portrayal["r"] = 0.5
        portrayal["Layer"] = 2

    return portrayal

canvas_element = CanvasGrid(LFABM_portrayal, 20, 20, 500, 500)
chart_element = ChartModule([{"Label": "Sellers", "Color": "#AA0000"},
    {"Label": "Buyers", "Color": "#666666"}])

server = ModularServer(Trade, [canvas_element, chart_element], "main")
server.launch()
