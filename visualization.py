from main import Buyer, Seller
from mesa.visualization.TextVisualization import TextVisualization, TextGrid

class LFABMVisualization(TextVisualization):
  '''
  ASCII visualization
    Each cell displays B if only buyers, S if only sellers, or X if both.
    (blank if none)
  '''

  def __init__(self, model):
    self.model = model
    grid_viz = TextGrid(self.model.grid, self.draw_cell)
    self.elements = [grid_viz]

  @staticmethod
  def draw_cell(cell):
    if len(cell) == 0:
      return " "
    if len([obj for obj in cell if isinstance(obj, Buyer)]) == len(cell):
      return "B"
    if len([obj for obj in cell if isinstance(obj, Seller)]) == len(cell):
      return "S"
    else:
      return "X"
