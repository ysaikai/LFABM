from main import Trade
from visualization import LFABMVisualization

if __name__ == "__main__":
    # model = Trade(grass=True)
    model = Trade()
    model.run_model()
    viz = visualization(model)
