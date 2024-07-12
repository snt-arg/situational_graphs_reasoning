from models import *
import numpy as np

don = DiffusionOrderingNetwork(10,5,4, out_channels=4)



class Graph():
    def __init__(self):
        self.x = np.zeros((3))


    
don.forward(Graph(), [1])