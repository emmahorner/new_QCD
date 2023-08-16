import RRR
import numpy as np
import importlib

importlib.reload(RRR)

ABC = np.mean(RRR.A) * np.std(RRR.B) * RRR.C
print("in NNN, ABC = {}".format(ABC))

def print_ABC():
    print(ABC)