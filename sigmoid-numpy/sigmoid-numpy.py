import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    if isinstance(x,list):
        return 1/(1+np.exp(-np.array(x)))
    return float(1/(1+np.exp(-x)))
    