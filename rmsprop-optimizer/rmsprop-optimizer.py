import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    w1=np.array(w)
    g1=np.array(g)
    s1=np.array(s)

    new_s=beta*s1+(1-beta)*g1**2
    new_w=w1-(lr/np.sqrt(new_s+eps))*g1

    return new_w.tolist(),new_s.tolist()
    