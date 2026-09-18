import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Write code here
    param=np.array(param)#convert input list into numpy arrays
    grad=np.array(grad)
    m=np.array(m)
    v=np.array(v)

    m_new=beta1*m+(1-beta1)*grad  #update first moment
    v_new=beta2*v+(1-beta2)*grad**2  #update biased second raw moment
    m_new1=m_new/(1-beta1**t)  #compute biased correct first moment
    v_new2=v_new/(1-beta2**t)  #xomputr biased correctd second raw momnet

    
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """

    param_new=param-(lr*m_new1)/(np.sqrt(v_new2)+eps)  #update params
    return param_new,m_new,v_new 
    
    