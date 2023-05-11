from .class_define import LIST_OR_NUMPY
import numpy as np


def listToNumpy(aList: LIST_OR_NUMPY) -> np.ndarray:
    """
    Convert a list to numpy array.
    """
    return np.array(aList)


def robot_independent_mapping(kappa: LIST_OR_NUMPY, phi: LIST_OR_NUMPY, ell: LIST_OR_NUMPY, ptsperseg: LIST_OR_NUMPY):
    """
    ROBOTINDEPENDENTMAPPING creates a framed curve for given configuration parameters
    
       EXAMPLE
           g = robotindependentmapping([1/40e-3;1/10e-3],[0,pi],[25e-3,20e-3],10)
           creates a 2-segment curve with radius of curvatures 1/40 and 1/10
           and segment lengths 25 and 20, where the second segment is rotated by pi rad.
    
       INPUT: configuration parameters
           kappa (nx1): segment curvatures
           phi (nx1): segment bending plane angles
           l (nx1): segment lengths
           ptsperseg (nx1): number of points per segment
                            if n=1 all segments with equal number of points
       OUTPUT: backbone curve
           g (n,16): backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
    """
    # TODO: size check
    # TODO: type check
    kappa = listToNumpy(kappa)
    phi = listToNumpy(phi)
    ell = listToNumpy(ell)
    ptsperseg = listToNumpy(ptsperseg).astype(np.unit8)
    # TODO: check assert
    assert kappa.ndim == 1 and phi.ndim == 1 and ell.ndim == 1 and ptsperseg.ndim == 1, "Input must be 1D array or " \
                                                                                        "1D list."
    assert kappa.size == phi.size == ell.size, "Input 'kappa', 'phi', 'ell' must have the same size."
    numseg = kappa.size
    if ptsperseg.size == 1 and numseg > 1:
        ptsperseg = np.ones(numseg) * ptsperseg

    g = np.zeros((int(ptsperseg.sum()), 16))
    T_base = np.eye(4)
    for i in range(numseg):
        T = np.zeros(int(ptsperseg[i]), 16)
        c_p = np.cos(phi[i])
        s_p = np.sin(phi[i])

        for j in range(int(ptsperseg[i])):
            c_ks = np.cos(kappa[i] * j * ell[i] / ptsperseg[i])
            s_ks = np.sin(kappa[i] * j * ell[i] / ptsperseg[i])
            # TODO: check kappa=0.0
            if kappa[i]:
                T_temp = [c_p * c_p * (c_ks - 1) + 1, s_p * c_p * (c_ks - 1), -c_p * s_ks, 0,
                          s_p * c_p * (c_ks - 1), c_p * c_p * (1 - c_ks) + c_ks, -s_p * s_ks, 0,
                          c_p * s_ks, s_p * s_ks, c_ks, 0,
                          (c_p * (1 - c_ks)) / kappa[i], (s_p * (1 - c_ks)) / kappa[i], s_ks / kappa[i], 1]
            else:
                T_temp = [c_p * c_p * (c_ks - 1) + 1, s_p * c_p * (c_ks - 1), -c_p * s_ks, 0,
                          s_p * c_p * (c_ks - 1), c_p * c_p * (1 - c_ks) + c_ks, -s_p * s_ks, 0,
                          c_p * s_ks, s_p * s_ks, c_ks, 0,
                          0, 0, (j - 1) * (ell[i] / (ptsperseg[i])), 1]
            T[j, :] = np.reshape(T_base * np.reshape(T_temp, (4, 4)), (1, 16))
        if i == 0:
            g[0:int(ptsperseg[i]), :] = T
        else:
            # TODO: check 0:i-1 or 0:i
            ts = int(ptsperseg[0:i - 1].sum())
            g[ts + np.arange(0, ts) + ptsperseg[i], :] = T
        T_base = np.reshape(T[ptsperseg[i], :], (4, 4))
