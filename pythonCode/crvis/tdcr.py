import matplotlib.pyplot as plt

from .class_define import *
from .utils import list_to_numpy


def draw_tdcr(
        g: LIST_OR_NUMPY, seg_end: LIST_OR_NUMPY, r_disk=2.5e-3, r_height=1.5e-3, **kwargs
):
    # DRAW_TDCR Creates a figure of a tendon-driven continuum robot (tdcr)
    #
    #   Takes a matrix with nx16 entries, where n is the number
    #   of points on the backbone curve. For each point on the curve, the 4x4
    #   transformation matrix is stored columnwise (16 entries). The x- and
    #   y-axis span the material orientation and the z-axis is tangent to the
    #   curve.
    #
    #   INPUT
    #   g(n,16): backbone curve with n 4x4 transformation matrices reshaped into 1x16 vector (columnwise)
    #   seg_end(1,m): indices of g where tdcr segments terminate
    #   r_disk: radius of spacer disks
    #   r_height: height of space disks
    #   options:
    #       tipframe (shows tip frame, default true/1)
    #       segframe (shows segment end frames, default false/0)
    #       baseframe (shows robot base frame, default false/0)
    #       projections (shows projections of backbone curve onto
    #                    coordinate axes, default false/0)
    #       baseplate (shows robot baseplate, default false/0)
    #
    #
    #   Author: Jessica Burgner-Kahrs <jbk@cs.toronto.edu>
    #   Date: 2023/01/04
    #   Version: 0.1
    #
    #   Copyright: 2023 Continuum Robotics Laboratory, University of Toronto
    g = list_to_numpy(g)  # dim: nx16
    seg_end = list_to_numpy(seg_end)  # dim:1xn
    tipframe = True
    segframe = False
    baseframe = False
    projections = False
    baseplate = False
    for k, v in kwargs.items():
        if k == 'tipframe':
            tipframe = bool(v)
        elif k == 'segframe':
            segframe = bool(v)
        elif k == 'baseframe':
            baseframe = bool(v)
        elif k == 'projections':
            projections = bool(v)
        elif k == 'baseplate':
            baseplate = bool(v)
        else:
            raise ValueError('Unknown option: ' + k)
    assert g.shape[1] == 16, "The dimension of g is wrong, should be (n,16)"
    assert seg_end.ndim == 1, "Make sure seg_end is a one dimension vector"
    assert g.shape[0] >= seg_end.size and g.shape[0] >= np.max(seg_end), "g and seg_end dimension mismatch"
    numseg = seg_end.size
    curvelength = np.linalg.norm(g[1:, 12:14] - g[0:-1, 12:14]).sum()

    # plot
    fig = plt.figure(figsize=(12, 10))  # change from 1280, 1024
    clearance = 0.03
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(left=-g[:, 12].abs().max() - clearance, right=g[:, 12].abs().max() + clearance)
    ax.set_ylim(left=-g[:, 13].abs().max() - clearance, right=g[:, 13].abs().max() + clearance)
    ax.set_zlim(left=0, right=curvelength + clearance)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.grid(True, alpha=0.3)
    ax.set_title('Tendon-Driven Continuum Robot (TDCR)')
    ax.view_init(elev=45, azim=45)
    # from matlab: view([0.5 0.5 0.5]), and refer to https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
    ax.set_aspect('equal')  # from matlab: daspect([1 1 1])

    col = np.linspace(0.2, 0.8, numseg)

    # backbone
    ax.plot(g[:seg_end[0], 12], g[:seg_end[0], 13], g[:seg_end[0], 14], color=np.full((1, 3), col[0]), linewidth=5)
    for i in range(1, numseg):
        ax.plot(g[seg_end[i - 1]:seg_end[i], 12], g[seg_end[i - 1]:seg_end[i], 13], g[seg_end[i - 1]:seg_end[i], 14],
                color=col[i] * np.ones((1, 3)), linewidth=5)

    # projections
    if projections:
        ax.plot(g[:, 12], np.full(g.shape[0], ax.get_ylim()[0], g[:, 14]), color=[0, 1, 0], linewidth=2)
        ax.plot(np.full(g.shape[0], ax.get_xlim()[0], g[:, 13], g[:, 14]), color=[1, 0, 0], linewidth=2)
        ax.plot(g[:, 12], g[:, 13], np.zeros(g.shape[0]), color=[0, 1, 0], linewidth=2)

    # tendons
    tendon1 = np.zeros((seg_end[numseg - 1], 3))
    tendon2 = tendon1.copy()
    tendon3 = tendon1.copy()

    # tendon locations on disk
