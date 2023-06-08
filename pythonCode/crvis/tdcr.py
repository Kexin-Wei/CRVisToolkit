import matplotlib.pyplot as plt
import numpy as np

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
    assert g.shape[0] >= seg_end.size and g.shape[0] >= np.max(
        seg_end), "g and seg_end dimension mismatch"
    numseg = seg_end.size
    curvelength = np.linalg.norm(g[1:, 12:14] - g[0:-1, 12:14]).sum()

    # plot
    fig = plt.figure(figsize=(12, 10))  # change from 1280, 1024
    clearance = 0.03
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(left=-g[:, 12].abs().max() - clearance,
                right=g[:, 12].abs().max() + clearance)
    ax.set_ylim(left=-g[:, 13].abs().max() - clearance,
                right=g[:, 13].abs().max() + clearance)
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
    ax.plot(g[:seg_end[0], 12], g[:seg_end[0], 13],
            g[:seg_end[0], 14], color=np.full((1, 3), col[0]), linewidth=5)
    for i in range(1, numseg):
        ax.plot(g[seg_end[i - 1]:seg_end[i], 12], g[seg_end[i - 1]:seg_end[i], 13], g[seg_end[i - 1]:seg_end[i], 14],
                color=col[i] * np.ones((1, 3)), linewidth=5)

    # projections
    if projections:
        ax.plot(g[:, 12], np.full(g.shape[0], ax.get_ylim()[
            0], g[:, 14]), color=[0, 1, 0], linewidth=2)
        ax.plot(np.full(g.shape[0], ax.get_xlim()[
            0], g[:, 13], g[:, 14]), color=[1, 0, 0], linewidth=2)
        ax.plot(g[:, 12], g[:, 13], np.zeros(
            g.shape[0]), color=[0, 1, 0], linewidth=2)

    # tendons
    tendon1 = np.zeros((seg_end[numseg - 1], 3))
    tendon2 = tendon1.copy()
    tendon3 = tendon1.copy()

    # tendon locations on disk
    r1 = np.array([0, r_disk, 0])
    r2 = np.array([np.cos(30 * np.pi / 180) * r_disk, -
                  np.sin(30 * np.pi / 180) * r_disk, 0])
    r3 = np.array([-np.cos(30 * np.pi / 180) * r_disk, -
                  np.sin(30 * np.pi / 180) * r_disk, 0])

    for i in range(0, seg_end[numseg]):
        RotMat = np.reshape(np.array(g[i, 0:3], g[i, 4:7], g[i, 8:11]), (3, 3))
        tendon1[i, :] = RotMat @ r1.T + g[i, 12:15]
        tendon2[i, :] = RotMat @ r2.T + g[i, 12:15]
        tendon3[i, :] = RotMat @ r3.T + g[i, 12:15]

    ax.plot(tendon1[:, 1], tendon1[:, 2], tendon1[:, 3], color=[0, 0, 0])
    ax.plot(tendon2[:, 1], tendon2[:, 2], tendon2[:, 3], color=[0, 0, 0])
    ax.plot(tendon3[:, 1], tendon3[:, 2], tendon3[:, 3], color=[0, 0, 0])

    # draw spheres to represent tendon location at end disks
    x, y, z = sphere()
    radius = 0.75e-3
    for i in range(0, numseg):
        ax.plot_surface(radius * x + tendon1[seg_end[i], 1],
                        radius * y + tendon1[seg_end[i], 2],
                        radius * z + tendon1[seg_end[i], 3], color=[0, 0, 0])
        ax.plot_surface(radius * x + tendon2[seg_end[i], 1],
                        radius * y + tendon2[seg_end[i], 2],
                        radius * z + tendon2[seg_end[i], 3], color=[0, 0, 0])
        ax.plot_surface(radius * x + tendon3[seg_end[i], 1],
                        radius * y + tendon3[seg_end[i], 2],
                        radius * z + tendon3[seg_end[i], 3], color=[0, 0, 0])

    # spacer disks
    for i in range(0, g.shape[0]):
        # change from matlab seg = find(seg_end >= i,1); # TODO seg_end = seg_end -1?
        seg = np.argwhere(seg_end > i)
        color = np.full((1, 3), col[seg[0]]
                        ) if seg.size else np.full((1, 3), col[0])
        RotMat = np.reshape(np.array(g[i, 0:3], g[i, 4:7], g[i, 8:11]), (3, 3))
        normal = RotMat[:, 2]  # TODO check if need .T
        pos = g[i, 12:15].T - RotMat @ np.array([0, 0, r_height]).T

        theta = np.arange(0, 2 * np.pi, 0.05)
        # TODO null space


def sphere(N=20):
    # https://stackoverflow.com/questions/51645694/how-to-plot-a-perfectly-smooth-sphere
    u = np.linspace(-np.pi, np.pi, N + 1)
    v = np.linspace(0, np.pi, N + 1)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z
