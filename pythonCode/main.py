import numpy as np


def test_robot_independent_mapping():
    from crvis.util import robot_independent_mapping

    kappa = [1 / 30e-3, 1 / 40e-3, 1 / 15e-3]
    phi = [0, np.deg2rad(160), np.deg2rad(30)]
    ell = [50e-3, 70e-3, 25e-3]
    pts_per_seg = 30

    g = robot_independent_mapping(kappa, phi, ell, pts_per_seg)
    print("done")


if __name__ == "__main__":
    test_robot_independent_mapping()
