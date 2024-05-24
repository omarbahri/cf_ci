#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:53:21 2024

@author: omar
"""

import numpy as np

def _clamp(loc, vel, box_size=5.0):
    """
    :param loc: 2xN location at one time stamp
    :param vel: 2xN velocity at one time stamp
    :return: location and velocity after hitting walls and returning after
        elastically colliding with walls
    """
    assert np.all(loc < box_size * 3)
    assert np.all(loc > -box_size * 3)

    over = loc > box_size
    loc[over] = 2 * box_size - loc[over]
    assert np.all(loc <= box_size)

    # assert(np.all(vel[over]>0))
    vel[over] = -np.abs(vel[over])

    under = loc < -box_size
    loc[under] = -2 * box_size - loc[under]
    # assert (np.all(vel[under] < 0))
    assert np.all(loc >= -box_size)
    vel[under] = np.abs(vel[under])

    return loc, vel

def simulate_step(loc, vel, edges, sample_freq=10, interaction_strength=0.1):
    _delta_T = 0.001
    _max_F = 0.1 / _delta_T
    
    n = len(edges)
    
    loc, vel = _clamp(loc, vel)
    loc_next, vel_next = loc, vel
    
    # disables division by zero warning, since I fix it with fill_diagonal
    with np.errstate(divide="ignore"):

        forces_size = -interaction_strength * edges
        np.fill_diagonal(
            forces_size, 0
        )  # self forces are zero (fixes division by zero)
        F = (
            forces_size.reshape(1, n, n)
            * np.concatenate(
                (
                    np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                        1, n, n
                    ),
                    np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                        1, n, n
                    ),
                )
            )
        ).sum(
            axis=-1
        )  # sum over influence from different particles to get their joint force
        F[F > _max_F] = _max_F
        F[F < -_max_F] = -_max_F

        vel_next += _delta_T * F
        
        # run leapfrog
        for i in range(sample_freq):
            loc_next += _delta_T * vel_next
            loc_next, vel_next = _clamp(loc_next, vel_next)

            forces_size = -interaction_strength * edges
            np.fill_diagonal(forces_size, 0)
            # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

            F = (
                forces_size.reshape(1, n, n)
                * np.concatenate(
                    (
                        np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                            1, n, n
                        ),
                        np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                            1, n, n
                        ),
                    )
                )
            ).sum(axis=-1)
            F[F > _max_F] = _max_F
            F[F < -_max_F] = -_max_F
            vel_next += _delta_T * F
    
    return loc_next, vel_next