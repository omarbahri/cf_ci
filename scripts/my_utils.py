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

def simulate_T(loc, vel, edges, T, sample_freq=10, interaction_strength=0.1):
    _delta_T = 0.001
    _max_F = 0.1 / _delta_T
    
    n = len(edges)
    
    assert T % sample_freq == 0
    T_save = int(T / sample_freq - 1)
    
    locs = np.zeros((T_save, 2, n))
    vels = np.zeros((T_save, 2, n))
    
    loc_next, vel_next = _clamp(loc, vel)
    locs[0, :, :], vels[0, :, :] = loc_next, vel_next
    
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
        
        counter = 1
        
        # run leapfrog (-sample_freq & counter=1 because we want to keep initial state, as opposed to original code)
        # we assume that the initial state is already in equilibrium
        for i in range(1, T - sample_freq):
            loc_next += _delta_T * vel_next
            loc_next, vel_next = _clamp(loc_next, vel_next)
            
            if i % sample_freq == 0:
                locs[counter, :, :], vels[counter, :, :] = loc_next, vel_next
                counter += 1

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
    
    return locs, vels

#simulate next trajectory time step
def simulate_t_step(loc, vel, edges, sample_freq=10, interaction_strength=0.1):
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

#simulate equilibrium + 1 step
def simulate_1_step(loc, vel, edges, interaction_strength=0.1):
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
        
        # run leapfrog for one step
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

#simulate equilibrium
def simulate_eq(loc, vel, edges, interaction_strength=0.1):
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
        
        # run leapfrog for one step
        loc_next += _delta_T * vel_next
        loc_next, vel_next = _clamp(loc_next, vel_next)

    return loc_next, vel_next


First, write loss for t=0 with dist for exogenous variables and dist(value, f(other_values))
    Test by running just that and zeroing t=1+
Either make a 2 step loss
Or write more fs(previous and other current values) for each point 


def get_exo_endo():
    adj_matrix = tf.constant(self.edges, dtype=tf.float32)
    
    # Check if rows are all zeros
    exogenous_mask = tf.reduce_all(tf.equal(adj_matrix, 0), axis=1)
    endogenous_mask = tf.logical_not(exogenous_mask)
    
    # Get the indices of exogenous and endogenous variables
    exogenous_indices = tf.where(exogenous_mask)[:, 0]
    endogenous_indices = tf.where(endogenous_mask)[:, 0]
    
    # Convert the indices to Python lists
    exogenous_indices = exogenous_indices.numpy().tolist()
    endogenous_indices = endogenous_indices.numpy().tolist()
    
    return exogenous_indices, endogenous_indices

def endo_dist(cf, orig):
    loc_cf, vel_cf = reshape_to_orig(cf)
    loc_orig, vel_orig = reshape_to_orig(orig)
        
    dist = 0.
    for i in endogenous:
        dist += tf.reduce_sum(tf.abs(loc_cf[0,:,i] - loc_orig[0,:,i]),
                              axis=None, name='l1_loc')
        dist += tf.reduce_sum(tf.abs(vel_cf[0,:,i] - vel_orig[0,:,i]),
                              axis=None, name='l1_vel')
    
    return

@tf.function
def reshape_orig(cf):
    loc = cf[:, :10, :]
    vel = cf[:, 10:, :]
    
    loc = loc.reshape(1, 5, 2, 49).transpose(0, 3, 2, 1)
    vel = vel.reshape(1, 5, 2, 49).transpose(0, 3, 2, 1)
    
    return loc.reshape((49, 2, 5)), vel.reshape((49, 2, 5))
