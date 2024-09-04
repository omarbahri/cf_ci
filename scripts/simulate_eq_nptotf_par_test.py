import tensorflow as tf
import numpy as np
import sys
import time

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
    
    # print('hereclamp', loc)

    return loc, vel


def _clamp_tf(loc, vel, box_size=5.0):
    """
    :param loc: 2xN location at one time stamp
    :param vel: 2xN velocity at one time stamp
    :return: location and velocity after hitting walls and returning after
        elastically colliding with walls
    """
    over = loc > box_size
    loc = tf.where(over, 2 * box_size - loc, loc)
    vel = tf.where(over, -tf.abs(vel), vel)

    under = loc < -box_size
    loc = tf.where(under, -2 * box_size - loc, loc)
    vel = tf.where(under, tf.abs(vel), vel)
    
    # print('hereclamptf', loc)

    return loc, vel

def simulate_T(loc, vel, edges, T, sample_freq=10, interaction_strength=0.1):
    _delta_T = 0.001
    _max_F = 0.1 / _delta_T
    
    n = len(edges)
    
    assert T % sample_freq == 0
    T_save = int(T / sample_freq - 1)
    
    locs = np.zeros((T_save, 2, n))
    vels = np.zeros((T_save, 2, n))
    
    # print('hereloc', loc)
    loc_next, vel_next = _clamp(loc, vel)
    locs[0, :, :], vels[0, :, :] = loc_next, vel_next
    
    # print('here', loc_next)
    # return
    
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

def simulate_T_tf(loc, vel, edges, T, sample_freq=10, interaction_strength=0.1):
    _delta_T = 0.001
    _max_F = 0.1 / _delta_T
    
    assert T % sample_freq == 0
    T_save = int(T / sample_freq - 1)
    
    # Prepare tensors to store results
    locs = tf.TensorArray(tf.float64, size=T_save)
    vels = tf.TensorArray(tf.float64, size=T_save)
    
    loc_next, vel_next = _clamp_tf(loc, vel)
    
    locs = locs.write(0, loc_next)
    vels = vels.write(0, vel_next)
    
    # Precompute forces size matrix
    forces_size = -interaction_strength * tf.cast(edges, loc.dtype)
    forces_size = tf.linalg.set_diag(forces_size, tf.zeros(forces_size.shape[0], dtype=forces_size.dtype))
    
    # Function to compute forces
    def compute_forces(loc):
        delta_loc = tf.expand_dims(loc, axis=-1) - tf.expand_dims(loc, axis=1)
        # Expand the dimensions of forces_size to make it compatible with delta_loc
        forces_size_expanded = tf.expand_dims(forces_size, axis=0)  # Shape becomes [1, 5, 5]
        force = tf.reduce_sum(forces_size_expanded * delta_loc, axis=2)  # Now shapes are [2, 5, 5] and [2, 5, 5]
        return tf.clip_by_value(force, -_max_F, _max_F)
            
    # Simulation loop
    for i in tf.range(1, T - sample_freq):
        # Compute forces
        F = compute_forces(loc_next)
        
        # Update velocities and locations
        vel_next += _delta_T * F
        loc_next += _delta_T * vel_next
        
        # Clamp locations and velocities
        loc_next, vel_next = _clamp_tf(loc_next, vel_next)
        
        # Write results at every sample_freq step
        if i % sample_freq == 0:
            locs = locs.write(i // sample_freq, loc_next)
            vels = vels.write(i // sample_freq, vel_next)
    
    return locs.stack(), vels.stack()


def test_equivalence_optimized():
    loc_np = np.array([[1.77813504, 2.7875522 , 1.9526118 , 2.39137164, 3.43193947],
           [3.25909722, 3.09977853, 1.92771383, 3.14199163, 2.81109376]])
    vel_np = np.array([[0.52507615, 0.70078394, 0.62429267, 0.39986521, 0.95504668],
           [0.97639808, 0.81367636, 0.49271305, 0.695973  , 0.63860918]])
    edges_np = np.array([[0., 1., 1., 1., 0.],
                         [0., 0., 1., 1., 1.],
                         [0., 1., 0., 0., 0.],
                         [0., 0., 1., 0., 0.],
                         [0., 0., 0., 0., 0.]])
    T = 1000
    
    start_time = time.time()
    locs_np, vels_np = simulate_T(loc_np, vel_np, edges_np, T)
    numpy_time = time.time() - start_time
    
    loc_np = np.array([[1.77813504, 2.7875522 , 1.9526118 , 2.39137164, 3.43193947],
           [3.25909722, 3.09977853, 1.92771383, 3.14199163, 2.81109376]])
    vel_np = np.array([[0.52507615, 0.70078394, 0.62429267, 0.39986521, 0.95504668],
           [0.97639808, 0.81367636, 0.49271305, 0.695973  , 0.63860918]])
    edges_np = np.array([[0., 1., 1., 1., 0.],
                         [0., 0., 1., 1., 1.],
                         [0., 1., 0., 0., 0.],
                         [0., 0., 1., 0., 0.],
                         [0., 0., 0., 0., 0.]])
    
    start_time = time.time()
    locs_tf, vels_tf = simulate_T_tf(tf.constant(loc_np), tf.constant(vel_np),
                                     tf.constant(edges_np), T)
    tensorflow_time = time.time() - start_time
    
    # # Find indices where the location arrays differ
    # loc_diff_indices = np.where(~np.isclose(locs_np, locs_tf.numpy(), rtol=1e-5))
    # vel_diff_indices = np.where(~np.isclose(vels_np, vels_tf.numpy(), rtol=1e-5))
    
    # print("Indices where location arrays differ:", loc_diff_indices)
    # print("Differences in locations at these indices:", locs_np[loc_diff_indices], locs_tf.numpy()[loc_diff_indices])
    
    # print("Indices where velocity arrays differ:", vel_diff_indices)
    # print("Differences in velocities at these indices:", vels_np[vel_diff_indices], vels_tf.numpy()[vel_diff_indices])


    # Check if the results are close enough
    try:
        np.testing.assert_allclose(locs_np, locs_tf.numpy(), rtol=1e-5)
        print("Location arrays are close enough!")
    except AssertionError as e:
        print("Location arrays are not close enough:", e)

    try:
        np.testing.assert_allclose(vels_np, vels_tf.numpy(), rtol=1e-5)
        print("Velocity arrays are close enough!")
    except AssertionError as e:
        print("Velocity arrays are not close enough:", e)
        
    
    print(f"NumPy function runtime: {numpy_time:.6f} seconds")
    print(f"TensorFlow function runtime: {tensorflow_time:.6f} seconds")


# Run the test
test_equivalence_optimized()
