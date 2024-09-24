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

def is_valid_state(loc, vel, edges, threshold=1e-5):
    n_particles = loc.shape[0]
    
    # Check spring forces obey Hooke's law
    for edge, rest_length in zip(edges, spring_rest_lengths):
        i, j = edge
        dist = np.linalg.norm(loc[i] - loc[j])
        if np.abs(dist - rest_length) > threshold:
            return False  # Spring not at valid rest length
    
    # If all checks passed, the state is valid
    return True


def simulate_eq(loc, vel, edges, interaction_strength=0.1):
    _delta_T = 0.001
    _max_F = 0.1 / _delta_T
    
    n = len(edges)
        
    loc, vel = _clamp(loc, vel)
        
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
                    np.subtract.outer(loc[0, :], loc[0, :]).reshape(
                        1, n, n
                    ),
                    np.subtract.outer(loc[1, :], loc[1, :]).reshape(
                        1, n, n
                    ),
                )
            )
        ).sum(
            axis=-1
        )  # sum over influence from different particles to get their joint force
        F[F > _max_F] = _max_F
        F[F < -_max_F] = -_max_F

        vel += _delta_T * F
        loc += _delta_T * vel
        loc, vel = _clamp(loc, vel)
    
    return loc, vel


def simulate_next(loc, vel, edges, sample_freq=10, interaction_strength=0.1):
    _delta_T = 0.001
    _max_F = 0.1 / _delta_T
    
    n = len(edges)
    
    print('init', loc, vel)
        
    # loc, vel = simulate_eq(loc, vel, edges)
    loc, vel = _clamp(loc, vel)
    
    print('eq', loc, vel)
    
    # run leapfrog (-sample_freq & counter=1 because we want to keep initial state, as opposed to original code)
    # we assume that the initial state is already in equilibrium
    for i in range(0, sample_freq):
        
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
                        np.subtract.outer(loc[0, :], loc[0, :]).reshape(
                            1, n, n
                        ),
                        np.subtract.outer(loc[1, :], loc[1, :]).reshape(
                            1, n, n
                        ),
                    )
                )
            ).sum(
                axis=-1
            )  # sum over influence from different particles to get their joint force
            F[F > _max_F] = _max_F
            F[F < -_max_F] = -_max_F
    
            vel += _delta_T * F
            loc += _delta_T * vel
            loc, vel = _clamp(loc, vel)
            
            print(loc)
        
    return loc, vel

def simulate_T(loc, vel, edges, T, sample_freq=10, interaction_strength=0.1):
    _delta_T = 0.001
    _max_F = 0.1 / _delta_T
    
    n = len(edges)
    
    print(loc, vel)
    
    assert T % sample_freq == 0
    T_save = int(T / sample_freq - 1)
    
    locs = np.zeros((T_save, 2, n))
    vels = np.zeros((T_save, 2, n))
    
    loc_next, vel_next = _clamp(loc, vel)
    locs[0, :, :], vels[0, :, :] = loc_next, vel_next
    
    print(loc_next, vel_next)
        
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
            
            if i < 11:
                print(loc_next, vel_next)
            
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
            
        # Add noise to observations
        locs += np.random.randn(T_save, 2, n) * noise_var
        vels += np.random.randn(T_save, 2, n) * noise_var
    
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
            
    locs, vels = locs.stack(), vels.stack()        
    
    # Add noise to the observations
    locs += tf.random.normal(locs.shape, mean=0.0, stddev=noise_var, dtype=locs.dtype)
    vels += tf.random.normal(vels.shape, mean=0.0, stddev=noise_var, dtype=vels.dtype)
    
    return locs, vels 

noise_var = 0.0

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

init [[1.77918574 2.7889536  1.95386064 2.39217124 3.43384957]
 [3.26104954 3.10140545 1.92869961 3.14338321 2.81237098]] [[0.52543559 0.70066658 0.62445966 0.39977748 0.95504668]
 [0.97607643 0.8133926  0.4929475  0.69573012 0.63860918]]
eq clamp [[1.77918574 2.7889536  1.95386064 2.39217124 3.43384957]
 [3.26104954 3.10140545 1.92869961 3.14338321 2.81237098]] [[0.52543559 0.70066658 0.62445966 0.39977748 0.95504668]
 [0.97607643 0.8133926  0.4929475  0.69573012 0.63860918]]
eq [[1.77971136 2.78965421 1.95448518 2.39257097 3.43480462]
 [3.26202546 3.1022187  1.92919267 3.14407882 2.81300959]] [[0.52561533 0.70060788 0.62454317 0.39973365 0.95504668]
 [0.97591546 0.81325062 0.49306477 0.69560865 0.63860918]]
[[1.78023715 2.79035476 1.95510981 2.39297066 3.43575966]
 [3.26300121 3.10303181 1.92968586 3.14477431 2.8136482 ]]
[[1.78076313 2.79105525 1.95573452 2.39337031 3.43671471]
 [3.2639768  3.10384478 1.93017916 3.14546967 2.81428681]]
[[1.78128928 2.79175568 1.95635931 2.39376991 3.43766976]
 [3.26495224 3.1046576  1.93067257 3.14616492 2.81492542]]
[[1.78181561 2.79245605 1.95698419 2.39416947 3.4386248 ]
 [3.26592751 3.10547028 1.93116611 3.14686004 2.81556403]]
[[1.78234213 2.79315637 1.95760915 2.39456899 3.43957985]
 [3.26690262 3.10628282 1.93165976 3.14755504 2.81620264]]
[[1.78286882 2.79385662 1.9582342  2.39496846 3.4405349 ]
 [3.26787756 3.10709522 1.93215353 3.14824992 2.81684124]]
[[1.7833957  2.79455682 1.95885932 2.39536788 3.44148994]
 [3.26885235 3.10790747 1.93264741 3.14894468 2.81747985]]
[[1.78392275 2.79525696 1.95948454 2.39576727 3.44244499]
 [3.26982697 3.10871959 1.93314142 3.14963931 2.81811846]]
[[1.78444998 2.79595704 1.96010983 2.39616661 3.44340004]
 [3.27080144 3.10953156 1.93363554 3.15033383 2.81875707]]