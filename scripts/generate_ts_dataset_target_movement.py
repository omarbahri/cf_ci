"""Based on https://github.com/ethanfetaya/NRI (MIT License)."""

import os
import json
import time
import numpy as np
import argparse

import utils 
from synthetic_sim import SpringSim

def _energy(loc, vel, edges, interaction_strength):
    num_simulations, length_of_timeseries, _, num_balls = vel.shape
    
    energy_train = np.zeros((num_simulations, length_of_timeseries, num_balls))
    
    # Compute the kinetic and potential energy for each particle at each time step
    for sim in range(num_simulations):
        for t in range(length_of_timeseries):
            K = 0.5 * (vel[sim, t] ** 2).sum(axis=0)  # Kinetic energy of each ball
            
            U = np.zeros(num_balls)  # Potential energy contribution for each ball
            for i in range(num_balls):
                for j in range(num_balls):
                    if i != j:
                        r = loc[sim, t, :, i] - loc[sim, t, :, j]
                        dist = np.sqrt((r ** 2).sum())
                        U[i] += (
                            0.5
                            * interaction_strength
                            * edges[sim, i, j]
                            * (dist ** 2)
                            / 2
                        )
            # Sum kinetic and potential energy for each ball
            energy_train[sim, t] = K + U
    
    return energy_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation", type=str, default="springs", help="What simulation to generate."
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=50000,
        help="Number of training simulations to generate.",
    )
    parser.add_argument(
        "--num-valid",
        type=int,
        default=10000,
        help="Number of validation simulations to generate.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=10000,
        help="Number of test simulations to generate.",
    )
    parser.add_argument(
        "--length", type=int, default=5000, help="Length of trajectory."
    )
    parser.add_argument(
        "--sample_freq",
        type=int,
        default=100,
        help="How often to sample the trajectory.",
    )
    parser.add_argument(
        "--n_balls", type=int, default=5, help="Number of balls in the simulation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--datadir", type=str, default="data", help="Name of directory to save data to."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature of SpringSim simulation.",
    )
    parser.add_argument(
        "--temperature_dist",
        action="store_true",
        default=False,
        help="Generate with a varying latent temperature from a categorical distribution.",
    )
    parser.add_argument(
        '--temperature_alpha', 
        type=float, 
        default=None,
        help='middle category of categorical temperature distribution.'
    )
    parser.add_argument(
        '--temperature_num_cats', 
        type=int, 
        default=None,
        help='number of categories of categorical temperature distribution.'
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        default=False,
        help="Have symmetric connections (non-causal)",
    )
    parser.add_argument(
        "--fixed_particle",
        action="store_true",
        default=False,
        help="Have one particle fixed in place and influence all others",
    )
    parser.add_argument(
        "--influencer_particle",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) influences all",
    )
    parser.add_argument(
        "--confounder",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) influences at least two others",
    )
    parser.add_argument(
        "--uninfluenced_particle",
        action="store_true",
        default=False,
        help="Unobserved particle (last one) is not influence by others",
    )
    parser.add_argument(
        "--fixed_connectivity",
        action="store_true",
        default=False,
        help="Have one inherent causal structure for ALL simulations",
    )
    parser.add_argument(
        "--noise_var",
        type=float,
        default=0.0,
        help="Noise variance.",
    )
    args = parser.parse_args()

    if args.fixed_particle:
        args.influencer_particle = True
        args.uninfluenced_particle = True

    assert not (args.confounder and args.influencer_particle), "These options are mutually exclusive."

    # args.length_test = args.length * 2
    args.length_test = args.length

    print(args)
    return args

def generate_dataset(num_sims, length, sample_freq, sampled_sims=None,
                     train=True, edges=None):
    if not sampled_sims is None:
        assert len(sampled_sims) == num_sims

    loc_all = list()
    vel_all = list()
    edges_all = list()

    if args.fixed_connectivity:
        if train:
            edges = sim.get_edges(
                undirected=args.undirected,
                influencer=args.influencer_particle,
                uninfluenced=args.uninfluenced_particle,
                confounder=args.confounder
            )
            print('\x1b[5;30;41m' + "Edges are fixed to be: " + '\x1b[0m')
            print(edges)
        else:
            edges = edges
            print(edges)
    else:
        edges = None

    for i in range(num_sims):
        if not sampled_sims is None:
            sim_i = sampled_sims[i]
        else:
            sim_i = sim
        t = time.time()
        loc, vel, edges = sim_i.sample_trajectory(
            T=length,
            sample_freq=sample_freq,
            undirected=args.undirected,
            fixed_particle=args.fixed_particle,
            influencer=args.influencer_particle,
            uninfluenced=args.uninfluenced_particle,
            confounder=args.confounder,
            edges=edges
        )

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

        if not args.fixed_connectivity:
            edges = None

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    if train:
        return loc_all, vel_all, edges_all, edges
    else:
        return loc_all, vel_all, edges_all


if __name__ == "__main__":

    args = parse_args()

    if args.temperature_dist:
        categories = utils.get_categorical_temperature_prior(
                args.temperature_alpha, 
                args.temperature_num_cats, 
                to_torch=False, 
                to_cuda=False
            )
        print('Drawing from uniform categorical distribution over: ', categories)

    if args.simulation == "springs":
        if not args.temperature_dist:
            sim = SpringSim(
                noise_var=args.noise_var,
                n_balls=args.n_balls,
                interaction_strength=args.temperature,
            )
        else:
            temperature_samples_train = np.random.choice(categories, size=args.num_train)
            sims_train = [
                SpringSim(noise_var=0.0, n_balls=args.n_balls, interaction_strength=t)
                for t in temperature_samples_train
            ]
            temperature_samples_valid = np.random.choice(categories, size=args.num_valid)
            sims_valid = [
                SpringSim(noise_var=0.0, n_balls=args.n_balls, interaction_strength=t)
                for t in temperature_samples_valid
            ]
            temperature_samples_test = np.random.choice(categories, size=args.num_test)
            sims_test = [
                SpringSim(noise_var=0.0, n_balls=args.n_balls, interaction_strength=t)
                for t in temperature_samples_test
            ]
    else:
        raise ValueError("Simulation {} not implemented".format(args.simulation))

    suffix = "_" + args.simulation

    suffix += str(args.n_balls)
    
    suffix +=  '_' + '_'.join(['size', str(args.num_train), str(args.num_valid),
                        str(args.num_test)])

    if args.undirected:
        suffix += "undir"

    if args.fixed_particle:
        suffix += "_fixed"

    if args.uninfluenced_particle:
        suffix += "_uninfluenced2"

    if args.influencer_particle:
        suffix += "_influencer"

    if args.confounder:
        suffix += "_conf"

    if args.temperature != 0.1:
        suffix += "_inter" + str(args.temperature)

    if args.length != 5000:
        suffix += "_l" + str(args.length)

    if args.num_train != 50000:
        suffix += "_s" + str(args.num_train)

    if args.sample_freq != 1000:
        suffix += "_sf" + str(args.sample_freq)

    if args.fixed_connectivity:
        suffix += "_oneconnect"
        
    if args.noise_var != 0.0:
        suffix += "_noise_var" + str(args.noise_var)

    print(suffix)

    np.random.seed(args.seed)

    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train, edges = generate_dataset(
        args.num_train,
        args.length,
        args.sample_freq,
        sampled_sims=(None if not args.temperature_dist else sims_train),
    )

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, edges_valid = generate_dataset(
        args.num_valid,
        args.length,
        args.sample_freq,
        sampled_sims=(None if not args.temperature_dist else sims_valid),
        train=False,
        edges=edges
    )

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, edges_test = generate_dataset(
        args.num_test,
        args.length_test,
        args.sample_freq,
        sampled_sims=(None if not args.temperature_dist else sims_test),
        train=False,
        edges=edges
    )
    
    target_ball = 0
    
    interaction_strength = args.temperature
    
    n_balls = edges_train.shape[1]
    t_steps = loc_train.shape[1]
    
    energy_train = _energy(loc_train, vel_train, edges_train, interaction_strength)
    energy_valid = _energy(loc_valid, vel_valid, edges_valid, interaction_strength)
    energy_test = _energy(loc_test, vel_test, edges_test, interaction_strength)
    
    print('energy_test', energy_test.shape)    
    
    # Extract the final energies of the ball at index 0 for each simulation
    final_energies_0_train = energy_train[:, -1, target_ball]  # Extract final energies for the ball at index target_ball
    final_energies_0_valid = energy_valid[:, -1, target_ball]  # Extract final energies for the ball at index target_ball
    final_energies_0_test = energy_test[:, -1, target_ball]  # Extract final energies for the ball at index target_ball
    
    print('final_energies_0_test', final_energies_0_test.shape) 
    
    u = np.mean(final_energies_0_train.reshape(-1))
    std = np.std(final_energies_0_train.reshape(-1))
    
    print(u)
    print(std)
    
    thres = u + 2 * std   
    
    # Create the binary variable by comparing the final energies with the threshold
    final_energy_0_0_train = final_energies_0_train > thres
    final_energy_0_0_valid = final_energies_0_valid > thres
    final_energy_0_0_test = final_energies_0_test > thres
    
    print('final_energy_0_0_test', final_energy_0_0_test.shape)    
    
    # Convert boolean array to integer (0 or 1)
    y_train = final_energy_0_0_train.astype(int)
    y_valid = final_energy_0_0_valid.astype(int)
    y_test = final_energy_0_0_test.astype(int)
    
    print('y_test', y_test.shape)    
    
    vel_train = vel_train.transpose(0, 3, 2, 1).reshape(-1, 2 * n_balls, t_steps)
    vel_valid = vel_valid.transpose(0, 3, 2, 1).reshape(-1, 2 * n_balls, t_steps)
    vel_test = vel_test.transpose(0, 3, 2, 1).reshape(-1, 2 * n_balls, t_steps)
    
    # Reshape loc_train to shape (50000, 2 * 5, 50)
    loc_train = loc_train.transpose(0, 3, 2, 1).reshape(-1, 2 * n_balls, t_steps)
    loc_valid = loc_valid.transpose(0, 3, 2, 1).reshape(-1, 2 * n_balls, t_steps)
    loc_test = loc_test.transpose(0, 3, 2, 1).reshape(-1, 2 * n_balls, t_steps)
    
    print('loc_test', loc_test.shape)    
    
    # Concatenate along the new feature axis (should be axis=1)
    X_train = np.concatenate((loc_train, vel_train), axis=1)
    X_valid = np.concatenate((loc_valid, vel_valid), axis=1)
    X_test = np.concatenate((loc_test, vel_test), axis=1)
    
    print('X_test', X_test.shape)    

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
    dataset_path = os.path.join(root_dir, 'cf_ci', 'data', 'particles_spring', suffix + '_target_movement')
        
    if target_ball != 0:
        dataset_path = dataset_path + '_' + str(target_ball)
        
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    np.save(os.path.join(dataset_path, 'X_train.npy'), X_train)
    np.save(os.path.join(dataset_path, 'y_train.npy'), y_train)
    np.save(os.path.join(dataset_path, 'X_valid.npy'), X_valid)
    np.save(os.path.join(dataset_path, 'y_valid.npy'), y_valid)
    np.save(os.path.join(dataset_path, 'X_test.npy'), X_test)
    np.save(os.path.join(dataset_path, 'y_test.npy'), y_test)
    np.save(os.path.join(dataset_path, 'edges.npy'), edges_train[0])
    
    print(suffix + '_target_movement')