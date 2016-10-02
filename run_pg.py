#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""


from gym.envs import make
from gym.spaces import Box
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os, logging
import gym

global PAPER_STATS

PAPER_STATS = {
        'iter_time': [],
        'iter_mean': [],
        'iter_validation': [],
        'final_score': 0.0,
        'nb_params:': 0.0,
        'state_size': 0.0,
        'action_size': 0.0,
        'continuous': False,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)    
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env = make(args.env)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    # env.monitor.start(mondir, video_callable=None if args.video else VIDEO_NEVER)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0: 
        args.timestep_limit = env_spec.timestep_limit    
    cfg = args.__dict__
    np.random.seed(args.seed)
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)


    PAPER_STATS['nb_params'] = np.prod(env.observation_space.shape)
    if isinstance(env.observation_space, Box):
        PAPER_STATS['continuous'] = True
        PAPER_STATS['state_size'] = np.prod(env.observation_space.shape)
    else:
        PAPER_STATS['state_size'] = env.observation_space.n
    if isinstance(env.action_space, Box):
        PAPER_STATS['continuous'] = True
        PAPER_STATS['action_size'] = np.prod(env.action_space.shape)
    else:
        PAPER_STATS['action_size'] = env.action_space.n

    COUNTER = 0
    def callback(stats):
        global COUNTER
        COUNTER += 1  
        # Print stats
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k,v) : np.asarray(v).size==1, stats.items())) #pylint: disable=W0110
        # Store to hdf5
        if args.use_hdf:
            for (stat,val) in stats.items():
                if np.asarray(val).ndim==0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)): 
                hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
        # Plot
        if args.plot:
            animate_rollout(env, agent, min(500, args.timestep_limit))

    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg = cfg, PAPER_STATS=PAPER_STATS)

    if args.use_hdf:
        hdf['env_id'] = env_spec.id
        try: hdf['env'] = np.array(cPickle.dumps(env, -1))
        except Exception: print "failed to pickle env" #pylint: disable=W0703
    
    # env.monitor.close()


    PAPER_STATS['final_score'] = PAPER_STATS['iter_validation'][-1]
    fname = 'results/results_trpo_' + args.env + '_' + str(PAPER_STATS['nb_params']) + '_' + str(args.n_iter) + '_' + str(args.cg_damping) + '.pkl'
    with open(fname, 'wb') as f:
        cPickle.dump(PAPER_STATS, f)
