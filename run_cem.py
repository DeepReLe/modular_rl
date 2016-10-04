#!/usr/bin/env python
"""
This script runs the cross-entropy method
"""

from gym.envs import make
from gym.spaces import Box
from modular_rl import *
import argparse, sys, cPickle, shutil
import gym, logging

from tabulate import tabulate

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
    parser.add_argument("--solved",required=True)
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env = make(args.env)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    # env.monitor.start(mondir,video_callable=None if args.video else VIDEO_NEVER)
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    update_argument_parser(parser, CEM_OPTIONS)
    args = parser.parse_args()
    cfg = args.__dict__
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    np.random.seed(args.seed)
    hdf, diagnostics = prepare_h5_file(args)

    if args.timestep_limit == 0: 
        args.timestep_limit = env_spec.timestep_limit

    gym.logger.setLevel(logging.WARN)

    env.solved_threshold = float(args.solved)
    PAPER_STATS['_solved_threshold'] = env.solved_threshold

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
        for (stat,val) in stats.items():
            diagnostics[stat].append(val)
        if args.plot:
            animate_rollout(env, agent, min(500, args.timestep_limit))
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k,v) : np.asarray(v).size==1, stats.items())) #pylint: disable=W0110
        COUNTER += 1
        #if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)): 
        #    hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))


    run_cem_algorithm(env, agent, callback=callback, usercfg = cfg, PAPER_STATS=PAPER_STATS)

    #hdf['env_id'] = env_spec.id
    #try: hdf['env'] = np.array(cPickle.dumps(env, -1))
    #except Exception: 
    #    print "failed to pickle env" #pylint: disable=W0703
    #    raise
    # env.monitor.close()

    PAPER_STATS['final_score'] = PAPER_STATS['iter_validation'][-1]
    fname = 'results/results_cem_' + args.env + '_' + str(PAPER_STATS['nb_params']) + '_' + str(args.n_iter) + '_' + str(args.batch_size) + '.pkl'
    with open(fname, 'ab') as f:
        cPickle.dump(PAPER_STATS, f)
