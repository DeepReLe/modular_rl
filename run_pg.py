#!/usr/bin/env python
"""
This script runs a policy gradient algorithm
"""


from gym.envs import make
from modular_rl import *
import argparse, sys, cPickle
from tabulate import tabulate
import shutil, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)    
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    rawenv = make(args.env)
    env_spec = rawenv.spec
    mondir = "/tmp/asdf"
    if os.path.exists(mondir): shutil.rmtree(mondir)
    os.mkdir(mondir)
    rawenv.monitor.start(mondir, "TRPO")
    env = FilteredEnv(rawenv, ZFilter(rawenv.observation_space.shape, clip=5), ZFilter((), demean=False, clip=10))
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    update_argument_parser(parser, PG_OPTIONS)
    args = parser.parse_args()
    cfg = args.__dict__
    np.random.seed(args.seed)
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    hdf, diagnostics = prepare_h5_file(args)

    if args.max_pathlength == 0: 
        args.max_pathlength = env_spec.timestep_limit

    COUNTER = 0
    def callback(stats):
        global COUNTER
        for (stat,val) in stats.items():
            if np.asarray(val).ndim==0:
                diagnostics[stat].append(val)
            else:
                assert val.ndim == 1
                diagnostics[stat].extend(val)
        if args.plot:
            animate_rollout(env, agent, min(500, args.max_pathlength))
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k,v) : np.asarray(v).size==1, stats.items())) #pylint: disable=W0110
        COUNTER += 1
        if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)): 
            hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
    run_policy_gradient_algorithm(env, agent.policy, agent.updater, agent.baseline, 
        callback=callback, usercfg = cfg)

    hdf['env_id'] = env_spec.id
    hdf['ob_filter'] = np.array(cPickle.dumps(env.ob_filter, -1))
    try: hdf['env'] = np.array(cPickle.dumps(env, -1))
    except Exception: print "failed to pickle env"
    # env.monitor.close()