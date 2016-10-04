#!/usr/bin/env python

import numpy as np
import os
import cPickle as pk
from seb.plot import Plot

CONTINUOUS_FACTOR = 3.0

if __name__ == '__main__':
    p = Plot()
    p.set_axis('Complexity', 'Time To Convergence')

    trpo = []
    cem = []

    trpo_conv = []
    trpo_compl = []
    cem_conv = []
    cem_compl = []
    
    for fname in os.listdir('./results'):
        with open('results/' + fname, 'rb') as f:
            if 'trpo' in fname:
                res = pk.load(f)
                compl = res['state_size'] + res['action_size']
                if res['continuous']:
                    compl *= CONTINUOUS_FACTOR
                conv = len(res['iter_mean'])
                trpo_compl.append(compl)
                trpo_conv.append(conv)
                trpo.append((compl, conv))
            if 'cem' in fname:
                results = []
                compl = 0.0
                conv = 0.0
                while True:
                    try:
                        res = pk.load(f)
                        results.append(res)
                        conv += len(res['iter_mean'])
                        compl += res['state_size'] + res['action_size']
                    except EOFError:
                        break
                if res['continuous']:
                    compl *= CONTINUOUS_FACTOR
                conv /= len(results)
                compl /= len(results)
                cem_compl.append(compl)
                cem_conv.append(conv)
                cem.append((compl, conv))
            print '\n', '-' * 20
            print fname
            print 'Convergence: ', conv
            print 'Complexity: ', compl

    # p.plot(x=trpo_compl, y=trpo_conv, label='TRPO', jitter=5.0)
    # p.plot(x=cem_compl, y=cem_conv, label='CEM', jitter=5.0)

    compl_sort = lambda x: x[0]
    trpo = sorted(trpo, key=compl_sort)
    cem = sorted(cem, key=compl_sort)

    trpo = np.array(trpo)
    cem = np.array(cem)
    p.plot(x=cem[:, 0], y=cem[:, 1], label='TRPO', jitter=5.0)
    p.plot(x=cem[:, 0], y=cem[:, 1], label='CEM', jitter=5.0)
                
# ./run_pg.py --env MountainCar-v0 --agent modular_rl.agentzoo.TrpoAgent --n_iter=300 --solved=-110.0


    p.save('results.svg')
