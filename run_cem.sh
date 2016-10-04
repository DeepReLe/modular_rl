#!/bin/bash
REPEAT=3

# Execute CartPole-v0 for REPEAT times
#for (( i=1; i<=$REPEAT; i++ ))
#do
#  python ./run_cem.py --env=CartPole-v0 --agent=modular_rl.agentzoo.DeterministicAgent --n_iter=300 --solved=195.0
#done

# MountainCar-v0
for (( i=1; i<=$REPEAT; i++ ))
do
  python ./run_cem.py --env=MountainCar-v0 --agent=modular_rl.agentzoo.DeterministicAgent --n_iter=300 --solved=-110.0
done

# Acrobot-v1
for (( i=1; i<=$REPEAT; i++ ))
do
  python ./run_cem.py --env=Acrobot-v1 --agent=modular_rl.agentzoo.DeterministicAgent --n_iter=300 --solved=-100.0
done

# Pendulum-v0
for (( i=1; i<=$REPEAT; i++ ))
do
  python ./run_cem.py --env=Pendulum-v0 --agent=modular_rl.agentzoo.DeterministicAgent --n_iter=300 --solved=-150.0
done

# InvertedPendulum-v1
for (( i=1; i<=$REPEAT; i++ ))
do
  python ./run_cem.py --env=InvertedPendulum-v1 --agent=modular_rl.agentzoo.DeterministicAgent --n_iter=300 --solved=950.0
done

# InvertedDoublePendulum-v1
for (( i=1; i<=$REPEAT; i++ ))
do
  python ./run_cem.py --env=InvertedDoublePendulum-v1 --agent=modular_rl.agentzoo.DeterministicAgent --n_iter=300 --solved=9100.0
done

# Reacher-v1
for (( i=1; i<=$REPEAT; i++ ))
do
  python ./run_cem.py --env=Reacher-v1 --agent=modular_rl.agentzoo.DeterministicAgent --n_iter=300 --solved=-3.75
done

# Swimmer-v1
for (( i=1; i<=$REPEAT; i++ ))
do
  python ./run_cem.py --env=Swimmer-v1 --agent=modular_rl.agentzoo.DeterministicAgent --n_iter=300 --solved=360.0
done




