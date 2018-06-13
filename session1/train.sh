#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=168:00:00
#PBS -N session1_default
#PBS -A course
#PBS -q GpuQ

export THEANO_FLAGS=device=cuda0,floatX=float32

python ./train_nmt.py



