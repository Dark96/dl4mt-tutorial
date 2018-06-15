#!/bin/bash


export THEANO_FLAGS=device=cpu,floatX=float32

python ./translate.py -n -p 8 \
	$HOME/models/model_session2.npz  \
	$HOME/codes/dl4mt-tutorial/data/europarl-v7.fr-en.en.tok.pkl \
	$HOME/codes/dl4mt-tutorial/data/europarl-v7.fr-en.fr.tok.pkl \
	$HOME/codes/dl4mt-tutorial/data/newstest2011.en.tok \
	./newstest2011.trans.fr.tok



