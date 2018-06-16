import numpy
import os

import numpy
import os

from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim=params['dim'][0],
                     n_words=params['n-words'][0],
                     n_words_src=params['n-words'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0],
                     patience=1000,
                     maxlen=50,
                     batch_size=32,
                     valid_batch_size=32,
                     validFreq=100,
                     dispFreq=10,
                     saveFreq=10000,
                     sampleFreq=100,
                     datasets=['/home/ubuntu/codes/dl4mt-tutorial/data/europarl-v7.fr-en.en.tok',
                               '/home/ubuntu/codes/dl4mt-tutorial/data/europarl-v7.fr-en.fr.tok'],
                     valid_datasets=['/home/ubuntu/codes/dl4mt-tutorial/data/newstest2011.en.tok',
                                     '/home/ubuntu/codes/dl4mt-tutorial/data/newstest2011.fr.tok'],
                     dictionaries=['/home/ubuntu/codes/dl4mt-tutorial/data/europarl-v7.fr-en.en.tok.pkl',
                                   '/home/ubuntu/codes/dl4mt-tutorial/data/europarl-v7.fr-en.fr.tok.pkl'],
                     use_dropout=params['use-dropout'][0],
                     overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/home/ubuntu/models/model_hal.npz'],
        'dim_word': [512],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'clip-c': [1.],
        'use-dropout': [False],
        'learning-rate': [0.003],
        'reload': [True]})
