import os

from nmt import train


def main(job_id, params):
    print params
    validerr = train(
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_word=params['dim_word'][0],
        dim=params['dim'][0],
        n_words=params['n-words'][0],
        n_words_src=params['n-words'][0],
        decay_c=params['decay-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        maxlen=50,
        batch_size=32,
        valid_batch_size=32,
        datasets=[
            '/home/ubuntu/codes/dl4mt-tutorial/data/all.en.concat.shuf.gz',
            '/home/ubuntu/codes/dl4mt-tutorial/data/all.fr.concat.shuf.gz'],
        valid_datasets=[
            '/home/ubuntu/codes/dl4mt-tutorial/data/newstest2011.en.tok',
            '/home/ubuntu/codes/dl4mt-tutorial/data/newstest2011.fr.tok'],
        dictionaries=[
            '/home/ubuntu/codes/dl4mt-tutorial/data/all.en.concat.gz.pkl',
            '/home/ubuntu/codes/dl4mt-tutorial/data/all.fr.concat.gz.pkl'],
        validFreq=5000,
        dispFreq=10,
        saveFreq=5000,
        sampleFreq=1000,
        use_dropout=params['use-dropout'][0],
        overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': [
            '/home/ubuntu/models/model_session1_all.npz'],
        'dim_word': [500],
        'dim': [1024],
        'n-words': [30000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.003],
        'reload': [False]})
