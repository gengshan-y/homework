import logging
from mlp import mlp
import mxnet as mx
import numpy as np
import pdb
import pickle
from sklearn.model_selection import train_test_split
import time
logging.basicConfig(filename='log/' + time.strftime("%m_%d_%H_%M") + '.log',\
                    level=logging.DEBUG)


expert_data = pickle.load(open('data/Hopper-v1.pkl','r'))
tr_data,val_data,tr_label,val_label = train_test_split(expert_data['observations'], \
                                      expert_data['actions'], train_size=0.7)
batch_size=1
train_iter = mx.io.NDArrayIter(tr_data,tr_label, batch_size, shuffle=True,\
                               label_name='label')
eval_iter = mx.io.NDArrayIter(val_data, val_label, batch_size, shuffle=False)

data = mx.sym.Variable('data'); label = mx.sym.Variable('label')
symbol = mlp(data,label,nlayers=3, fcns=[512,512,3])
mx.viz.plot_network(symbol).render('net_plot')
model = mx.mod.Module(symbol=symbol,
                      data_names=['data'],
                      label_names=['label'],
                      context = mx.gpu(0))

model.fit(train_iter, eval_iter,
          optimizer_params={'learning_rate':0.002, 'momentum': 0.9},
          num_epoch=200,
          eval_metric='mse',
          batch_end_callback = mx.callback.Speedometer(batch_size,100),
          epoch_end_callback  = mx.callback.do_checkpoint("model/pl-mlp"),)
