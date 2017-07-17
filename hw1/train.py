import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--envname', type=str, default='Hopper-v1')
args = parser.parse_args()
mode=args.envname

# load data
expert_data = pickle.load(open('data/%s.pkl'%mode,'r'))
dag_data = pickle.load(open('data/%s-dag.pkl'%mode,'r'))
expert_data['observations'] = np.concatenate((expert_data['observations'],\
                                              dag_data['observations']))
expert_data['actions'] = np.concatenate((expert_data['actions'],\
                                              dag_data['actions']))

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

lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[36000,72000], factor=0.5)
optimizer=mx.optimizer.SGD(learning_rate=0.001,
                            momentum=0.9,
                            lr_scheduler = lr_scheduler,
                            rescale_grad = 1./batch_size)
model.fit(train_iter, eval_iter,
          num_epoch=500,
          optimizer = optimizer,
          eval_metric='mse',
          batch_end_callback = mx.callback.Speedometer(batch_size,1000),
          epoch_end_callback  = mx.callback.do_checkpoint("model/%s-mlp"%mode))
