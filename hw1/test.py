import argparse
import gym
import logging
from mlp import mlp
import mxnet as mx
import numpy as np
import pickle
import pdb
import time

parser = argparse.ArgumentParser()
parser.add_argument('--envname', type=str, default='Hopper-v1')
parser.add_argument('--render', action='store_true')
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--act_dim', type=int, default=3)
parser.add_argument('--obs_dim', type=int, default=11)
args = parser.parse_args()


data = mx.sym.Variable('data'); label = mx.sym.Variable('label')
symbol = mlp(data,label,nlayers=3, fcns=[512,512,args.act_dim])
mx.viz.plot_network(symbol).render('net_plot')
model = mx.mod.Module(symbol=symbol,
                      data_names=['data'],
                      label_names=['label'],
                      context = mx.gpu(0))

_, arg_params, aux_params = mx.model.load_checkpoint('model/%s-mlp'%args.envname,args.epoch)
model.bind(data_shapes=[('data',(1,args.obs_dim))],\
           label_shapes=[('label',(1,1,args.act_dim))], for_training=False)
model.set_params(arg_params = arg_params, aux_params=aux_params)


env=gym.make(args.envname)
max_steps = env.spec.timestep_limit
returns = []; observations = []
for i in range(args.num_rollouts):
  print('iter',i)
  obs = env.reset()
  done=False
  totalr=0.
  steps=0
  while not done:
    batch=mx.io.NDArrayIter(data={'data':obs[None,:]})
    action = model.predict(batch).asnumpy()  # net pl
    observations.append(obs)
    obs,r,done,_ = env.step(action)
    totalr+=r
    steps+=1
    if args.render:
      env.render()
    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
    if steps >= max_steps: 
      break
    time.sleep(0.01)
  returns.append(totalr)
print('returns', returns)
print('mean return', np.mean(returns))
print('std of return', np.std(returns))

pickle.dump(observations, open('data/%s-obs.pkl'%args.envname,'w'))
