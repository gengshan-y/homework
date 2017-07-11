import logging
import gym
from mlp import mlp
import mxnet as mx
import numpy as np
import pdb
import time


epoch=200

data = mx.sym.Variable('data'); label = mx.sym.Variable('label')
symbol = mlp(data,label,nlayers=3, fcns=[512,512,3])
mx.viz.plot_network(symbol).render('net_plot')
model = mx.mod.Module(symbol=symbol,
                      data_names=['data'],
                      label_names=['label'],
                      context = mx.gpu(0))

_, arg_params, aux_params = mx.model.load_checkpoint('model/pl-mlp',epoch)
model.bind(data_shapes=[('data',(1,11))], label_shapes=[('label',(1,1,3))], for_training=False)
model.set_params(arg_params = arg_params, aux_params=aux_params)

pdb.set_trace()
env=gym.make('Hopper-v1')
max_steps = env.spec.timestep_limit

returns = []; observations = []; actions = []
obs = env.reset()
done=False
totalr=0.
steps=0
while 1:#not done:
  batch=mx.io.NDArrayIter(data={'data':obs[None,:]})
  action = model.predict(batch).asnumpy()
  actions.append(action)
  observations.append(obs)
  obs,r,done,_ = env.step(action)
  print done
  totalr+=r
  steps+=1
  env.render()
  if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
  if steps >= max_steps: 
    break
  time.sleep(0.1)
returns.append(totalr)
