import mxnet as mx

def mlp(data, label, nlayers=2, fcns=None):
  fcs=[]; acts=[]
  if fcns is None: fcns=[512]*(nlayers+1)
  fc = mx.sym.FullyConnected(data=data,name='fc1',num_hidden=fcns[0])
  fcs += [fc]

  for i in range(nlayers-1):
    act = mx.sym.Activation(data=fcs[i], name='relu%d'%(i+1), \
                            act_type='relu')
    acts += [act]
    fc = mx.sym.FullyConnected(data=acts[i], name='fc%d'%(i+2), \
                               num_hidden=fcns[i+1])
    fcs += [fc]

  lro = mx.sym.LinearRegressionOutput(data=fcs[-1],label=label,name='lro')
  return lro

