import argparse
import load_policy
import numpy as np
import pdb
import pickle
import tensorflow as tf
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str,default='Ant-v1')
args=parser.parse_args()
pdb.set_trace()
policy_fn = load_policy.load_policy('experts/%s.pkl'%args.envname)
observations = pickle.load(open('data/%s-obs.pkl'%args.envname,'r'))

with tf.Session():
  tf_util.initialize()
  actions=[]
  for obs in observations:
    action_e = policy_fn(obs[None,:])  # exp pl
    actions.append(action_e)

expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
pickle.dump(expert_data, open('data/%s-dag.pkl'%args.envname,'w'))
