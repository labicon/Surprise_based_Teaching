import argparse
from garage import rollout
from garage.experiment import Snapshotter

snapshotter = Snapshotter()

parser = argparse.ArgumentParser()
parser.add_argument('--itr', type=int, default=999)
parser.add_argument('--alg', type=str, default='curriculum')
parser.add_argument('--env', type=str, default='Sparse_HalfCheetah_Diffspeed')

args = parser.parse_args()

data_dir = '../experiments/' + args.env + '/' + args.alg
itr = args.itr
data = snapshotter.load(data_dir, itr=itr)

policy = data['algo'].policy
# You can also access other components of the experiment
env = data['env']

path = rollout(env, policy, animated=True, pause_per_frame=0.01)

# done = False
# x = env.reset()
# env.render(mode='human')
# state = x[0]
# while not done:
#     action = policy.get_action(state)[0]
#     x = env.step(action)
#     state = x.observation
#     done = x.terminal
# np.save('load_rollout.npy', path)