import multiprocessing
import os

from stable_baselines import logger
from stable_baselines.bench import monitor
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.ppo2 import ppo2
import gfootball.env as football_env
from gfootball.examples import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Google Research Football")
    # OPTIONAL ARGS
    parser.add_argument('-env', type=str, default='academy_empty_goal_close', help="Football Scenario to use, e.g.:\n\
                                                            academy_empty_goal_close\n11_vs_11_stochastic\n\
                                                            academy_corner")
    parser.add_argument('-state', type=str, default='extracted_stacked', help="Observation to be used for training, e.g.\nextracted\nextracted_stacked")
    parser.add_argument('-reward_experiment', type=str, default='scoring', help="Reward to be used for training, e.g.\nscoring\ncheckpoints")
    parser.add_argument('-policy', type=str, default='cnn', help="Policy Architecture, e.g.\ncnn\nlstm\nmlp\nimpala_cnn\ngfootball_impala_cnn")
    parser.add_argument('-num_timesteps', type=int, default=2e6, help="Number of timesteps to run for.")
    parser.add_argument('-num_envs', type=int, default=8, help="Number of environments to run in parallel.")
    parser.add_argument('-nsteps', type=int, default=128, help="Number of environment steps per epoch; ''batch size is nsteps * nenv")
    parser.add_argument('-eps', type=float, default=1.0, help="Starting value for epsilon in epsilon-greedy")
    parser.add_argument('-max_mem', type=int, default=75000, help="Maximum size for memory replay buffer")
    parser.add_argument('-repeat', type=int, default=4, help="Number of frames to repeat & stack")
    parser.add_argument('-bs', type=int, default=32, help="Batch size for memory replay sampling")
    parser.add_argument('-replace', type=int, default=1000, help="Interval for replacing target network")

    parser.add_argument('-gpu', type=str, default='1', help="GPU: 0 or 1")