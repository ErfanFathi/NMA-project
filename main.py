from model import save_ckpt_to_local, restore_ckpt_from_local
from plot import save_video, reward_plot
from environment import Hopper, Ant
from agent import create_agent
import argparse

from acme.utils import loggers
from acme import wrappers, specs, environment_loop

parser = argparse.ArgumentParser()
parser.add_argument("--logging_frequency", default=60.)
parser.add_argument("--env", default="Hopper") # Hopper
parser.add_argument("--agent", default="D4PG") # D4PG, DDPG, DDPO 
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--save_plot', action='store_true')
parser.add_argument("--lr", default=1e-4)
parser.add_argument("--steps", default=1000, type=int) # 100_000  # Number of environment loop steps. Adjust as needed!
args = parser.parse_args()

if args.env == "Hopper":
  env = Hopper(render=False)
elif args.env == "Ant":
  env = Ant(render=False)
else:
  raise ValueError("{} Environment Not Found".format(args.env))

env = wrappers.GymWrapper(env)
env = wrappers.SinglePrecisionWrapper(env)


learner_logger = loggers.TerminalLogger(label='Learner',
                                        time_delta=args.logging_frequency,
                                        print_fn=print)
loop_logger = loggers.TerminalLogger(label='Environment Loop',
                                     time_delta=args.logging_frequency,
                                     print_fn=print)


# Create agent.
agent = create_agent(specs.make_environment_spec(env),
                      env.action_spec(),
                      args.lr,
                      learner_logger,
                      args.save_model,
                      args.agent)

if args.load_model:
  restore_ckpt_from_local(agent)

loop = environment_loop.EnvironmentLoop(env, agent, logger=loop_logger)

# Start training!
loop.run(num_episodes=None,
         num_steps=args.steps)

if args.save_plot:
  save_video(agent, env)
  reward_plot(agent, env)

if args.save_model:
  save_ckpt_to_local(agent)