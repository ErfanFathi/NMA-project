import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os

def cal_frames_reward(agent, env):
  # Run the environment with the learned policy and display video.
  n_steps = 500
  
  frames = []  # Frames for video.
  reward = [[]]  # Reward at every timestep.
  timestep = env.reset()
  for _ in range(n_steps):
    frames.append(env.environment.render(mode='rgb_array').copy())
    action = agent.select_action(timestep.observation)
    timestep = env.step(action)
  
    # `timestep.reward` is None when episode terminates.
    if timestep.reward:
      # Old episode continues.
      reward[-1].append(timestep.reward.item())
    else:
      # New episode begins.
      reward.append([])

  return frames, reward

def save_video(agent, env, framerate=30):
  """Generates video from `frames`.

  Args:
    frames (ndarray): Array of shape (n_frames, height, width, 3).
    framerate (int): Frame rate in units of Hz.

  Returns:
    Display object.
  """

  if not os.path.exists("./utils"):
    os.makedirs("./utils")

  # Get frames
  frames, _ = cal_frames_reward(agent, env)

  height, width, _ = frames[0].shape
  dpi = 70
  orig_backend = matplotlib.get_backend()
  matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
  fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
  matplotlib.use(orig_backend)  # Switch back to the original backend.
  ax.set_axis_off()
  ax.set_aspect('equal')
  ax.set_position([0, 0, 1, 1])
  im = ax.imshow(frames[0])
  def update(frame):
    im.set_data(frame)
    return [im]
  interval = 1000/framerate
  anim = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False)

  # saving to m4 using ffmpeg writer
  writervideo = animation.FFMpegWriter(fps=framerate)
  anim.save('./utils/video.mp4', writer=writervideo)

def reward_plot(agent, env):
  if not os.path.exists("./utils"):
    os.makedirs("./utils")

  # Get Reward
  _, reward = cal_frames_reward(agent, env)
  env_step = 0
  for episode in reward:
    plt.plot(np.arange(env_step, env_step+len(episode)), episode)
    env_step += len(episode)
  plt.xlabel('Timestep', fontsize=14)
  plt.ylabel('Reward', fontsize=14)
  plt.grid()
  plt.savefig('./utils/plot.png', bbox_inches='tight')

  for i, episode in enumerate(reward):
    print(f"Total reward in episode {i}: {sum(episode):.2f}")
