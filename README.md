# gym-dm-atari
Atari RL environment which (attempts to) replicate the setup described in DeepMind's 2015 Nature article.

# Usage

```
import gym, gym_dm_atari
env = gym.make('DM-MsPacman-v0')

env.reset()
env.step(0)
env.render()
```

This package searches through previously-registered environments, and requires `gym[atari]` to be installed.

Environments are given a 'DM-' or 'DMEval-' prefix. The only difference is that 'DM-' environments terminate after one life is lost, whereas the 'DMEval-' prefix terminates at game over.

# Features

 - Max pooling over previous frames (default pool length: 2)
 - Frame greyscaling (default: enabled)
 - Frame rescaling (default size: 84x84)
 - History of the most recent observations (default: 4)
 - No-op initialization (default no-op max: 30)
 - Image upscaled during rendering (default: 3x)

States returned by `step` and `render` have the following shape:

`(hist_len, screen_height, screen_width, image_dim)`

`image_dim` will be 1 if greyscaling is used, or 3 (RGB) otherwise.
