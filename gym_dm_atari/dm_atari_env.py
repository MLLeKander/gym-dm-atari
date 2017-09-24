from gym.envs.atari import AtariEnv
import numpy as np
import gym
import cv2
import logging

logger = logging.getLogger(__name__)

class DMAtariEnv(AtariEnv):
    def __init__(self, game, use_greyscale=True, hist_len=4, action_repeat=4, pool_len=2,
                 noop_max=30, rescale_height=84, rescale_width=84, life_episode=True,
                 render_scale=3, render_hist=False):
        AtariEnv.__init__(self, game, obs_type='image', frameskip=1,
                          repeat_action_probability=0.)

        self.use_greyscale = use_greyscale
        self.action_repeat = action_repeat
        self.noop_max = noop_max
        self.life_episode = life_episode
        self.render_scale = render_scale
        self.render_hist = render_hist

        (self.orig_screen_width, self.orig_screen_height) = self.ale.getScreenDims()
        self.screen_width = rescale_width if rescale_width > 0 else self.orig_screen_width
        self.screen_height = rescale_height if rescale_height > 0 else self.orig_screen_height

        self.use_rescale = rescale_width > 0 or rescale_height > 0

        self.img_dims = 1 if self.use_greyscale else 3
        pool_shape = (pool_len, self.orig_screen_height, self.orig_screen_width, 3)
        hist_shape = (hist_len, self.screen_height, self.screen_width, self.img_dims)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=hist_shape)

        self.pool_fbuff = np.zeros(pool_shape, dtype=np.uint8)
        self.hist_fbuff = np.zeros(hist_shape, dtype=np.uint8)
        self.hist_fbuff_readonly = self.hist_fbuff.view()
        self.hist_fbuff_readonly.setflags(write=False)
        self._setup_image_pipeline()

    def _setup_image_pipeline(self):
        self.pool_fbuff_out = self.pool_fbuff[-1,:,:,:]
        self.hist_fbuff_in = self.hist_fbuff[0,:,:,:]

        if self.use_greyscale:
            self.greyscale_in = self.pool_fbuff_out
            if self.use_rescale:
                greyscale_out_shape = (self.orig_screen_height, self.orig_screen_width, 1)
                self.greyscale_out = np.zeros(greyscale_out_shape, dtype=np.uint8)
                self.rescale_in = self.greyscale_out
                self.rescale_out = self.hist_fbuff_in
            else:
                self.greyscale_out = self.hist_fbuff_in
        elif self.use_rescale:
            self.rescale_in = self.pool_fbuff_out
            self.rescale_out = self.hist_fbuff_in
        else:
            self.pool_fbuff_out = self.hist_fbuff_in

    def _reset(self):
        self.ale.reset_game()
        done = True
        while done:
            done = False
            n_repeat = self.np_random.randint(0,self.noop_max)

            self.pool_fbuff.fill(0)
            self._pool_fbuff_push()
            _, _, done, _ = self._step(0, action_repeat=n_repeat)
            if done:
                logger.warn('Episode terminated during initial no-ops... retrying.')

        self.hist_fbuff.fill(0)
        self._hist_fbuff_push()

        return self._get_obs()

    #TODO: It's possible to make this a circular queue... but not today.
    def _pool_fbuff_push(self):
        self.pool_fbuff[1:,:,:,:] = self.pool_fbuff[:-1,:,:,:]
        self.ale.getScreenRGB(self.pool_fbuff[0,:,:,:])

    def _hist_fbuff_push(self):
        self.hist_fbuff[1:,:,:,:] = self.hist_fbuff[:-1,:,:,:]

        #slow_max = np.max(self.pool_fbuff, axis=0)
        # Even if this gets called twice, it'll be okay since max is idempotent. It'll
        # still be pretty slow though...
        np.max(self.pool_fbuff, axis=0, out=self.pool_fbuff_out)
        #assert(np.array_equal(slow_max,self.pool_fbuff_out))

        if self.use_greyscale:
            cv2.cvtColor(self.greyscale_in, cv2.COLOR_RGB2GRAY, dst=self.greyscale_out)
        if self.use_rescale:
            cv2.resize(self.rescale_in, (self.screen_width, self.screen_height),
                       interpolation=cv2.INTER_LINEAR, dst=self.rescale_out)

    def _step(self, a, action_repeat=None):
        reward = 0.0
        action = self._action_set[a]
        if action_repeat is None:
            action_repeat = self.action_repeat
        prev_lives = self.ale.lives()

        for _ in range(action_repeat):
            reward += self.ale.act(action)
            self._pool_fbuff_push()
            if self._is_over(prev_lives):
                break
        self._hist_fbuff_push()
        return self._get_obs(), reward, self._is_over(prev_lives), None

    def _is_over(self, prev_lives):
        if self.ale.game_over():
            return True
        return self.life_episode and prev_lives > self.ale.lives()

    def _get_obs(self):
        return self.hist_fbuff_readonly

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode != 'human':
            raise ValueError('Rendering mode must be \'human\'')

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        if self.render_hist:
            img = np.hstack(self._get_obs())
        else:
            img = self._get_obs()[0]

        if self.use_greyscale:
            img = img.repeat(3,axis=-1)

        self.viewer.imshow(np.repeat(np.repeat(img, self.render_scale, axis=0), self.render_scale, axis=1))

    #TODO: State management is more complex, since we have to take into account
    # frame history... This should be revisited later.
    def clone_state(self):
        raise NotImplementedError()
    def restore_state(self):
        raise NotImplementedError()
    def clone_full_state(self):
        raise NotImplementedError()
    def restore_full_state(self):
        raise NotImplementedError()
