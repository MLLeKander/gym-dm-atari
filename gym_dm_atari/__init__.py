from gym.envs.registration import registry as _registry, register as _register
from gym_dm_atari.dm_atari_env import DMAtariEnv

_identifier = 'Deterministic-v0'

for spec in _registry.env_specs.values():
    if spec._entry_point == 'gym.envs.atari:AtariEnv' and spec.id.endswith(_identifier):
        name = spec.id[:-len(_identifier)]
        game = spec._kwargs['game']

        _register(
            id='DM-%s-v0'%name,
            entry_point='gym_dm_atari:DMAtariEnv',
            kwargs={'game':game},
            nondeterministic=False
        )

        _register(
            id='DMEval-%s-v0'%name,
            entry_point='gym_dm_atari:DMAtariEnv',
            kwargs={'game':game, 'life_episode':False},
            nondeterministic=False
        )
