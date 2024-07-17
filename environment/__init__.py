from gymnasium.envs.registration import register

register(
    id='house-v0',
    entry_point='environment.battery_env:EnvBase'
)

register(
    id='house-v1',
    entry_point='environment.battery_env:EnvTimestamp'
)

register(
    id='house-v2',
    entry_point='environment.battery_env:EnvRadiation'
)

register(
    id='house-v3',
    entry_point='environment.battery_env:EnvTimestampRadiation'
)
