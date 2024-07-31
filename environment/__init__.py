from gymnasium.envs.registration import register

register(
    id='house_base',
    entry_point='environment.battery_env:EnvBase'
)

register(
    id='house_base_test',
    entry_point='environment.battery_env:EnvTest'
)

register(
    id='house_timestamp',
    entry_point='environment.battery_env:EnvTimestamp'
)
