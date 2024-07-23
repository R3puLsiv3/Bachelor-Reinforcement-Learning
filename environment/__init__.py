from gymnasium.envs.registration import register

register(
    id='house_base',
    entry_point='environment.battery_env:EnvBase'
)

register(
    id='house_timestamp',
    entry_point='environment.battery_env:EnvTimestamp'
)

register(
    id='house_radiation',
    entry_point='environment.battery_env:EnvRadiation'
)

register(
    id='house_timestamp_radiation',
    entry_point='environment.battery_env:EnvTimestampRadiation'
)
