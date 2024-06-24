from gymnasium.envs.registration import register

register(
    id='house-v0',
    entry_point='environment.battery_env:Env'
)
