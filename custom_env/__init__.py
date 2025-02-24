import os

from gymnasium.envs.registration import register

ENV_IDS = []

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        env_id = f"CustomPickAndPlace{control_suffix}{reward_suffix}-v3"

        register(
            id=env_id,
            entry_point=f"custom_env.custom_env:CustomPickAndPlaceEnv",
            kwargs={"reward_type": reward_type, "control_type": control_type},
            max_episode_steps=50,
        )

        ENV_IDS.append(env_id)
