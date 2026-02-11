import robosuite as suite
import numpy as np

env = suite.make(
    "PickPlace",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=False, 
    control_freq=20,
)

env.reset()

print("Body names in the simulation:")
for body_name in env.sim.model.body_names:
    print(body_name)

print("\nSpecific Object positions:")
target_objects = ["Milk_main", "Bread_main", "Cereal_main", "Can_main", "bin1", "bin2"]
for obj in target_objects:
    try:
        id = env.sim.model.body_name2id(obj)
        pos = env.sim.data.body_xpos[id]
        print(f"{obj}: {pos}")
    except:
        print(f"{obj} not found")

# Check for bins
print("\nChecking for bins:")
for body_name in env.sim.model.body_names:
    if "bin" in body_name.lower():
        print(body_name)

env.close()
