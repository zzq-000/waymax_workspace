import numpy as np
import mediapy
from tqdm import tqdm
import dataclasses


from waymax import config as _config
from waymax.config import DatasetConfig, DataFormat
from waymax import dataloader
from waymax import datatypes
from waymax import visualization


myconfig = DatasetConfig(
    # path='./training.tfrecord-00100-of-01000',
    # path='gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000',
    # path="/home/yqnj/testing/testing.tfrecord@150",
    # path = './training.tfrecord@1',
    path = "./uncompressed_tf_example_training_training_tfexample.tfrecord@1000",
    max_num_rg_points=20000,
    data_format=DataFormat.TFRECORD,
)

config = dataclasses.replace(myconfig, max_num_objects=32)

print(type(config))
data_iter = dataloader.simulator_state_generator(config=config)

scenario = next(data_iter)

# img = visualization.plot_simulator_state(scenario, use_log_traj = True)

# mediapy.show_image(img)
# print("hello")
imgs = []

state = scenario
for _ in range(scenario.remaining_timesteps):
  state = datatypes.update_state_by_log(state, num_steps=1)
  imgs.append(visualization.plot_simulator_state(state, use_log_traj=True))

mediapy.show_video(imgs, fps=10)



# from waymax import config
# from waymax import dataloader

# scenarios = dataloader.simulator_state_generator(config.WOD_1_1_0_TRAINING)
# scenario = next(scenarios)

# import tensorflow as tf

# print(tf.test.is_gpu_available())