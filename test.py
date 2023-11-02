import tensorflow as tf
# import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print("Num GPUs Available: ", len(tf.config.list_physical_devices()))
print(tf.config.list_physical_devices())
# →
# Num GPUs Available: 1
print(tf.sysconfig.get_build_info())

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
