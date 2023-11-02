
import os
import random
import functools
from typing import Callable, Iterator, Optional, Sequence, TypeVar




import jax
import tensorflow as tf


from waymax import config as _config
from waymax.dataloader.dataloader_utils import generate_sharded_filenames


T = TypeVar('T')
AUTOTUNE = tf.data.AUTOTUNE

# data = tf.data.TFRecordDataset("./training.tfrecord-00100-of-01000")


# print("hello, world")


def tf_dataset_from_path(
    path: str,
    data_format: _config.DataFormat,
    preprocess_fn: Callable[[bytes], dict[str, tf.Tensor]],
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 100,
    repeat: Optional[int] = None,
    batch_dims: Sequence[int] = (),
    num_shards: int = 1,
    deterministic: bool = True,
    drop_remainder: bool = True,
    tf_data_service_address: Optional[str] = None,
    batch_by_scenario: bool = True,
)->tf.data.Dataset:
    if data_format == _config.DataFormat.TFRECORD:
        dataset_fn = tf.data.TFRecordDataset
    else:
        raise ValueError('Data format %s is not supported.' % data_format)

    files_to_load = [path]
    if '@' in os.path.basename(path):
        files_to_load = generate_sharded_filenames(path)
    if shuffle_seed:
        random.seed(shuffle_seed)
        random.shuffle(files_to_load)
    files = tf.data.Dataset.from_tensor_slices(files_to_load)
    # Split files across multiple processes for distributed training/eval.
    files = files.shard(jax.process_count(), jax.process_index())

    def _make_dataset(
        shard_index: int, num_shards: int, local_files: tf.data.Dataset
    ):
        # Outer parallelism.
        local_files = local_files.shard(num_shards, shard_index)
        ds = local_files.interleave(
            dataset_fn,
            num_parallel_calls=AUTOTUNE,
            cycle_length=AUTOTUNE,
            deterministic=deterministic,
        )

        ds = ds.repeat(repeat)
        if shuffle_seed is not None:
        # Makes sure each host uses a different RNG for shuffling.
            local_seed = jax.random.fold_in(
                jax.random.PRNGKey(shuffle_seed), jax.process_index()
            )[0]
            ds = ds.shuffle(shuffle_buffer_size, seed=local_seed)

        ds = ds.map(
            preprocess_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic
        )
        if not batch_by_scenario:
            ds = ds.unbatch()
        if batch_dims:
            for batch_size in reversed(batch_dims):
                ds = ds.batch(
                    batch_size,
                    drop_remainder=drop_remainder,
                    num_parallel_calls=AUTOTUNE,
                    deterministic=deterministic,
                )
        return ds

    make_dataset_fn = functools.partial(
        _make_dataset, num_shards=num_shards, local_files=files
    )
    indices = tf.data.Dataset.range(num_shards)
    dataset = indices.interleave(
        make_dataset_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic
    )

    if tf_data_service_address is not None:
        dataset = dataset.apply(
            tf.data.experimental.service.distribute(
                processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
                service=tf_data_service_address,
            )
        )
    return dataset.prefetch(AUTOTUNE)

path = "./training.tfrecord@1"

files = generate_sharded_filenames(path)
print(files)