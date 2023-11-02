from typing import Sequence
import math
import functools


def zzq_generate_sharded_filenames(path: str, suffix: int = 0, file_slices: int = 0) -> Sequence[str]:
  base_name, num_shards = path.split('@')
  num_shards = int(file_slices if file_slices else num_shards)
  shard_width = max(5, int(math.log10(num_shards) + 1))
  format_str = base_name + '-%0' + str(shard_width) + 'd-of-%05d'
  return [format_str % (i, (suffix if suffix else num_shards)) for i in range(num_shards)]




def generate_sharded_filenames(path: str) -> Sequence[str]:
  """Returns the filenames of individual sharded files.

  A sharded file is a set of files of the format filename-XXXXX-of-YYYYY,
  where XXXXX is a placeholder for the index of the shard, and YYYYY is the
  total number of shards. These files are collectively referred to by a
  sharded path filename@YYYYY.

  For example, the sharded path `myfile@100` refers to the set of files
    - myfile-00000-of-00100
    - myfile-00001-of-00100
    - ...
    - myfile-00098-of-00100
    - myfile-00099-of-00100

  Args:
    path: A path to a sharded file, with format `filename@shards`, where shards
      is an integer denoting the number of total shards.

  Returns:
    An iterator through the complete set of filenames that the path refers to,
    with each filename having the format `filename-XXXXX-of-YYYYY`
  """
  base_name, num_shards = path.split('@')
  num_shards = int(num_shards)
  shard_width = max(5, int(math.log10(num_shards) + 1))
  format_str = base_name + '-%0' + str(shard_width) + 'd-of-%05d'
  return [format_str % (i, num_shards) for i in range(num_shards)]


def replace_method(new_method):
    def decorator(original_function):
        def wrapper(*args, **kwargs):
            # 在这里执行替代逻辑，替换原方法
            result = new_method(*args, **kwargs)
            return result

        return wrapper

    return decorator


@replace_method(functools.partial(zzq_generate_sharded_filenames, suffix=150, file_slices=10))
def generate_sharded_filenames(path: str) -> Sequence[str]:

  base_name, num_shards = path.split('@')
  num_shards = int(num_shards)
  shard_width = max(5, int(math.log10(num_shards) + 1))
  format_str = base_name + '-%0' + str(shard_width) + 'd-of-%05d'
  return [format_str % (i, num_shards) for i in range(num_shards)]


def main():
   path="/home/yqnj/testing/testing.tfrecord@150"
   data = generate_sharded_filenames(path)
   print(data)

main()