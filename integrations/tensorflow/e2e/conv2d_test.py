# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
import numpy as np
from pyiree.tf.support import tf_test_utils
from pyiree.tf.support import tf_utils
import tensorflow.compat.v2 as tf


class Conv2DModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([1, 3, 3, 4], tf.float32),
      tf.TensorSpec([2, 2, 4, 32], tf.float32),
  ])
  def conv2d_stride1(self, img, kernel):
    return tf.nn.conv2d(img, kernel, strides=[1, 1], padding="VALID")


class Conv2DTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(Conv2DModule)

  def test_conv2d_stride1(self):

    def conv2d_stride1(module):
      image = np.random.randint(0, 17, size=(1, 3, 3, 4)) * 0.5
      kernel = np.random.randint(0, 11, size=(2, 2, 4, 32)) * 0.5
      module.conv2d_stride1(image.astype(np.float32), kernel.astype(np.float32))

    self.compare_backends(conv2d_stride1, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
