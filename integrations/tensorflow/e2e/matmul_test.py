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
"""Tests for ops in the tf.math module."""

from absl import app
import numpy as np
from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf


class MatmulModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([1024, 1024], tf.float32),
      tf.TensorSpec([1024, 1024], tf.float32)
  ])
  def matmul1Kx1Kx1K(self, x, y):
    return tf.linalg.matmul(x, y)


class MatmulTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(MatmulModule)

  def test_matmul1Kx1Kx1K(self):

    def matmul1Kx1Kx1K(module):
      a = np.random.randint(0, 17, size=(1024, 1024)) * 0.5
      b = np.random.randint(0, 11, size=(1024, 1024)) * 0.5
      module.matmul1Kx1Kx1K(a.astype(np.float32), b.astype(np.float32))

    self.compare_backends(matmul1Kx1Kx1K, self._modules)


def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
