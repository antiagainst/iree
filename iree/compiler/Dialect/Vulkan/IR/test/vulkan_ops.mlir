// RUN: iree-opt -split-input-file %s IreeFileCheck %s

func @invoke_glsl_kernel(%arg0: tensor<8xf32>) -> (tensor<8xf32>) {
  %0 = vk.invoke_glsl_kernel inputs=(%arg0: tensor<8xf32>), outputs=(tensor<8xf32>),
    source="""
      #version 450

      layout(set = 0, binding = 0) buffer InputBuffer { float input_values[8]; };
      layout(set = 0, binding = 1) buffer OutputBuffer { float output_values[8]; };

      layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

      void main() {
        uint index = gl_GlobalInvocationID.x;
        if (index < 8) {
          output_values[index] = input_values[index] * 2;
        }
      }
    """
  return %0: tensor<8xf32>
}
