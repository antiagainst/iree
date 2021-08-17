// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-codegen-linalg-to-spirv-pipeline))' %s | IreeFileCheck %s

hal.executable @fused_conv_element_wise attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @s0b0_ro_external, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}> {
    hal.executable.entry_point @fused_conv_element_wise attributes {interface = @io, ordinal = 0 : index}
    builtin.module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader, Float16, Int16, Int8, StorageBuffer16BitAccess, StorageUniform16, StoragePushConstant16, StorageBuffer8BitAccess, UniformAndStorageBuffer8BitAccess, StoragePushConstant8, GroupNonUniform, GroupNonUniformVote, GroupNonUniformArithmetic, GroupNonUniformBallot, GroupNonUniformClustered, GroupNonUniformQuad, VariablePointers, VariablePointersStorageBuffer], [SPV_KHR_16bit_storage, SPV_KHR_8bit_storage, SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>, ARM:IntegratedGPU, {cooperative_matrix_properties_nv = [], max_compute_shared_memory_size = 32768 : i32, max_compute_workgroup_invocations = 512 : i32, max_compute_workgroup_size = dense<512> : vector<3xi32>, subgroup_size = 16 : i32}>}  {
      builtin.func @fused_conv_element_wise() {
        %c0 = constant 0 : index
        %c32 = constant 32 : index
        %c112 = constant 112 : index
        %cst = constant 0.000000e+00 : f32
        %cst_0 = constant 6.000000e+00 : f32
        %cst_1 = constant 0x7FC00000 : f32
        %cst_2 = constant dense<1.0> : tensor<32xf32>
        %cst_3 = constant dense<2.0> : tensor<32xf32>
        %cst_4 = constant dense<3.0> : tensor<32xf32>
        %cst_5 = constant dense<4.0> : tensor<32xf32>
        %0 = hal.interface.binding.subspan @io::@s0b1_ro_external[%c0] : !flow.dispatch.tensor<readonly:1x225x225x3xf32>
        %1 = hal.interface.binding.subspan @io::@s0b0_ro_constant[%c0] : !flow.dispatch.tensor<readonly:3x3x3x32xf32>
        %2 = hal.interface.binding.subspan @io::@s0b2_xw_external[%c0] : !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_size_z = hal.interface.workgroup.size[2] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_z, %workgroup_size_z]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_z, %workgroup_size_z]
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
            %8 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
            scf.for %arg2 = %7 to %c32 step %8 {
              %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %10 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %11 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %12 = tensor.extract_slice %cst_2[%arg2] [%11] [1] : tensor<32xf32> to tensor<?xf32>
              %13 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %14 = tensor.extract_slice %cst_3[%arg2] [%13] [1] : tensor<32xf32> to tensor<?xf32>
              %15 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %16 = tensor.extract_slice %cst_4[%arg2] [%15] [1] : tensor<32xf32> to tensor<?xf32>
              %17 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %18 = tensor.extract_slice %cst_5[%arg2] [%17] [1] : tensor<32xf32> to tensor<?xf32>
              %19 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg0)[%workgroup_size_z]
              %20 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 112)>(%arg1)[%workgroup_size_y]
              %21 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 32)>(%arg2)[%workgroup_size_x]
              %22 = linalg.init_tensor [1, %19, %20, %21] : tensor<1x?x?x?xf32>
              %23 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %24 = affine.min affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>(%9, %arg0)
              %25 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %26 = affine.min affine_map<(d0, d1) -> (d0 * 2 + 1, d1 * -2 + 225)>(%10, %arg1)
              %27 = flow.dispatch.tensor.load %0, offsets = [0, %23, %25, 0], sizes = [1, %24, %26, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:1x225x225x3xf32> -> tensor<1x?x?x3xf32>
              %28 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %29 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 3, %28], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:3x3x3x32xf32> -> tensor<3x3x3x?xf32>
              %30 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg0)[%workgroup_size_z]
              %31 = affine.min affine_map<(d0)[s0] -> (-d0 + 112, s0)>(%arg1)[%workgroup_size_y]
              %32 = affine.min affine_map<(d0)[s0] -> (-d0 + 32, s0)>(%arg2)[%workgroup_size_x]
              %33 = linalg.init_tensor [1, %30, %31, %32] : tensor<1x?x?x?xf32>
              %34 = linalg.fill(%cst, %33) : f32, tensor<1x?x?x?xf32> -> tensor<1x?x?x?xf32>
              %35 = linalg.conv_2d_input_nhwc_filter_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%27, %29 : tensor<1x?x?x3xf32>, tensor<3x3x3x?xf32>) outs(%34 : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
              %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35, %12, %14, %16, %18 : tensor<1x?x?x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%22 : tensor<1x?x?x?xf32>) attrs =  {__internal_linalg_transform__ = "workgroup"} {
              ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
                %37 = subf %arg3, %arg4 : f32
                %38 = mulf %37, %arg5 : f32
                %39 = divf %38, %arg6 : f32
                %40 = addf %39, %arg7 : f32
                %41 = cmpf olt, %40, %cst_0 : f32
                %42 = select %41, %40, %cst_0 : f32
                %43 = cmpf uno, %40, %cst_0 : f32
                %44 = select %43, %cst_1, %42 : f32
                %45 = cmpf ogt, %44, %cst : f32
                %46 = select %45, %44, %cst : f32
                %47 = cmpf uno, %44, %cst : f32
                %48 = select %47, %cst_1, %46 : f32
                linalg.yield %48 : f32
              } -> tensor<1x?x?x?xf32>
              flow.dispatch.tensor.store %36, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, %19, %20, %21], strides = [1, 1, 1, 1] : tensor<1x?x?x?xf32> -> !flow.dispatch.tensor<writeonly:1x112x112x32xf32>
            }
          }
        }
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @s0b0_ro_constant, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b1_ro_external, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @s0b2_xw_external, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

//   CHECK-LABEL: spv.func @fused_conv_element_wise()
// CHECK-COUNT-4: spv.Variable
//         CHECK: spv.mlir.loop
// CHECK-COUNT-4:   spv.Variable
//         CHECK:   spv.mlir.loop
// CHECK-COUNT-3:     spv.Load
// CHECK-COUNT-3:     spv.GLSL.Fma
// CHECK-COUNT-3:     spv.Load
// CHECK-COUNT-3:     spv.GLSL.Fma
// CHECK-COUNT-3:     spv.Load
// CHECK-COUNT-3:     spv.GLSL.Fma
// CHECK-COUNT-4:     spv.Store
// CHECK-COUNT-4:   spv.Load
// CHECK-COUNT-4:   spv.Store
// CHECK-COUNT-4: spv.Load
// CHECK-COUNT-4: spv.Store
