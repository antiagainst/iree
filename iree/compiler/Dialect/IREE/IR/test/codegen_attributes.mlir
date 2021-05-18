// Test parsing and printing CodeGen attributes.

// RUN: iree-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | IreeFileCheck %s

"some_op"() {
// CHECK:  #iree.codegen.action.distribute<along_dimensions = [2, 1, 0], with_workgroup_size = [32, 16, 1]>
  action = #iree.codegen.action.distribute<along_dimensions = [2, 1, 0], with_workgroup_size = [32, 16, 1]>
} : () -> ()

// -----

"some_op"() {
// CHECK:  #iree.codegen.action.tile<along_dimensions = [0, 1, 2, 3], with_tile_sizes = [8, 16, 32, 4]>
  action = #iree.codegen.action.tile<along_dimensions = [0, 1, 2, 3], with_tile_sizes = [8, 16, 32, 4]>
} : () -> ()

// -----

"some_op"() {
// CHECK:  #iree.codegen.action.tile_and_distribute<along_dimensions = [3, 2, 1], with_tile_sizes = [8, 16, 32], Workgroup>
  action = #iree.codegen.action.tile_and_distribute<along_dimensions = [3, 2, 1], with_tile_sizes = [8, 16, 32], Workgroup>
} : () -> ()

// -----

"some_op"() {
// CHECK:  #iree.codegen.action.vectorize<with_vector_size = 4>
  action = #iree.codegen.action.vectorize<with_vector_size = 4>
} : () -> ()

// -----

"some_op"() {
// CHECK:  #iree.codegen.type.filter<TileableBy:tensor<1x?x?x8xf32>>
  filter = #iree.codegen.type.filter<TileableBy:tensor<1x?x?x8xf32>>
} : () -> ()

// -----

"some_op"() {
  // expected-error @+1 {{invalid TypeMatchCriterion attribute value: UnknownAction}}
  filter = #iree.codegen.type.filter<UnknownAction:tensor<1x?x?x8xf32>>
} : () -> ()

// -----

"some_op"() {
// CHECK: #iree.codegen.op.filter<on_output_types = [#iree.codegen.type.filter<{{.+}}>, #iree.codegen.type.filter<{{.+}}>]>
  filter = #iree.codegen.op.filter<
    on_output_types=[
      #iree.codegen.type.filter<TileableBy:tensor<1x?x?x16xf32>>,
      #iree.codegen.type.filter<TileableBy:tensor<1x?x?x8xf32>>
    ]
  >
} : () -> ()

// -----

"some_op"() {
// CHECK: #iree.codegen.op.choice<#iree.codegen.op.filter<{{.+}}> -> [#iree.codegen.action.distribute<{{.+}}>]>
  choice = #iree.codegen.op.choice<
    #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<8x32x64xf32>>]> ->
    [
      #iree.codegen.action.distribute<along_dimensions = [2, 1, 0], with_workgroup_size = [32, 16, 1]>
    ]
  >
} : () -> ()

// -----

"some_op"() {
// CHECK: #iree.codegen.op.policy<"linalg.conv_2d", FirstMatch, [#iree.codegen.op.choice<{{.*}}>, #iree.codegen.op.choice<{{.+}}>]>
  policy = #iree.codegen.op.policy<"linalg.conv_2d", FirstMatch, [
    #iree.codegen.op.choice<
      #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<1x4x4x16xf32>>]> ->
      [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_sizes = [4, 4, 16]>]
    >,
    #iree.codegen.op.choice<
      #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<1x2x2x64xf32>>]> ->
      [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_sizes = [2, 2, 64]>]
    >
  ]>
} : () -> ()

// -----

"some_op"() {
  // expected-error @+1 {{invalid PolicyMatchCriterion attribute value: UnknownMatch}}
  policy = #iree.codegen.op.policy<"linalg.conv_2d", UnknownMatch, []>
} : () -> ()

// -----

"some_op"() {
// CHECK: #iree.codegen.target.choice<{gpu = "mali-g77", os = "android-11"} -> [#iree.codegen.op.policy<{{.+}}>, #iree.codegen.op.policy<{{.+}}>]>
  choice = #iree.codegen.target.choice<
    {"gpu" = "mali-g77", "os" = "android-11"} ->
    [
      #iree.codegen.op.policy<"linalg.conv_2d", FirstMatch, [
        #iree.codegen.op.choice<
          #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<1x2x2x64xf32>>]> ->
          [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_sizes = [2, 2, 64]>]
        >
      ]>,
      #iree.codegen.op.policy<"linalg.matmul", FirstMatch, [
        #iree.codegen.op.choice<
          #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<16x64xf32>>]> ->
          [#iree.codegen.action.tile<along_dimensions = [0, 1], with_tile_sizes = [16, 64]>]
        >
      ]>
    ]
  >
} : () -> ()

// -----

"some_op"() {
// CHECK: #iree.codegen.target.policy<"GPU", FirstMatch, [#iree.codegen.target.choice<{{.+}}>]>
  choice = #iree.codegen.target.policy<"GPU", FirstMatch, [
    #iree.codegen.target.choice<
      {"gpu" = "mali-g77", "os" = "android-11"} ->
      [
        #iree.codegen.op.policy<"linalg.conv_2d", FirstMatch, [
          #iree.codegen.op.choice<
            #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<1x2x2x64xf32>>]> ->
            [#iree.codegen.action.tile<along_dimensions = [1, 2, 3], with_tile_sizes = [2, 2, 64]>]
          >
        ]>,
        #iree.codegen.op.policy<"linalg.matmul", FirstMatch, [
          #iree.codegen.op.choice<
            #iree.codegen.op.filter<on_output_types=[#iree.codegen.type.filter<TileableBy:tensor<16x64xf32>>]> ->
            [#iree.codegen.action.tile<along_dimensions = [0, 1], with_tile_sizes = [16, 64]>]
          >
        ]>
      ]
    >
  ]>
} : () -> ()
