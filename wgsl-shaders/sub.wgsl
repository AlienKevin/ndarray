struct Array {
    data: array<f32>
};

@group(0)
@binding(0)
var<storage, read_write> in_indices1: Array;

@group(0)
@binding(1)
var<storage, read_write> in_indices2: Array;

@group(0)
@binding(2)
var<storage, read_write> out_indices: Array;

@compute
@workgroup_size(16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    out_indices.data[global_id.x] = in_indices1.data[global_id.x] - in_indices2.data[global_id.x];
}
