struct Array {
    data: array<f32>
};

@group(0) @binding(0) var<storage, read> shape: array<i32>;
@group(0) @binding(1) var<storage, read> lhs_strides: array<i32>;
@group(0) @binding(2) var<storage, read> lhs: Array;
@group(0) @binding(3) var<storage, read> rhs_strides: array<i32>;
@group(0) @binding(4) var<storage, read> rhs: Array;
@group(0) @binding(5) var<storage, read_write> result: Array;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    
    var id_: i32 = i32(id);
    var lhs_id: i32 = 0;
    var rhs_id: i32 = 0;
    for (var i: i32 = $len - 1; i >= 0; i--) {
        let idx = id_ % shape[i];
        id_ -= idx;
        id_ /= shape[i];
        lhs_id += idx * lhs_strides[i];
        rhs_id += idx * rhs_strides[i];
    }
    
    result.data[id] = lhs.data[$lhs_offset + lhs_id] + rhs.data[$rhs_offset + rhs_id];
}
