struct Array {
    data: array<f32>
};

@group(0) @binding(0) var<storage, read> shape: array<u32>;
@group(0) @binding(1) var<storage, read> lhs_strides: array<i32>;
@group(0) @binding(2) var<storage, read> lhs: Array;
@group(0) @binding(3) var<storage, read> scalar: f32;
@group(0) @binding(4) var<storage, read_write> result: Array;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    
    var id_: u32 = id;
    var lhs_id: i32 = 0;
    for (var i: i32 = $ndim; i >= 0; i--) {
        // See StackOverflow question for context: https://stackoverflow.com/q/46782444/6798201
        // For code see: https://github.com/stdlib-js/ndarray-base-ind2sub/blob/c759c6f6d53bf6ff63c8781fad57aa3def83c666/src/main.c#L107
        let s = shape[i];
        let idx = id_ % s;
        id_ -= idx;
        id_ /= s;
        lhs_id += i32(idx) * lhs_strides[i];
    }

    result.data[id] = lhs.data[$lhs_offset + lhs_id] $op scalar;
}
