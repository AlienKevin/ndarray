use ndarray::Array;
use ndarray::WgpuDevice;
use ndarray::arr3;
use ndarray::array;
use ndarray::Ix2;
use ndarray::s;

#[test]
fn test_wgpu() {
    env_logger::init();
    let d = futures::executor::block_on(WgpuDevice::new()).unwrap();
    let a_cpu: Array<f32, _> = Array::ones((5, 5)) * 2.;
    let b_cpu: Array<f32, _> = Array::ones((5, 5)) * 3.;
    let c_cpu: Array<f32, _> = Array::ones((5, 5)) * 6.;

    let a_gpu = a_cpu.into_wgpu(&d);
    let b_gpu = b_cpu.into_wgpu(&d);
    let c_gpu = c_cpu.into_wgpu(&d);

    let x_gpu = a_gpu + b_gpu;
    // let y_gpu = x_gpu - c_gpu;
    let y_cpu = x_gpu.into_cpu();
    assert_eq!(y_cpu, Array::<f32, _>::ones((5, 5)) * 5.);

    // let y_cpu = y_gpu.into_cpu();

    // assert_eq!(y_cpu, Array::<f32, _>::ones((5, 5)) * -1.);
}

#[test]
fn test_wgpu_2() {
    env_logger::init();
    let d = futures::executor::block_on(WgpuDevice::new()).unwrap();
    let a_cpu: Array<f32, _> = Array::range(0., 10., 1.0).into_shape((2, 5)).unwrap();

    let a_gpu = a_cpu.clone().into_wgpu(&d);
    let a_t_gpu = a_gpu.reversed_axes();
    let a_t_cpu = a_t_gpu.clone().into_cpu();
    assert_eq!(a_t_cpu, a_cpu.clone().reversed_axes());

    let b_cpu: Array<f32, _> = Array::range(9., -1.0, -1.0).into_shape((2, 5)).unwrap();
    let b_t_cpu = b_cpu.clone().reversed_axes();
    let b_gpu = b_cpu.clone().into_wgpu(&d);
    let b_t_gpu = b_gpu.clone().reversed_axes();
    let c_cpu = Array::from_elem((5, 2), 9.0);

    assert_eq!(&a_t_gpu.dim(), &(5, 2));
    assert_eq!(&a_t_gpu.strides(), &[1, 5]);
    assert_eq!(&a_t_gpu.clone().into_cpu(), a_t_cpu);

    assert_eq!(&b_t_gpu.dim(), &(5, 2));
    assert_eq!(&b_t_gpu.strides(), &[1, 5]);
    assert_eq!(&b_t_gpu.clone().into_cpu(), b_t_cpu);

    // dbg!(&a_cpu.clone().iter().collect::<Vec<_>>());
    // dbg!(&b_cpu.clone().iter().collect::<Vec<_>>());
    assert_eq!(a_t_cpu + b_t_cpu, c_cpu);
    assert_eq!((a_t_gpu + b_t_gpu).into_cpu(), c_cpu);
}

#[test]
fn test_wgpu_3() {
    // this method needs to be inside main() method
    std::env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();
    let dev = futures::executor::block_on(WgpuDevice::new()).unwrap();

    let a: Array<f32, _> = arr3(
                  &[[[ 1.,  2.,  3.],  // -- 2 rows  \_
                  [ 4.,  5.,  6.]],    // --         /
                  [[ 7.,  8.,  9.],     //            \_ 2 submatrices
                  [10., 11., 12.]]]);  //            /
    //  3 columns ..../.../.../
    dbg!(&a.as_ptr());
    let a_gpu = a.clone().into_wgpu(&dev);
    dbg!(&a_gpu.dim());
    dbg!(&a_gpu.strides());
    dbg!(&a_gpu.as_ptr());
    assert_eq!(a_gpu.shape(), &[2, 2, 3]);

    // Let’s create a slice with
    //
    // - Both of the submatrices of the greatest dimension: `..`
    // - Only the first row in each submatrix: `0..1`
    // - Every element in each row: `..`

    let b = a.slice(s![.., 0..1, ..]);
    let b_gpu = b.clone().into_owned().into_wgpu(&dev);
    let c: Array<f32, _> = arr3(&[[[ 1.,  2.,  3.]],
        [[ 7.,  8.,  9.]]]);
    assert_eq!(b_gpu.clone().into_cpu(), c);
    assert_eq!(b_gpu.shape(), &[2, 1, 3]);

    // Let’s create a slice with
    //
    // - Both submatrices of the greatest dimension: `..`
    // - The last row in each submatrix: `-1..`
    // - Row elements in reverse order: `..;-1`
    let d = a.slice(s![.., -1.., ..;-1]);
    dbg!(&d.dim());
    dbg!(&d.strides());
    dbg!(&d.as_ptr());
    let d_2 = a_gpu.slice(s![.., -1.., ..;-1]);
    let d_gpu = d_2.into_wgpu();
    dbg!(&d_gpu.data);
    dbg!(&d_gpu.dim());
    dbg!(&d_gpu.strides());
    dbg!(d_gpu.as_ptr());
    let e: Array<f32, _> = arr3(&[[[ 6.,  5.,  4.]],
        [[12., 11., 10.]]]);
    assert_eq!(d_gpu.clone().into_cpu(), e);
    assert_eq!(d.shape(), &[2, 1, 3]);

    let f: Array<f32, _> = arr3(&[[[ 7., 7., 7.]], [[19., 19., 19.]]]);
    assert_eq!((b_gpu + d_gpu).into_cpu(), f);
}

#[test]
fn test_ind2sub() {
    fn ind2sub(id_: u32, shape: &[i32], indices: &mut Vec<i32>) {
        let mut id: i32 = id_ as i32;
        for i in (0..2).rev() {
            indices[i] = id % shape[i];
            id -= indices[i];
            id /= shape[i];
        }
    }

    let mut indices = vec![0, 0];
    ind2sub(0, &[2, 5], &mut indices);
    assert_eq!(&indices, &[0, 0]);
    ind2sub(4, &[2, 5], &mut indices);
    assert_eq!(&indices, &[0, 4]);
    ind2sub(5, &[2, 5], &mut indices);
    assert_eq!(&indices, &[1, 0]);
    ind2sub(6, &[2, 5], &mut indices);
    assert_eq!(&indices, &[1, 1]);
    ind2sub(9, &[2, 5], &mut indices);
    assert_eq!(&indices, &[1, 4]);
}

#[test]
fn test_add() {
    struct Array<'a> {
        data: &'a mut [f32]
    }

    fn add(id: usize, len: usize, shape: &[i32], lhs_strides: &[i32], lhs: &Array, rhs_strides: &[i32], rhs: &Array, result: &mut Array) {
        let mut id_: i32 = id as i32;
        let mut lhs_id: usize = 0;
        let mut rhs_id: usize = 0;
        for i in (0..len - 1).rev() {
            let idx = id_ % shape[i];
            id_ -= idx;
            id_ /= shape[i];
            // TODO: handle negative strides
            lhs_id += (idx * lhs_strides[i]) as usize;
            rhs_id += (idx * rhs_strides[i]) as usize;
        }
        
        result.data[id] = lhs.data[lhs_id] + rhs.data[rhs_id];
    }

    let len = 2;
    let shape = &[5, 2];
    let mut result = Array { data: &mut [0.0; 10] };
    let lhs_strides = &[1, 5];
    let lhs = Array { data: &mut [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] };
    let rhs_strides = &[1, 5];
    let rhs = Array { data: &mut [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0] };
    for id in 0..10 {
        add(id, len, shape, lhs_strides, &lhs, rhs_strides, &rhs, &mut result);
    }
    assert_eq!(result.data, &[9.0; 10]);
}
