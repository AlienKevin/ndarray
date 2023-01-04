use ndarray::Array;
use ndarray::WgpuDevice;
use ndarray::arr3;
use ndarray::s;
use approx::assert_abs_diff_eq;

#[cfg(test)]
#[ctor::ctor]
fn init() {
    std::env::set_var("RUST_BACKTRACE", "1");
    env_logger::init();
}

#[test]
fn test_wgpu_add_1() {
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
fn test_wgpu_add_2() {
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
fn test_wgpu_add_3() {
    let dev = futures::executor::block_on(WgpuDevice::new()).unwrap();

    let a: Array<f32, _> = arr3(
                  &[[[ 1.,  2.,  3.],  // -- 2 rows  \_
                  [ 4.,  5.,  6.]],    // --         /
                  [[ 7.,  8.,  9.],     //            \_ 2 submatrices
                  [10., 11., 12.]]]);  //            /
    //  3 columns ..../.../.../
    // dbg!(&a.as_ptr());
    let a_gpu = a.clone().into_wgpu(&dev);
    // dbg!(&a_gpu.dim());
    // dbg!(&a_gpu.strides());
    // dbg!(&a_gpu.as_ptr());
    assert_eq!(a_gpu.shape(), &[2, 2, 3]);

    // Let’s create a slice with
    //
    // - Both of the submatrices of the greatest dimension: `..`
    // - Only the first row in each submatrix: `0..1`
    // - Every element in each row: `..`

    let b = a.slice(s![.., 0..1, ..]);
    // let b_gpu = b.clone().into_owned().into_wgpu(&dev);
    // let c: Array<f32, _> = arr3(&[[[ 1.,  2.,  3.]],
    //     [[ 7.,  8.,  9.]]]);
    // assert_eq!(b_gpu.clone().into_cpu(), c);
    // assert_eq!(b_gpu.shape(), &[2, 1, 3]);
    let b_gpu = a_gpu.slice(s![.., 0..1, ..]).into_wgpu(&dev);

    // Let’s create a slice with
    //
    // - Both submatrices of the greatest dimension: `..`
    // - The last row in each submatrix: `-1..`
    // - Row elements in reverse order: `..;-1`
    let d = a.slice(s![.., -1.., ..;-1]);
    // dbg!(&d.dim());
    // dbg!(&d.strides());
    // dbg!(&d.as_ptr());
    let d_gpu = a_gpu.slice(s![.., -1.., ..;-1]).into_wgpu(&dev);
    // dbg!(&d_gpu.dim());
    // dbg!(&d_gpu.strides());
    // dbg!(d_gpu.as_ptr());
    let e: Array<f32, _> = arr3(&[[[ 6.,  5.,  4.]],
        [[12., 11., 10.]]]);
    // dbg!(&d);
    // assert_eq!(d.clone(), e);
    // assert_eq!(d_gpu.clone().into_cpu(), e);
    // assert_eq!(d.shape(), &[2, 1, 3]);

    // dbg!(b_gpu.get_data());
    // dbg!(d_gpu.get_data());
    // dbg!(b_gpu.clone().into_cpu());
    // dbg!(d_gpu.clone().into_cpu());
    let f: Array<f32, _> = arr3(&[[[ 7., 7., 7.]], [[19., 19., 19.]]]);
    assert_eq!(&b + &d, f);
    assert_eq!((b_gpu + d_gpu).into_cpu(), f);
}

#[test]
#[cfg(feature = "approx")]
fn test_wgpu_binary_elementwise() {
    let dev = futures::executor::block_on(WgpuDevice::new()).unwrap();

    let a: Array<f32, _> = arr3(
                  &[[[ 1.,  2.,  3.],  // -- 2 rows  \_
                  [ 4.,  5.,  6.]],    // --         /
                  [[ 7.,  8.,  9.],     //            \_ 2 submatrices
                  [10., 11., 12.]]]);  //            /
    let a_gpu = a.clone().into_wgpu(&dev);
    let b = a.slice(s![.., 0..1, ..]);
    let b_gpu = a_gpu.slice(s![.., 0..1, ..]).into_wgpu(&dev);
    let c: Array<f32, _> = arr3(&[[[ 1.,  2.,  3.]],
            [[ 7.,  8.,  9.]]]);
    let d = a.slice(s![.., -1.., ..;-1]);
    let d_gpu = a_gpu.slice(s![.., -1.., ..;-1]).into_wgpu(&dev);
    let e: Array<f32, _> = arr3(&[[[ 6.,  5.,  4.]],
        [[12., 11., 10.]]]);

    // add
    assert_eq!(&b + &d, &c + &e);
    assert_eq!((b_gpu.clone() + d_gpu.clone()).into_cpu(), &c + &e);
    // subtract
    assert_eq!(&b - &d, &c - &e);
    assert_eq!((b_gpu.clone() - d_gpu.clone()).into_cpu(), &c - &e);
    // multiply
    assert_eq!(&b * &d, &c * &e);
    assert_eq!((b_gpu.clone() * d_gpu.clone()).into_cpu(), &c * &e);
    // divide
    assert_eq!(&b / &d, &c / &e);
    assert_abs_diff_eq!((b_gpu.clone() / d_gpu.clone()).into_cpu(), &c / &e);
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
    struct Array_<'a> {
        data: &'a [f32]
    }

    struct ArrayMut_<'a> {
        data: &'a mut [f32]
    }

    fn add(id: usize, len: usize, shape: &[usize], lhs_offset: usize, lhs_strides: &[isize], lhs: &Array_, rhs_offset: usize, rhs_strides: &[isize], rhs: &Array_, result: &mut ArrayMut_) {
        let mut id_: isize = id as isize;
        let mut lhs_id: isize = 0;
        let mut rhs_id: isize = 0;
        for i in (0..len).rev() {
            let s = shape[i] as isize;
            let idx = id_ % s;
            // println!("{} % {} = {}", id_, s, id_ % s);
            id_ -= idx;
            id_ /= s;
            // TODO: handle negative strides
            lhs_id += idx * lhs_strides[i];
            rhs_id += idx * rhs_strides[i];
        }
        
        result.data[id] = lhs.data[(lhs_offset as isize + lhs_id) as usize] + rhs.data[(rhs_offset as isize + rhs_id) as usize];
    }

    {
        let len = 2;
        let shape = &[5, 2];
        let mut result = ArrayMut_ { data: &mut [0.0; 10] };
        let lhs_offset = 0;
        let lhs_strides = &[1, 5];
        let lhs = Array_ { data: &mut [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] };
        let rhs_offset = 0;
        let rhs_strides = &[1, 5];
        let rhs = Array_ { data: &mut [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0] };
        for id in 0..10 {
            add(id, len, shape, lhs_offset, lhs_strides, &lhs, rhs_offset, rhs_strides, &rhs, &mut result);
        }
        assert_eq!(result.data, &[9.0; 10]);
    }

    {
        let len = 2;
        let shape = &[5, 2];
        let mut result = ArrayMut_ { data: &mut [0.0; 10] };
        let lhs_offset = 9;
        let lhs_strides = &[-1, -5];
        let lhs = Array_ { data: &mut [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] };
        let rhs_offset = 9;
        let rhs_strides = &[-1, -5];
        let rhs = Array_ { data: &mut [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0] };
        for id in 0..10 {
            add(id, len, shape, lhs_offset, lhs_strides, &lhs, rhs_offset, rhs_strides, &rhs, &mut result);
        }
        assert_eq!(result.data, &[9.0; 10]);
    }

    {
        let a: Array<f32, _> = arr3(
            &[[[ 1.,  2.,  3.],  // -- 2 rows  \_
            [ 4.,  5.,  6.]],    // --         /
            [[ 7.,  8.,  9.],     //            \_ 2 submatrices
            [10., 11., 12.]]]);  //            /
        let lhs = a.slice(s![.., 0..1, ..]);
        let lhs_array = Array_ { data: a.as_slice().unwrap() };
        let lhs_strides = lhs.strides();
        let lhs_offset = 0;
        dbg!(lhs_strides);
        
        let rhs = a.slice(s![.., -1.., ..;-1]);
        let rhs_array = Array_ { data: a.as_slice().unwrap() };
        let rhs_strides = rhs.strides();
        let rhs_offset = 5;
        dbg!(rhs_strides);

        let mut result = ArrayMut_ { data: &mut [0.0; 6] };
        let len = lhs.ndim();
        let shape = lhs.shape();
        dbg!(&shape);

        for id in 0..6 {
            add(id, len, shape, lhs_offset, lhs_strides, &lhs_array, rhs_offset, rhs_strides, &rhs_array, &mut result);
        }
        assert_eq!(result.data, &[7.0, 7.0, 7.0, 19.0, 19.0, 19.0]);
    }

    {
        let dev = futures::executor::block_on(WgpuDevice::new()).unwrap();

        let a: Array<f32, _> = arr3(
            &[[[ 1.,  2.,  3.],  // -- 2 rows  \_
            [ 4.,  5.,  6.]],    // --         /
            [[ 7.,  8.,  9.],     //            \_ 2 submatrices
            [10., 11., 12.]]]);  //            /
        let a_gpu = a.clone().into_wgpu(&dev);
        
        let lhs = a_gpu.slice(s![.., 0..1, ..]).into_wgpu(&dev);
        let lhs_array = Array_ { data: a.as_slice().unwrap() };
        let lhs_strides = lhs.strides();
        let lhs_offset = 0;
        dbg!(lhs_strides);
        
        let rhs = a_gpu.slice(s![.., -1.., ..;-1]).into_wgpu(&dev);
        let rhs_array = Array_ { data: a.as_slice().unwrap() };
        let rhs_strides = rhs.strides();
        let rhs_offset = 5;
        dbg!(rhs_strides);

        let mut result = ArrayMut_ { data: &mut [0.0; 6] };
        let len = lhs.ndim();
        let shape = lhs.shape();
        dbg!(&shape);

        for id in 0..6 {
            add(id, len, shape, lhs_offset, lhs_strides, &lhs_array, rhs_offset, rhs_strides, &rhs_array, &mut result);
        }
        assert_eq!(result.data, &[7.0, 7.0, 7.0, 19.0, 19.0, 19.0]);
    }
}
