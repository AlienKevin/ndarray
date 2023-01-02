use ndarray::Array;
use ndarray::WgpuDevice;
use ndarray::array;
use ndarray::Ix2;

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
