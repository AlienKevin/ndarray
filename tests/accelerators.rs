use ndarray::Array;
use ndarray::WgpuDevice;
use ndarray::array;
use ndarray::Ix2;

#[test]
fn test_wgpu() {
    let d = futures::executor::block_on(WgpuDevice::new()).unwrap();
    let a_cpu: Array<f32, _> = Array::ones((5, 5)) * 2.;
    let b_cpu: Array<f32, _> = Array::ones((5, 5)) * 3.;
    let c_cpu: Array<f32, _> = Array::ones((5, 5)) * 6.;

    let a_gpu = a_cpu.into_wgpu(&d);
    let b_gpu = b_cpu.into_wgpu(&d);
    let c_gpu = c_cpu.into_wgpu(&d);

    let x_gpu = a_gpu + b_gpu;
    let y_gpu = x_gpu - c_gpu;

    let y_cpu = y_gpu.into_cpu();

    assert_eq!(y_cpu, Array::<f32, _>::ones((5, 5)) * -1.);
}

#[test]
fn test_wgpu_2() {
    let d = futures::executor::block_on(WgpuDevice::new()).unwrap();
    let a_cpu = Array::range(0., 10., 1.0).into_shape((2, 5)).unwrap();

    let a_gpu = a_cpu.clone().into_wgpu(&d);    
    let a_t_gpu = a_gpu.reversed_axes();
    let a_t_cpu = a_t_gpu.into_cpu();
    assert_eq!(a_t_cpu, a_cpu.reversed_axes());
}
