use crate::Dimension;
use crate::WgpuArray;
use crate::WgpuRepr;
use crate::WgpuDevice;
use crate::DimMax;

use std::borrow::Cow;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::convert::TryFrom;

macro_rules! impl_binary_op(
    ($trt:ident, $operator:tt, $mth:ident, $iop:tt, $shader:expr, $doc:expr) => (

impl<'d, A, D> $trt<WgpuArray<'d, A, D>> for WgpuArray<'d, A, D>
where
    A: bytemuck::Pod + std::fmt::Debug + Default,
    D: Dimension,
{
    type Output = WgpuArray<'d, A, D>;
    fn $mth(self, rhs: WgpuArray<'d, A, D>) -> Self::Output
    {
        self.$mth(&rhs)
    }
}

impl<'a, 'd, A, D, E> $trt<&WgpuArray<'_, A, E>> for WgpuArray<'d, A, D>
where
    A: bytemuck::Pod + std::fmt::Debug + Default,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    type Output = WgpuArray<'d, A, <D as DimMax<E>>::Output>;
    fn $mth(self, rhs: &WgpuArray<A,E>) -> Self::Output
    {
        let (lhs_view, rhs_view) = self.broadcast_with(&rhs).unwrap();
        let lhs_offset = WgpuDevice::ptr_to_offset(lhs_view.ptr);
        let rhs_offset = WgpuDevice::ptr_to_offset(rhs_view.ptr);
        dbg!(&(lhs_view.dim.ndim() - 1).to_string());
        dbg!(lhs_offset.to_string());
        dbg!(rhs_offset.to_string());
        dbg!(lhs_view.dim.ndim());

        let cs_module =
            self.data
                .wgpu_device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&include_str!($shader)
                        .replace("$ndim", &(lhs_view.dim.ndim() - 1).to_string())
                        .replace("$lhs_offset", &lhs_offset.to_string())
                        .replace("$rhs_offset", &rhs_offset.to_string())))
                });
        
        dbg!(lhs_view.dim());
        dbg!(lhs_view.strides());
        dbg!(lhs_view.len());

        dbg!(rhs_view.dim());
        dbg!(rhs_view.strides());
        dbg!(rhs_view.len());

        let dim = self.data.wgpu_device.create_storage_buffer(&lhs_view.dim.slice().iter().map(|s| u32::try_from(*s).unwrap()).collect::<Vec<u32>>()[..]);
        let lhs_strides_buffer = self.data.wgpu_device.create_storage_buffer(&lhs_view.strides().iter().map(|s| i32::try_from(*s).unwrap()).collect::<Vec<i32>>()[..]);
        let rhs_strides_buffer = self.data.wgpu_device.create_storage_buffer(&rhs_view.strides().iter().map(|s| i32::try_from(*s).unwrap()).collect::<Vec<i32>>()[..]);
        let (result_buffer, result_buffer_ptr) = self.data.wgpu_device.allocate_storage_buffer(vec![A::default(); lhs_view.len()].as_slice());
        let compute_pipeline =
            self
                .data
                .wgpu_device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: &cs_module,
                    entry_point: "main",
                });
        
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self
            .data
            .wgpu_device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: dim.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lhs_strides_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.data.storage_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: rhs_strides_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: rhs.data.storage_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: result_buffer.as_entire_binding(),
                    },
                ],
            });
        
        let mut encoder = self
            .data
            .wgpu_device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker($doc);
            cpass.dispatch_workgroups(lhs_view.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }

        // Submits command encoder for processing
        self.data.wgpu_device.queue.submit(Some(encoder.finish()));
        let data: WgpuRepr<A> = WgpuRepr {
            wgpu_device: self.data.wgpu_device,
            storage_buffer: result_buffer,
            len: lhs_view.len(),
            life: PhantomData
        };
        let strides = lhs_view.dim.default_strides();
        let array = WgpuArray {
            data,
            ptr: result_buffer_ptr,
            dim: lhs_view.dim,
            strides,
        };
        array
    }
}););

use std::ops::*;
impl_binary_op!(Add, +, add, +=, "../wgsl-shaders/add.wgsl", "addition");
// impl_binary_op!(Sub, -, sub, -=, "../wgsl-shaders/sub.wgsl", "subtraction");
// impl_binary_op!(Mul, *, mul, *=, "../wgsl-shaders/mul.wgsl", "multiplication");
// impl_binary_op!(Div, /, div, /=, "../wgsl-shaders/div.wgsl", "division");
