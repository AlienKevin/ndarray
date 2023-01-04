use crate::Array;
use crate::Dimension;
use crate::StrideShape;
use crate::WgpuArray;
use crate::WgpuDevice;
use crate::shape_builder;
use rawpointer::PointerExt;
use std::convert::TryInto;

impl<'d, A, D> WgpuArray<'d, A, D>
where
    A: bytemuck::Pod + std::fmt::Debug,
    D: Dimension,
{
    pub fn get_data(&self) -> Vec<A> {
        // Get number of bytes
        let slice_size = self.data.len * std::mem::size_of::<A>();
        let size = slice_size as u64;

        // Create a CPU buffer to store result
        let staging_buffer = self.data.wgpu_device.create_staging_buffer(size);

        let mut encoder = self
            .data
            .wgpu_device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.data.storage_buffer, 0, &staging_buffer, 0, size);
        self.data.wgpu_device.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        
        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.data.wgpu_device.device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Some(Ok(())) = futures::executor::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<A> = bytemuck::cast_slice(&data).to_vec();            
            drop(data);
            staging_buffer.unmap();
            return result;
        } else {
            panic!("Failed to map GPU buffer to CPU");
        }
    }

    pub fn into_cpu(self) -> Array<A, D> {
        let mut result = self.get_data();
        let offset = WgpuDevice::ptr_to_offset(self.ptr).try_into().unwrap();
        let ptr = unsafe { crate::extension::nonnull::nonnull_from_vec_data(&mut result).add(offset) };
        let array = unsafe { crate::ArrayBase::from_data_ptr(crate::DataOwned::new(result), ptr).with_strides_dim(self.strides, self.dim) };
        // let mut array = unsafe {
        //     Array::<A, D>::from_shape_vec_unchecked(
        //         StrideShape {
        //             dim: self.dim,
        //             strides: shape_builder::Strides::Custom(self.strides)
        //         }, result)
        // };
        // dbg!(array.ptr);
        // array.ptr = unsafe { array.ptr.add(WgpuDevice::ptr_to_offset(self.ptr).try_into().unwrap()) };
        // array.ptr = unsafe { array.ptr.add(4) };
        // dbg!(array.ptr);
        return array;
    }

    pub fn get_wgpu_device(&'d self) -> &'d WgpuDevice {
        self.data.wgpu_device
    }
}


impl <A,D> Clone for WgpuArray<'_, A, D>
where
    A: bytemuck::Pod,
    D: Dimension,
{
    fn clone(&self) -> Self {
        // Below shows how to do a deep clone
        // let (data, mut ptr) = self.data.deep_clone();
        // ptr = unsafe { ptr.add(WgpuDevice::ptr_to_offset(self.ptr)) };
        // But we don't need a deep clone because device buffers are never modified
        // all kernel operations return a new buffer
        let data = self.data.clone();
        let ptr = self.ptr;
        WgpuArray {
            data,
            ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
        }
    }
}
