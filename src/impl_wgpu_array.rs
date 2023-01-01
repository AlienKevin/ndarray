use crate::Array;
use crate::Dimension;
use crate::WgpuArray;
use crate::WgpuDevice;
use futures::executor;

impl<'d, A, D> WgpuArray<'d, A, D>
where
    A: bytemuck::Pod + std::fmt::Debug,
    D: Dimension,
{
    pub fn into_cpu(self) -> Array<A, D> {
        let buffer_slice = self.data.storage_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        
        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.data.wgpu_device.device.poll(wgpu::Maintain::Wait);

        // Awaits until `buffer_future` can be read from
        if let Some(Ok(())) = executor::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<A> = data
                .chunks_exact(std::mem::size_of::<A>())
                .map(|b| *bytemuck::from_bytes::<A>(b))
                .collect();
            let array = unsafe { Array::<A, D>::from_shape_vec_unchecked(self.dim, result) };
            return array;
        }
        todo!()
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
        WgpuArray {
            data: self.data.clone(),
            ptr: self.ptr,
            dim: self.dim.clone(),
            strides: self.strides.clone(),
        }
    }
}
