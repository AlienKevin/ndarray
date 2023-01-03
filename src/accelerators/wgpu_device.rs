use wgpu::util::DeviceExt;
use parking_lot::Mutex;
use core::ptr::NonNull;
use std::sync::Arc;

lazy_static::lazy_static! {
    static ref WGPU_BUFFERS: Mutex<Vec<((usize, usize), Arc<wgpu::Buffer>)>> = Mutex::new(vec![]);
}

pub struct WgpuDevice {
    pub device: wgpu::Device,
    pub adapter: wgpu::Adapter,
    pub queue: wgpu::Queue,
    pub workgroup_size: u32,
}

impl WgpuDevice {
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        Some(WgpuDevice {
            device,
            adapter,
            queue,
            workgroup_size: 8,
        })
    }

    pub fn ptr_to_buffer<A>(ptr: NonNull<A>) -> Arc<wgpu::Buffer> {
        let ptr = ptr.as_ptr() as usize;
        for ((start, end), buffer) in &*WGPU_BUFFERS.lock() {
            if ptr >= *start && ptr < *end {
                return buffer.clone();
            }
        }
        panic!("Invalid ptr to wgpu buffer");
    }

    pub fn ptr_to_offset<A>(ptr: NonNull<A>) -> usize {
        let ptr = ptr.as_ptr() as usize;
        for ((start, end), _buffer) in &*WGPU_BUFFERS.lock() {
            if ptr >= *start && ptr < *end {
                return (ptr - *start) / std::mem::size_of::<A>();
            }
        }
        panic!("Invalid ptr to wgpu buffer");
    }
    
    fn allocate_buffer<A: Sized>(buffer: wgpu::Buffer) -> (Arc<wgpu::Buffer>, NonNull<A>) {
        let buffers = &mut *WGPU_BUFFERS.lock();
        let arc_buffer = Arc::new(buffer);
        let alignment = std::mem::align_of::<A>();
        let start;
        if buffers.is_empty() {
            start = alignment;
            let end = start + arc_buffer.size() as usize;
            buffers.push(((start, end), arc_buffer.clone()));
        } else {
            let last_buffer = &buffers[buffers.len() - 1];
            let (_, last_buffer_end) = last_buffer.0;
            start = (last_buffer_end / alignment + 1) * alignment;
            let end = start + arc_buffer.size() as usize;
            buffers.push(((start, end), arc_buffer.clone()));
        }
        (arc_buffer, unsafe { NonNull::new_unchecked(start as *mut A) })
    }

    pub fn create_storage_buffer<A: bytemuck::Pod>(&self, slice: &[A]) -> wgpu::Buffer {
        let storage_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(slice),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        
        storage_buffer
    }

    pub fn create_storage_buffer_sized<A: bytemuck::Pod>(&self, size: u64) -> wgpu::Buffer {
        let storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Storage Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        storage_buffer
    }

    pub fn allocate_storage_buffer<A: bytemuck::Pod>(&self, slice: &[A]) -> (Arc<wgpu::Buffer>, NonNull<A>) {
        Self::allocate_buffer::<A>(self.create_storage_buffer(slice))
    }

    pub fn allocate_storage_buffer_sized<A: bytemuck::Pod>(&self, size: u64) -> (Arc<wgpu::Buffer>, NonNull<A>) {
        Self::allocate_buffer::<A>(self.create_storage_buffer_sized::<A>(size))
    }

    pub fn create_staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}
