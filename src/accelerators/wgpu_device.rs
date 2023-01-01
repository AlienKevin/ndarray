use wgpu::util::DeviceExt;

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

    pub fn create_storage_buffer_sized(&self, size: u64) -> wgpu::Buffer {
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

    pub fn create_staging_buffer(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}
