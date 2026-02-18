use bytemuck::{bytes_of, Pod, Zeroable};
use std::time::{Duration, Instant};
use wgpu::{
    include_wgsl, Instance, LoadOpDontCare, PipelineCompilationOptions, Surface, TextureFormat,
};
use wgpu::util::RenderEncoder;

macro_rules! default {
    () => {
        Default::default()
    };
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Immediates {
    origin: [f32; 3],
    padding1: f32,
    right: [f32; 3],
    padding2: f32,
    up: [f32; 3],
    padding3: f32,
    forward: [f32; 3],
    padding4: f32,
    screen_size: [f32; 2],
    len: f32,
    padding5: f32,
}

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub size: (u32, u32),
    render_pipeline: wgpu::RenderPipeline,
    elapsed: f32,
    texture_format: wgpu::TextureFormat,
    immediates: Immediates,
}

pub struct Config {
    pub kernel_iterations: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            kernel_iterations: 5,
        }
    }
}

impl State {
    pub fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.texture_format,
            view_formats: vec![self.texture_format],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.0,
            height: self.size.1,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::Immediate,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    pub async fn new(instance: Instance, surface: Surface<'static>, size: (u32, u32)) -> Self {
        let adapter = instance.request_adapter(&default!()).await.unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::IMMEDIATES,
                required_limits: wgpu::Limits {
                    max_immediate_size: 80,
                    ..default!()
                },
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        // Do not use srgb suffix. This makes wgpu think all colors we give are already in a
        // non-linear sRGB space and do not do an automatic gamma correction.
        let mut texture_format = TextureFormat::Bgra8Unorm;
        if !surface_caps.formats.iter().any(|x| x == &texture_format) {
            texture_format = surface_caps.formats[0].remove_srgb_suffix();
        }

        let shader = device.create_shader_module(include_wgsl!("vsbm.wgsl"));
        // let shader = device.create_shader_module(include_spirv!("../a.spv"));
        // let shader_vs = device.create_shader_module(include_spirv!("../vs.spv"));
        // let shader_fs = device.create_shader_module(include_spirv!("../fs.spv"));
        let shader_vs = &shader;
        let shader_fs = &shader;

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                immediate_size: 80,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_vs,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_fs,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: default!(),
                    // constants: &[("KERNEL_ITERATIONS", config.kernel_iterations as f64)],
                    constants: &[],
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: None,
                    write_mask: Default::default(),
                })],
            }),
            multiview_mask: None,
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
        });

        let state = Self {
            surface,
            device,
            queue,
            size,
            render_pipeline,
            elapsed: 0f32,
            texture_format,
            immediates: Zeroable::zeroed(),
        };
        state.configure_surface();
        state
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        self.size = new_size;

        // reconfigure the surface
        self.configure_surface();
    }

    fn update(&mut self) {
        self.elapsed += 0.012;
        let ang1 = 2.8 + self.elapsed * 0.5; // rotation
        let ang2: f32 = 0.4;
        let len = 1.6;

        let origin = [
            len * ang1.cos() * ang2.cos(),
            len * ang2.sin(),
            len * ang1.sin() * ang2.cos(),
        ];
        let right = [ang1.sin(), 0.0, -ang1.cos()];
        let up = [
            -ang2.sin() * ang1.cos(),
            ang2.cos(),
            -ang2.sin() * ang1.sin(),
        ];
        let forward = [
            -ang1.cos() * ang2.cos(),
            -ang2.sin(),
            -ang1.sin() * ang2.cos(),
        ];

        self.immediates = Immediates {
            origin,
            padding1: 0.0,
            right,
            padding2: 0.0,
            up,
            padding3: 0.0,
            forward,
            padding4: 0.0,
            screen_size: [1.0, 1.0],
            len,
            padding5: 0.0,
        };
    }

    pub fn frame(
        &mut self,
        before_submit_callback: impl FnOnce(),
    ) -> Result<(), wgpu::SurfaceError> {
        self.update();

        let surface_texture = self.surface.get_current_texture()?;

        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.texture_format),
                ..Default::default()
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(LoadOpDontCare::default()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.render_pipeline);
            pass.set_immediates(0, bytes_of(&self.immediates));
            pass.draw(0..6, 0..1);
        }
        let command_buffer = encoder.finish();

        before_submit_callback();
        self.queue.submit([command_buffer]);
        surface_texture.present();
        Ok(())
    }
}

pub struct Fps {
    instant: Instant,
    counter: usize,
}

impl Fps {
    pub fn new() -> Self {
        Self {
            instant: Instant::now(),
            counter: 0,
        }
    }

    pub fn hint_and_get(&mut self) -> (Duration, f32) {
        self.counter += 1;
        let duration = self.instant.elapsed();
        (
            duration,
            (self.counter as f64 / duration.as_secs_f64()) as f32,
        )
    }
}
