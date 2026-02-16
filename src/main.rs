#![feature(try_blocks)]

//! https://cznull.github.io/vsbm wgpu port
//!
//! At 1024x1024 surface dimension, DX12 on Windows 10 has ~5 fps higher than
//! Vulkan on Windows 10 & Vulkan on Linux. Test hardware: NVIDIA GeForce RTX 3060 Mobile / Max-Q.

use clap::Parser;
use std::env;
use std::sync::Arc;
use vsbm::{Fps, State};
use wgpu::{Backends, Instance, InstanceDescriptor, InstanceFlags};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
};

#[derive(Default)]
struct App {
    pub state: Option<State>,
    pub window: Option<Arc<Window>>,
    pub fps: Option<Fps>,
    pub max_frame: usize,
}

#[derive(Parser)]
struct Args {
    max_frame: Option<usize>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let width = if env::var("WAYLAND_DISPLAY").is_ok() {
            let my_wl_scale_factor: f64 = 1.333333;
            (1024.0 / my_wl_scale_factor) as u32
        } else {
            1024
        };
        // Create window object
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_resizable(false)
                        .with_inner_size(PhysicalSize::new(width, width)),
                )
                .unwrap(),
        );

        pollster::block_on(async {
            let result: anyhow::Result<()> = try {
                // let size = window.inner_size();
                let size = (1024, 1024);
                let instance = Instance::new(&InstanceDescriptor {
                    backends: Backends::from_env().unwrap_or_default(),
                    flags: InstanceFlags::from_env_or_default(),
                    memory_budget_thresholds: Default::default(),
                    backend_options: Default::default(),
                });
                let surface = instance.create_surface(Arc::clone(&window))?;
                let state = State::new(instance, surface, size).await;
                self.state = Some(state);
            };
            result
        })
        .unwrap();

        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = self.state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => state.resize((1024, 1024)),
            WindowEvent::RedrawRequested => {
                let Some(w) = &self.window else {
                    return;
                };

                match state.frame(|| w.pre_present_notify()) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{:?}", e),
                }
                self.max_frame -= 1;
                if self.max_frame == 0 {
                    event_loop.exit();
                    self.max_frame = usize::MAX;
                }

                // print the FPS
                if let Some(f) = &mut self.fps {
                    let (d, fps) = f.hint_and_get();
                    if d.as_secs_f64() > 1.0 {
                        println!("FPS: {}", fps);
                        self.fps = Some(Fps::new());
                    }
                } else {
                    self.fps = Some(Fps::new());
                }

                w.request_redraw();
            }
            _ => {}
        }
    }
}

pub fn main() {
    let args = Args::parse();
    unsafe {
        env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    app.max_frame = args.max_frame.unwrap_or(usize::MAX);
    event_loop.run_app(&mut app).unwrap();
}
