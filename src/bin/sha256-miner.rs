#![feature(decl_macro)]

use anyhow::anyhow;
use bytemuck::{cast_slice, cast_slice_mut};
use std::borrow::Cow;
use std::process::exit;
use std::time::Instant;
use tokio::sync::oneshot;
use wgpu::wgt::PollType;
use wgpu::{
    include_spirv_raw, include_wgsl, Backends, BindGroup, BindGroupDescriptor,
    BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor,
    Queue, ShaderModuleDescriptorPassthrough, ShaderStages,
};

/// Sha256 buffer type the shader uses.
type FatSha256Buf = [u32; SHA256_BYTES];

const SHA256_BYTES: usize = 32;
const INPUT_SIZE: usize = 32;
/// The shader treats `u32`s as `u8`s.
const BLOCK_BUFFER_IN_SHADER: u64 = size_of::<FatSha256Buf>() as _;

use clap::Parser;
use num_format::{Locale, ToFormattedString};
use wgpu_benchmarks::{default, set_up_logger};

#[derive(Parser, Debug)]
#[command(about = "GPU Sha256 Miner Simulator")]
struct Args {
    /// Number of threads per workgroup (WORKGROUP_SIZE)
    #[arg(long, default_value_t = 256)]
    workgroup_size: u32,

    /// Number of workgroups to dispatch in the X dimension (DISPATCH_X)
    #[arg(long, default_value_t = 2048)]
    dispatch_x: u32,

    /// Number of hash iterations performed by each individual thread
    #[arg(short, long, default_value_t = 64)]
    iterations: u32,

    /// Target difficulty in bits
    #[arg(short, long, default_value_t = 32)]
    difficulty: u32,

    /// The start hex data (in hex string).
    #[arg(long)]
    start: Option<String>,
}

struct State {
    device: Device,
    queue: Queue,
    pipeline: ComputePipeline,
    input_buffer: Buffer,
    result_buffer: Buffer,
    map_read_buffer: Buffer,
    bind_group: BindGroup,
}

impl State {
    async fn new(args: &Args) -> anyhow::Result<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::from_env().unwrap_or_default(),
            ..default!()
        });
        let adapter = instance.request_adapter(&default!()).await?;
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_features: Features::EXPERIMENTAL_PASSTHROUGH_SHADERS,
                experimental_features: unsafe { ExperimentalFeatures::enabled() },
                ..default!()
            })
            .await?;

        // let shader_module = device.create_shader_module(ShaderModuleDescriptor {
        //     label: None,
        //     source: ShaderSource::Wgsl(wgsl_source(args.difficulty).into()),
        // });
        // let mut desc = include_spirv_raw!("../../shader.spv");
        // desc.entry_point = "main".into();

        // let desc = include_wgsl!("../../sha256-miner-d32.wgsl");

        let desc = ShaderModuleDescriptorPassthrough {
            entry_point: "main".to_string(),
            label: None,
            num_workgroups: (256, 0, 0),
            runtime_checks: Default::default(),
            spirv: None,
            dxil: Some(Cow::Borrowed(include_bytes!("../../shader.dxil"))),
            msl: None,
            hlsl: None,
            glsl: None,
            wgsl: None,
        };
        let shader_module = unsafe { device.create_shader_module_passthrough(desc) };
        // let shader_module = device.create_shader_module(desc);

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions {
                constants: &[
                    // ("WORKGROUP_SIZE", args.workgroup_size as f64),
                    // ("ITERATIONS_PER_THREAD", args.iterations as f64),
                    // (
                    //     "RUNS_PER_DISPATCH",
                    //     (args.dispatch_x * args.workgroup_size) as f64,
                    // ),
                    // ("DIFFICULTY_BITS", args.difficulty as f64),
                ],
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });

        let input_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: INPUT_SIZE as u64 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let result_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: BLOCK_BUFFER_IN_SHADER,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let map_read_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: BLOCK_BUFFER_IN_SHADER,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &input_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &result_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        Ok(Self {
            queue,
            device,
            pipeline,
            input_buffer,
            bind_group,
            result_buffer,
            map_read_buffer,
        })
    }

    fn write_input_data(&self, buf: &[u8]) {
        let mut input_data = [0_u32; INPUT_SIZE];
        for (i, &b) in buf.iter().enumerate() {
            input_data[i] = b as _;
        }
        self.queue
            .write_buffer(&self.input_buffer, 0, cast_slice(&input_data));
    }

    fn compute_dispatch(&self, workgroups_x: u32) {
        let mut encoder = self.device.create_command_encoder(&default!());

        let mut pass = encoder.begin_compute_pass(&default!());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, default!());
        pass.dispatch_workgroups(workgroups_x, 1, 1);
        drop(pass);

        encoder.copy_buffer_to_buffer(&self.result_buffer, 0, &self.map_read_buffer, 0, None);

        let command_buffer = encoder.finish();
        self.queue.submit([command_buffer]);
    }

    async fn read_result(&self, to: &mut [u8]) -> anyhow::Result<()> {
        let (tx, rx) = oneshot::channel();
        self.map_read_buffer.map_async(MapMode::Read, .., |e| {
            tx.send(e).unwrap();
        });
        self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        rx.await??;

        to[..(self.map_read_buffer.size() as usize)]
            .copy_from_slice(cast_slice(&*self.map_read_buffer.get_mapped_range(..)));
        self.map_read_buffer.unmap();
        Ok(())
    }
}

fn add_big_int(data: &mut [u8; 32], n: u32) {
    let mut carry = n;

    for byte in data.iter_mut() {
        if carry == 0 {
            break;
        }

        let sum = *byte as u32 + carry;

        *byte = (sum & 0xFF) as u8;

        carry = sum >> 8;
    }
}

#[inline(always)]
fn convert_fat_buf(buf: &FatSha256Buf) -> [u8; SHA256_BYTES] {
    buf.map(|x| x as u8)
}

fn generate_check_difficulty_wgsl(difficulty_bits: u32) -> String {
    let mut conditions = Vec::new();

    let full_bytes = difficulty_bits / 8;
    for i in 0..full_bytes {
        conditions.push(format!("buf[{}] == 0u", i));
    }

    let remaining_bits = difficulty_bits % 8;
    if remaining_bits > 0 {
        let shift = 8 - remaining_bits;
        conditions.push(format!("(buf[{}] >> {}u) == 0u", full_bytes, shift));
    }

    let final_condition = if conditions.is_empty() {
        "true".to_string()
    } else {
        conditions.join(" && ")
    };

    format!(
        r#"
fn check_difficulty(buf: ptr<function, array<u32, SHA256_BLOCK_SIZE>>) -> bool {{
    return {};
}}
"#,
        final_condition
    )
}

fn wgsl_source(difficulty_bits: u32) -> String {
    let mut source = include_str!("../sha256-miner.wgsl")
        .lines()
        .collect::<Vec<_>>();
    source.remove(0);
    let generated = generate_check_difficulty_wgsl(difficulty_bits);
    for x in generated.lines().into_iter().rev() {
        source.insert(0, x);
    }
    source.join("\n")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let program_start = Instant::now();
    set_up_logger();

    let args = Args::parse();
    let runs_per_dispatch = args.dispatch_x * args.workgroup_size;

    eprintln!("Args: {:?}", args);

    let arg_start = hex::decode(args.start.as_ref().map(|x| x.as_str()).unwrap_or_default())?;
    if arg_start.len() > 32 {
        return Err(anyhow!("Length of `start` must be <= 32"));
    }

    let state = State::new(&args).await?;
    let mut input_data = [0_u8; INPUT_SIZE];
    input_data[..arg_start.len()].copy_from_slice(&arg_start);
    let mut result = [0_u32; SHA256_BYTES];
    let mut counter = 0_usize;
    let compute_start = Instant::now();
    let mut hashes = 0_u64;
    loop {
        eprintln!(
            "dispatch: {}, start: {}, elapsed: {:?}, hashes: {}, hashrate: {} H/s",
            counter,
            hex::encode(input_data),
            compute_start.elapsed(),
            hashes.to_formatted_string(&Locale::en),
            ((hashes as f64 / compute_start.elapsed().as_secs_f64()).round() as u64)
                .to_formatted_string(&Locale::en)
        );
        state.write_input_data(&input_data);
        state.compute_dispatch(args.dispatch_x);
        let hashes_computed = runs_per_dispatch * args.iterations;
        hashes += hashes_computed as u64;
        add_big_int(&mut input_data, hashes_computed);
        state.read_result(cast_slice_mut(&mut result)).await?;
        if result.iter().any(|x| *x != 0) {
            // print result and exit
            use sha2::Digest;
            let mut hasher = sha2::Sha256::new();
            hasher.update(convert_fat_buf(&result));
            let hash = hex::encode(hasher.finalize());

            println!("Result:");
            println!("  input: {}", hex::encode(convert_fat_buf(&result)));
            println!("  sha256: {}", hash);
            println!(
                "  preparation time: {:?}",
                compute_start.duration_since(program_start)
            );
            println!("  computation elapsed: {:?}", compute_start.elapsed());
            exit(0);
        }
        counter += 1;
    }
}
