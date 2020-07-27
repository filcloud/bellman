use log::{info, warn, error};
use rust_gpu_tools::*;
use crate::gpu::error::{GPUError, GPUResult};

use std::collections::HashMap;
use std::env;

lazy_static::lazy_static! {
    static ref CORE_COUNTS: HashMap<String, usize> = {
        let mut core_counts : HashMap<String, usize> = vec![
            // AMD
            ("gfx1010".to_string(), 2560),
            // This value was chosen to give (approximately) empirically best performance for a Radeon Pro VII.
            ("gfx906".to_string(), 7400),

            // NVIDIA
            ("Quadro RTX 6000".to_string(), 4608),

            ("TITAN RTX".to_string(), 4608),

            ("Tesla V100".to_string(), 5120),
            ("Tesla P100".to_string(), 3584),
            ("Tesla T4".to_string(), 2560),
            ("Quadro M5000".to_string(), 2048),

            ("GeForce RTX 3090".to_string(), 10496),
            ("GeForce RTX 3080".to_string(), 8704),
            ("GeForce RTX 3070".to_string(), 5888),

            ("GeForce RTX 2080 Ti".to_string(), 4352),
            ("GeForce RTX 2080 SUPER".to_string(), 3072),
            ("GeForce RTX 2080".to_string(), 2944),
            ("GeForce RTX 2070 SUPER".to_string(), 2560),

            ("GeForce GTX 1080 Ti".to_string(), 3584),
            ("GeForce GTX 1080".to_string(), 2560),
            ("GeForce GTX 2060".to_string(), 1920),
            ("GeForce GTX 1660 Ti".to_string(), 1536),
            ("GeForce GTX 1060".to_string(), 1280),
            ("GeForce GTX 1650 SUPER".to_string(), 1280),
            ("GeForce GTX 1650".to_string(), 896),
        ].into_iter().collect();

        match env::var("BELLMAN_CUSTOM_GPU").and_then(|var| {
            for card in var.split(",") {
                let splitted = card.split(":").collect::<Vec<_>>();
                if splitted.len() != 2 { panic!("Invalid BELLMAN_CUSTOM_GPU!"); }
                let name = splitted[0].trim().to_string();
                let cores : usize = splitted[1].trim().parse().expect("Invalid BELLMAN_CUSTOM_GPU!");
                info!("Adding \"{}\" to GPU list with {} CUDA cores.", name, cores);
                core_counts.insert(name, cores);
            }
            Ok(())
        }) { Err(_) => { }, Ok(_) => { } }

        core_counts
    };
}

const DEFAULT_CORE_COUNT: usize = 2560;
pub fn get_core_count(d: &opencl::Device) -> usize {
    let name = d.name();
    match CORE_COUNTS.get(&name[..]) {
        Some(&cores) => cores,
        None => {
            warn!(
                "Number of CUDA cores for your device ({}) is unknown! Best performance is \
                 only achieved when the number of CUDA cores is known! You can find the \
                 instructions on how to support custom GPUs here: \
                 https://lotu.sh/en+hardware-mining",
                name
            );
            DEFAULT_CORE_COUNT
        }
    }
}

pub fn dump_device_list() {
    for d in opencl::Device::all() {
        info!("Device: {:?}", d);
    }
}

#[cfg(feature = "gpu")]
#[test]
pub fn test_list_devices() {
    let _ = env_logger::try_init();
    dump_device_list();
}

#[cfg(feature = "gpu")]
lazy_static::lazy_static! {
    pub static ref GPU_NVIDIA_DEVICES: Vec<opencl::Device> = opencl::Device::by_brand(opencl::Brand::Nvidia).unwrap_or(&vec![]).to_vec();
}

pub fn get_gpu_index() -> GPUResult<opencl::Device> {
    let devices = &GPU_NVIDIA_DEVICES;
    if devices.is_empty() {
        return Err(GPUError::Simple("No working GPUs found!"));
    }
    let index: usize = std::env::var("BELLMAN_GPU_INDEX").or::<std::env::VarError>(Ok(String::from("0")))
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid BELLMAN_GPU_INDEX! Defaulting to 0...");
                Ok(0)
            }
        }).unwrap();
    Ok(devices[index].clone())
}
