use std::env;
use std::time::{Duration, Instant};

pub fn set_up_logger() {
    unsafe {
        env::set_var("RUST_LOG", "info");
    }
    env_logger::init();
}

#[macro_export]
macro_rules! default {
    () => {
        Default::default()
    };
}

pub struct FpsStat {
    instant: Instant,
    counter: usize,
}

impl FpsStat {
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
