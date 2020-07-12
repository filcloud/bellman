use crate::gpu::get_platform;
use crate::gpu::{
    error::{GPUError, GPUResult},
    get_devices, locks, sources, structs, utils,
};
use ff::Field;
use log::*;
use ocl::{Buffer, MemFlags, ProQue};
use paired::Engine;
use std::cmp;

// NOTE: Please read `structs.rs` for an explanation for unsafe transmutes of this code!

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_RADIX_DEGREE: u32 = 8; // Radix256
const MAX_LOCAL_WORK_SIZE_DEGREE: u32 = 7; // 128

pub struct FFTKernel<E>
where
    E: Engine,
{
    proque: ProQue,
    fft_pq_buffer: Buffer<structs::PrimeFieldStruct<E::Fr>>,
    fft_omg_buffer: Buffer<structs::PrimeFieldStruct<E::Fr>>,
    _lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
    priority: bool,
}

impl<E> FFTKernel<E>
where
    E: Engine,
{
    pub fn create(n: u32, priority: bool) -> GPUResult<FFTKernel<E>> {
        let lock = locks::GPULock::lock();
        let src = sources::kernel::<E>();

        let platform = get_platform(None)?;
        info!("Platform selected: {}", platform.name()?);

        let devices = get_devices(&platform).unwrap_or_default();
        if devices.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }

        // Select the first device for FFT
        let device = devices[0];

        let pq = ProQue::builder()
            .platform(platform)
            .device(device)
            .src(src)
            .dims(n)
            .build()?;
        let pqbuff = Buffer::builder()
            .queue(pq.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(1 << MAX_RADIX_DEGREE >> 1)
            .build()?;
        let omgbuff = Buffer::builder()
            .queue(pq.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(LOG2_MAX_ELEMENTS)
            .build()?;

        info!("FFT: 1 working device(s) selected.");
        info!("FFT: Device 0: {}", pq.device().name()?);

        Ok(FFTKernel {
            proque: pq,
            fft_pq_buffer: pqbuff,
            fft_omg_buffer: omgbuff,
            _lock: lock,
            priority,
        })
    }

    /// Peforms a FFT round
    /// * `lgn` - Specifies log2 of number of elements
    /// * `lgp` - Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    /// * `deg` - 1=>radix2, 2=>radix4, 3=>radix8, ...
    /// * `max_deg` - The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    fn radix_fft_round(
        &mut self,
        fft_src_buffer: &Buffer<structs::PrimeFieldStruct<E::Fr>>,
        fft_dst_buffer: &Buffer<structs::PrimeFieldStruct<E::Fr>>,
        lgn: u32,
        lgp: u32,
        deg: u32,
        max_deg: u32,
        in_src: bool,
    ) -> GPUResult<()> {
        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        let n = 1u32 << lgn;
        let lwsd = cmp::min(deg - 1, MAX_LOCAL_WORK_SIZE_DEGREE);
        let kernel = self
            .proque
            .kernel_builder("radix_fft")
            .global_work_size([n >> deg << lwsd])
            .local_work_size(1 << lwsd)
            .arg(if in_src {
                fft_src_buffer
            } else {
                fft_dst_buffer
            })
            .arg(if in_src {
                fft_dst_buffer
            } else {
                fft_src_buffer
            })
            .arg(&self.fft_pq_buffer)
            .arg(&self.fft_omg_buffer)
            .arg_local::<structs::PrimeFieldStruct<E::Fr>>(1 << deg)
            .arg(n)
            .arg(lgp)
            .arg(deg)
            .arg(max_deg)
            .build()?;
        unsafe {
            kernel.enq()?;
        } // Running a GPU kernel is unsafe!
        Ok(())
    }

    /// Share some precalculated values between threads to boost the performance
    fn setup_pq(&mut self, omega: &E::Fr, n: usize, max_deg: u32) -> ocl::Result<()> {
        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut tpq = vec![structs::PrimeFieldStruct::<E::Fr>::default(); 1 << max_deg >> 1];
        let pq = unsafe {
            std::mem::transmute::<&mut [structs::PrimeFieldStruct<E::Fr>], &mut [E::Fr]>(&mut tpq)
        };
        let tw = omega.pow([(n >> max_deg) as u64]);
        pq[0] = E::Fr::one();
        if max_deg > 1 {
            pq[1] = tw;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&tw);
            }
        }
        self.fft_pq_buffer.write(&tpq).enq()?;

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut tom = vec![structs::PrimeFieldStruct::<E::Fr>::default(); 32];
        let om = unsafe {
            std::mem::transmute::<&mut [structs::PrimeFieldStruct<E::Fr>], &mut [E::Fr]>(&mut tom)
        };
        om[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            om[i] = om[i - 1].pow([2u64]);
        }
        self.fft_omg_buffer.write(&tom).enq()?;

        Ok(())
    }

    /// Performs FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `lgn` - Specifies log2 of number of elements
    pub fn radix_fft(&mut self, a: &mut [E::Fr], omega: &E::Fr, lgn: u32) -> GPUResult<()> {
        let n = 1 << lgn;

        let fft_src_buffer = Buffer::<structs::PrimeFieldStruct<E::Fr>>::builder()
            .queue(self.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(n)
            .build()?;
        let fft_dst_buffer = Buffer::<structs::PrimeFieldStruct<E::Fr>>::builder()
            .queue(self.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(n)
            .build()?;

        let ta = unsafe {
            std::mem::transmute::<&mut [E::Fr], &mut [structs::PrimeFieldStruct<E::Fr>]>(a)
        };

        let max_deg = cmp::min(MAX_RADIX_DEGREE, lgn);
        self.setup_pq(omega, n, max_deg)?;

        fft_src_buffer.write(&*ta).enq()?;
        let mut in_src = true;
        let mut lgp = 0u32;
        while lgp < lgn {
            let deg = cmp::min(max_deg, lgn - lgp);
            self.radix_fft_round(
                &fft_src_buffer,
                &fft_dst_buffer,
                lgn,
                lgp,
                deg,
                max_deg,
                in_src,
            )?;
            lgp += deg;
            in_src = !in_src; // Destination of this FFT round is source of the next round.
        }
        if in_src {
            fft_src_buffer.read(ta).enq()?;
        } else {
            fft_dst_buffer.read(ta).enq()?;
        }
        self.proque.finish()?; // Wait for all commands in the queue (Including read command)

        Ok(())
    }

    /// Performs inplace FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `lgn` - Specifies log2 of number of elements
    pub fn inplace_fft(&mut self, a: &mut [E::Fr], omega: &E::Fr, lgn: u32) -> GPUResult<()> {
        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        let n = 1 << lgn;

        let fft_buffer = Buffer::<structs::PrimeFieldStruct<E::Fr>>::builder()
            .queue(self.proque.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(n)
            .build()?;

        let ta = unsafe {
            std::mem::transmute::<&mut [E::Fr], &mut [structs::PrimeFieldStruct<E::Fr>]>(a)
        };

        let max_deg = cmp::min(MAX_RADIX_DEGREE, lgn);
        self.setup_pq(omega, n, max_deg)?;

        fft_buffer.write(&*ta).enq()?;

        let kernel = self
            .proque
            .kernel_builder("reverse_bits")
            .global_work_size([n])
            .arg(&fft_buffer)
            .arg(lgn)
            .build()?;
        unsafe {
            kernel.enq()?;
        } // Running a GPU kernel is unsafe!

        for lgm in 0..lgn {
            let kernel = self
                .proque
                .kernel_builder("inplace_fft")
                .global_work_size([n >> 1])
                .arg(&fft_buffer)
                .arg(&self.fft_omg_buffer)
                .arg(lgn)
                .arg(lgm)
                .build()?;
            unsafe {
                kernel.enq()?;
            } // Running a GPU kernel is unsafe!
        }

        fft_buffer.read(ta).enq()?;
        self.proque.finish()?; // Wait for all commands in the queue (Including read command)

        Ok(())
    }

    pub fn fft(&mut self, a: &mut [E::Fr], omega: &E::Fr, lgn: u32) -> GPUResult<()> {
        const MIN_RADIX_MEMORY: u64 = 9 * 1024 * 1024 * 1024; // 9GB
        let d = self.proque.device();
        let mem = utils::get_memory(d)?;
        if mem > MIN_RADIX_MEMORY {
            self.radix_fft(a, omega, lgn)
        } else {
            warn!("FFT: Memory not enough for radix_fft! Using inplace_fft instead...");
            self.inplace_fft(a, omega, lgn)
        }
    }
}
