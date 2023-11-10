use super::*;
use cudart::{
    memory::{CudaHostAllocFlags, HostAllocation},
    slice::{CudaSlice, CudaSliceMut},
};
use derivative::*;
use std::alloc::{Allocator, Global, Layout};
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Derivative)]
#[derivative(Clone, Debug)]
pub struct StaticHostAllocator {
    memory: Arc<HostAllocation<u8>>,
    memory_size: usize,
    block_size_in_bytes: usize,
    // TODO: Can we use deque
    bitmap: Arc<Vec<AtomicBool>>,
}

impl Default for StaticHostAllocator {
    fn default() -> Self {
        _host_alloc().clone()
    }
}

impl StaticAllocator for StaticHostAllocator {}
impl GoodAllocator for StaticHostAllocator {}

impl StaticHostAllocator {
    fn init_bitmap(num_blocks: usize) -> Vec<AtomicBool> {
        let mut bitmap = vec![];
        for _ in 0..num_blocks {
            bitmap.push(AtomicBool::new(false))
        }

        bitmap
    }

    pub fn as_ptr(&self) -> *const u8 {
        use cudart::slice::CudaSlice;
        self.memory.as_ptr()
    }

    pub fn block_size_in_bytes(&self) -> usize {
        self.block_size_in_bytes
    }

    
    pub fn init(memory_size_in_bytes: usize) -> CudaResult<Self> {
        // assert!(memory_size_in_bytes.is_power_of_two());
        // let block_size_in_bytes = std::mem::size_of::<F>();
        let block_size_in_bytes = 4;
        assert_eq!(memory_size_in_bytes % block_size_in_bytes, 0);
        let memory =
            HostAllocation::alloc(memory_size_in_bytes, CudaHostAllocFlags::DEFAULT).expect(
                &format!("failed to allocate {} bytes", memory_size_in_bytes),
            );

            println!("allocated {} bytes({}gb) on device on host", memory_size_in_bytes, memory_size_in_bytes / 0x40000000);
        let num_blocks = memory_size_in_bytes / block_size_in_bytes;
        let alloc = StaticHostAllocator {
            memory: Arc::new(memory),
            memory_size: memory_size_in_bytes,
            block_size_in_bytes,
            bitmap: Arc::new(Self::init_bitmap(num_blocks)),
        };

        Ok(alloc)
    }

    fn find_free_block(&self) -> Option<usize> {
        for (idx, entry) in self.bitmap.iter().enumerate() {
            if entry
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return Some(idx);
            }
        }
        None
    }
    // TODO: handle thread-safety
    fn find_adjacent_free_blocks(
        &self,
        requested_num_blocks: usize,
    ) -> Option<std::ops::Range<usize>> {
        if requested_num_blocks > self.bitmap.len() {
            return None;
        }
        let mut range_of_blocks_found = false;
        let mut found_range = 0..0;

        let mut start = 0;
        let mut end = requested_num_blocks;
        let mut busy_block_idx = 0;
        loop {
            let mut has_busy_block = false;
            for (idx, sub_entry) in self.bitmap[start..end].iter().enumerate() {
                if sub_entry.load(Ordering::SeqCst) {
                    has_busy_block = true;
                    busy_block_idx = start + idx;
                }
            }
            if has_busy_block == false {
                for entry in self.bitmap[start..end].iter() {
                    entry.store(true, Ordering::SeqCst);
                }
                return Some(start..end);
            } else {
                start = busy_block_idx + 1;
                end = start + requested_num_blocks;
                if end > self.bitmap.len() {
                    break;
                }
            }
        }
        panic!("not found block {} {} {}", start, end, self.bitmap.len());
        None
    }

    fn free_block(&self, index: usize) {
        assert!(self.bitmap[index].load(Ordering::SeqCst));
        self.bitmap[index]
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
    }

    pub fn free(self) -> CudaResult<()> {
        println!("freeing static host allocation");
        assert_eq!(Arc::weak_count(&self.memory), 0);
        assert_eq!(Arc::strong_count(&self.memory), 1);
        let Self { memory, .. } = self;
        let memory = Arc::try_unwrap(memory).expect("exclusive access");
        memory.free()?;
        Ok(())
    }
}

unsafe impl Send for StaticHostAllocator {}
unsafe impl Sync for StaticHostAllocator {}

unsafe impl Allocator for StaticHostAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        let size = layout.size();
        assert!(size > 0);
        assert_eq!(size % self.block_size_in_bytes, 0);
        let alignment = layout.align();
        if size > self.block_size_in_bytes {
            let num_blocks = size / self.block_size_in_bytes;
            if let Some(range) = self.find_adjacent_free_blocks(num_blocks) {
                let index = range.start;
                let offset = index * self.block_size_in_bytes;
                let ptr = unsafe { self.as_ptr().add(offset) };
                let ptr = unsafe { NonNull::new_unchecked(ptr as _) };
                return Ok(NonNull::slice_from_raw_parts(ptr, size));
            }
            panic!("allocation of {} blocks has failed", num_blocks);
            return Err(std::alloc::AllocError);
        }

        if let Some(index) = self.find_free_block() {
            let offset = index * self.block_size_in_bytes;
            let ptr = unsafe { self.as_ptr().add(offset) };
            let ptr = unsafe { NonNull::new_unchecked(ptr as _) };
            Ok(NonNull::slice_from_raw_parts(ptr, size))
        } else {
            panic!("allocation of 1 block has failed");
            Err(std::alloc::AllocError)
        }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        todo!()
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        assert!(size > 0);
        assert_eq!(size % self.block_size_in_bytes, 0);
        let offset = unsafe { ptr.as_ptr().offset_from(self.as_ptr()) } as usize;
        assert_eq!(offset % self.block_size_in_bytes, 0);
        let index = offset / self.block_size_in_bytes;
        let num_blocks = size / self.block_size_in_bytes;
        assert!(num_blocks > 0);
        for actual_idx in index..index + num_blocks {
            self.free_block(actual_idx);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SmallStaticHostAllocator {
    inner: StaticHostAllocator,
}

impl SmallStaticHostAllocator {
    pub fn init() -> CudaResult<Self> {      
        let inner = StaticHostAllocator::init(SMALL_HOST_ALLOCATION_SIZE)?;
        Ok(Self { inner })
    }

    pub fn free(self) -> CudaResult<()> {
        self.inner.free()
    }

    pub fn block_size_in_bytes(&self) -> usize {
        self.inner.block_size_in_bytes
    }
}

unsafe impl Allocator for SmallStaticHostAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        self.inner.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.inner.deallocate(ptr, layout)
    }
}

impl StaticAllocator for SmallStaticHostAllocator {}
impl SmallStaticAllocator for SmallStaticHostAllocator {}
impl GoodAllocator for SmallStaticHostAllocator {}
