/// Check if allocation size is safe for stack
pub const fn check_stack_size<T>() {
    const MAX_SAFE_STACK_ALLOC: usize = 64 * 1024; // 64 KB
    
    let size = std::mem::size_of::<T>();
    
    if size > MAX_SAFE_STACK_ALLOC {
        panic!("Type too large for safe stack allocation");
    }
}

/// Wrapper that enforces size limit
pub struct StackSafe<T> {
    inner: T,
}

impl<T> StackSafe<T> {
    pub const fn new(inner: T) -> Self {
        check_stack_size::<T>();
        Self { inner }
    }
}
