/// A TempTensor allocated from a SweepArena must not be used after
/// the arena is reset. The borrow checker must reject this because
/// `reset(&mut self)` requires exclusive access, which is incompatible
/// with the shared borrow held by the tensor.
fn main() {
    use tk_core::{SweepArena, TensorShape};

    let mut arena = SweepArena::new(1024);
    let t = arena.alloc_tensor::<f64>(TensorShape::row_major(&[4, 4]));
    arena.reset();
    let _ = t.as_slice(); // use-after-reset: must not compile
}
