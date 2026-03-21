/// A TempTensor must not escape the scope of the arena it was allocated from.
/// Returning a borrowed tensor from a function that owns the arena must fail.
fn make_tensor() -> tk_core::DenseTensor<'static, f64> {
    use tk_core::{SweepArena, TensorShape};

    let arena = SweepArena::new(1024);
    let t = arena.alloc_tensor::<f64>(TensorShape::row_major(&[2, 3]));
    t.into_owned() // this is fine — but returning `t` directly should fail
}

fn make_tensor_bad() -> tk_core::DenseTensor<'static, f64> {
    use tk_core::{SweepArena, TensorShape};

    let arena = SweepArena::new(1024);
    let t = arena.alloc_tensor::<f64>(TensorShape::row_major(&[2, 3]));
    t // ERROR: cannot return value referencing local variable `arena`
}

fn main() {
    let _ = make_tensor_bad();
}
