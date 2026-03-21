/// A sliced tensor view must not outlive the tensor it was sliced from,
/// when the original tensor owns the data. The borrow checker must reject
/// using the slice after the original is consumed.
fn main() {
    use tk_core::{DenseTensor, TensorShape};

    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let t = DenseTensor::from_vec(TensorShape::row_major(&[3, 4]), data);
    let sliced = t.slice_axis(0, 1, 3); // borrows from `t`
    drop(t); // move `t` — invalidates the borrow
    let _ = sliced.as_slice(); // ERROR: `t` was moved while `sliced` borrows it
}
