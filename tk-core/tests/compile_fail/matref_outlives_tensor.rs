/// A MatRef obtained from a DenseTensor must not outlive the tensor.
/// The borrow checker must reject using the MatRef after the tensor
/// is moved or dropped.
fn main() {
    use tk_core::{DenseTensor, TensorShape};

    let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let t = DenseTensor::from_vec(TensorShape::row_major(&[2, 3]), data);
    let mat = t.as_mat_ref().unwrap(); // borrows from `t`
    drop(t); // move `t`
    let _ = mat.get(0, 0); // ERROR: `t` was moved while `mat` borrows it
}
