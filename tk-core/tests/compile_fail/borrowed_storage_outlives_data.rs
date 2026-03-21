/// A TensorStorage::Borrowed must not outlive the data it borrows from.
/// The borrow checker must reject using borrowed storage after the
/// source Vec is dropped.
fn main() {
    use tk_core::storage::TensorStorage;

    let storage = {
        let data = vec![1.0_f64, 2.0, 3.0];
        TensorStorage::from_slice(&data)
        // `data` is dropped here
    };
    let _ = storage.as_slice(); // ERROR: `data` does not live long enough
}
