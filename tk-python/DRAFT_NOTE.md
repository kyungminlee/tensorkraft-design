# tk-python Draft Implementation Notes

## Status

- **`cargo check`**: Passes (0 errors, upstream warnings only)
- **`cargo test`**: 11/11 tests pass (requires `--no-default-features --features backend-faer,parallel` + `DYLD_LIBRARY_PATH` pointing to libpython)
- **`maturin build`**: Not tested (requires maturin installed in a Python environment)

## Files Created

```
tk-python/
├── Cargo.toml
├── pyproject.toml
├── build.rs
└── src/
    ├── lib.rs                    # #[pymodule] tensorkraft
    ├── error.rs                  # DmftError → PyErr bridge
    ├── monitor/mod.rs            # CancellationMonitor (AtomicBool + mpsc)
    ├── dispatch/
    │   ├── mod.rs                # DmftLoopVariant enum
    │   └── macros.rs             # dispatch_variant! / dispatch_variant_mut!
    ├── config/
    │   ├── mod.rs                # PyDmftConfig, PyDmrgConfig, PyTimeEvolutionConfig, PyLinearPredictionConfig
    │   └── defaults.rs           # (placeholder)
    ├── dmft/
    │   ├── mod.rs                # PyDmftLoop (#[pyclass])
    │   ├── solve.rs              # GIL-release + cancellation pattern
    │   └── stats.rs              # PyDmftStats
    ├── spectral/
    │   ├── mod.rs                # PySpectralFunction
    │   └── numpy_interop.rs      # (placeholder for future zero-copy)
    └── bath/
        └── mod.rs                # PyBathParameters
```

## Spec-vs-Reality Gaps

### 1. ComplexU1 variant impossible (BLOCKER)

The spec lists `ComplexU1(DMFTLoop<Complex<f64>, U1, DefaultDevice>)` as a variant.
This cannot compile because `DeviceFaer` does not implement `LinAlgBackend<Complex<f64>>`.

**Resolution**: Removed `ComplexU1` from `DmftLoopVariant`. Added `complex_u1()` static method that returns `RuntimeError` with an explanation. Only `RealU1` and `RealZ2` are available.

**Spec feedback**: The spec should annotate `ComplexU1` as blocked-on `tk-linalg` complex backend and not list it as an implementable variant.

### 2. PyO3 `extension-module` vs test linking

The spec does not mention the PyO3 `extension-module` feature conflict. When `pyo3/extension-module` is enabled (needed for cdylib wheels), test binaries cannot link against libpython.

**Resolution**: Moved `extension-module` behind a Cargo feature flag:
```toml
pyo3 = { version = "0.21", features = ["abi3-py38"] }  # no extension-module here

[features]
default = ["extension-module", "backend-faer", "parallel"]
extension-module = ["pyo3/extension-module"]
```

Tests must be run with `--no-default-features --features backend-faer,parallel`. On macOS, `DYLD_LIBRARY_PATH` must point to the libpython directory. The spec's testing section should document this pattern.

### 3. `DMRGConfig` is not Clone (config mirror pattern)

`DMRGConfig` contains `Box<dyn IterativeEigensolver<f64>>`, which is not `Clone`. This means we cannot store a `DMFTConfig` directly in the Python class and clone it when constructing `DMFTLoop`.

**Resolution**: `PyDmrgConfig` maintains its own `Clone` mirror of all fields. `to_rust_config()` reconstructs the Rust config on demand, including re-creating the eigensolver box from a string name.

**Spec feedback**: The spec mentions `to_rust_config()` but doesn't explain *why* it's needed. The non-Clone nature of `DMRGConfig` should be called out explicitly as a design constraint.

### 4. `AndersonImpurityModel::new()` takes 5 args, not 4

The spec suggests `AIM::new(u, eps, None, BathParameters::uniform(...))`, but the actual API is `AIM::new(u, epsilon_imp, n_bath, bandwidth, v0)` — five positional scalar arguments.

**Spec feedback**: The constructor signature should be listed verbatim.

### 5. `LinearPredictionConfig` field names differ from spec

- Spec says `lp_order` → actual name is `prediction_order`
- Spec says `solver` → actual name is `toeplitz_solver`
- `ToeplitzSolver` variants are struct variants (`LevinsonDurbin { tikhonov_lambda }`) not unit variants

**Spec feedback**: Field names should be verified against the actual upstream code.

### 6. `DeviceAPI` has no `Default` impl

The spec assumes `DefaultDevice::default()` works. In reality, `DeviceAPI::new(DeviceFaer, DeviceFaer)` must be called explicitly.

**Spec feedback**: Document that `DefaultDevice` is a type alias, not a type with `Default`.

### 7. `dispatch_variant_mut!` borrow semantics

The mutable dispatch macro must use `match $inner` (not `match &mut $inner`) because the caller already passes a `&mut` reference. Using `&mut` in the macro causes a double-borrow error.

### 8. Nested config getter returns clone, not reference

PyO3 `#[getter]` cannot return `&T` for `#[pyclass]` types. `config.dmrg` returns a clone of `PyDmrgConfig`. This means:
```python
config.dmrg.max_bond_dim = 300  # This modifies a TEMPORARY COPY, not config.dmrg
```

The spec doesn't mention this PyO3 limitation. A future improvement would be to use `Py<PyDmrgConfig>` stored in the parent, or to provide `set_dmrg()` accepting keyword arguments.

## Design Decisions

### GIL deadlock prevention

The critical invariant is: `monitor.shutdown()` must be called inside `py.allow_threads()`, before the GIL is re-acquired. The sequence is:

```
py.allow_threads(|| {
    let result = solver.solve_with_cancel_flag(&flag);
    monitor.shutdown();  // <-- MUST be here, not after allow_threads
    result
})
```

If `shutdown()` runs after `allow_threads` returns, the main thread holds the GIL while `monitor.shutdown()` tries to join the monitor thread, which may be blocked on `Python::with_gil` — classic AB/BA deadlock.

### Error bridge: Cancelled → KeyboardInterrupt

`DmftError::Cancelled` is intercepted at the call site in `solve.rs` and converted to `PyKeyboardInterrupt`. It must NOT flow through `PythonError`, which has an `unreachable!` arm for the `Cancelled` variant. This ensures the cancellation path is always handled explicitly.

### String-based enum dispatch for Python

Rust enums like `SpectralSolverMode`, `MixingScheme`, and `ToeplitzSolver` are exposed to Python as strings ("tdvp", "chebyshev", "adaptive", etc.). This is simpler than creating `#[pyclass]` wrappers for each enum, though it loses type safety.

## Not Implemented / Deferred

1. **TRIQS integration** — feature-gated (`triqs`) but no implementation. Requires runtime Python object inspection and zero-copy via `PyReadonlyArray3`.

2. **Custom exception hierarchy** — `register_exceptions()` is a placeholder. Production code should use `create_exception!` macros for proper `TensorkraftError` → `DmftConvergenceError`, `BathDiscretizationError`, etc.

3. **Zero-copy NumPy interop** — Currently all getters clone data into new NumPy arrays. `numpy_interop.rs` is a placeholder for future `PyArray::from_owned_array` or `PyBuffer` approaches.

4. **SU(2) symmetry variant** — Feature-gated but not wired up. Would need a new `DmftLoopVariant` arm.

5. **Proper `__init__.pyi` stubs** — Not generated. Would improve IDE autocompletion.

6. **Nested config setters** — `config.dmrg = ...` works, but `config.dmrg.max_bond_dim = 300` silently modifies a copy due to PyO3's getter semantics.

## Lessons Learned

1. **Always verify upstream API signatures before writing bindings.** Constructor argument counts, field names, and trait bounds all diverged from the spec. Reading the actual source first would have saved multiple rounds of fixes.

2. **PyO3 `extension-module` is incompatible with `cargo test`.** This is a well-known issue but specs should document it. The standard pattern is a feature flag plus conditional enablement in pyproject.toml.

3. **Non-Clone types propagate through the entire binding layer.** One `Box<dyn Trait>` field in `DMRGConfig` forces the entire config chain to use mirror-and-reconstruct. This should be flagged at the Rust API design stage.

4. **PyO3 getter/setter semantics for nested `#[pyclass]` objects are surprising.** Returning a clone from a getter means mutations to the returned object are lost. This is a fundamental PyO3 limitation that the spec should address.

5. **`DeviceFaer`'s trait bounds gate which `DmftLoopVariant` arms are possible.** The binding layer cannot work around missing trait implementations in upstream crates. The spec should list which combinations are actually compilable.
