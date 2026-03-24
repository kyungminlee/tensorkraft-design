# tk-python Draft Implementation Notes

**Status:** `cargo check` passes. 11/11 tests pass (requires `--no-default-features --features backend-faer,parallel`). `maturin build` not tested.
**Date:** March 2026

---

## What is implemented

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

### GIL deadlock prevention (complete)

`monitor.shutdown()` is called inside `py.allow_threads()`, before the GIL is re-acquired. Prevents AB/BA deadlock between main thread (GIL) and monitor thread (`Python::with_gil`).

### Error bridge (complete)

`DmftError::Cancelled` → `PyKeyboardInterrupt`. Intercepted at the call site in `solve.rs`, never flows through `PythonError` (which has `unreachable!` for Cancelled).

### String-based enum dispatch (complete)

Rust enums (`SpectralSolverMode`, `MixingScheme`, `ToeplitzSolver`) exposed to Python as strings ("tdvp", "chebyshev", "adaptive", etc.).

### Config mirror pattern (complete)

`PyDmrgConfig` maintains its own `Clone` mirror of all fields. `to_rust_config()` reconstructs the Rust config on demand, including re-creating the eigensolver box from a string name. Required because `DMRGConfig` contains `Box<dyn IterativeEigensolver<f64>>` which is not `Clone`.

---

## Remaining limitations

### Not implemented / deferred

1. **TRIQS integration** — feature-gated (`triqs`) but no implementation. Requires runtime Python object inspection and zero-copy via `PyReadonlyArray3`.
2. **Custom exception hierarchy** — `register_exceptions()` is a placeholder. Production needs `create_exception!` macros.
3. **Zero-copy NumPy interop** — Currently all getters clone data. `numpy_interop.rs` is a placeholder.
4. **SU(2) symmetry variant** — Feature-gated but not wired up.
5. **`__init__.pyi` type stubs** — Not generated.
6. **Nested config setters** — `config.dmrg.max_bond_dim = 300` silently modifies a copy due to PyO3 getter semantics.

### Spec-vs-reality gaps

| Issue | Resolution |
|:------|:-----------|
| `ComplexU1` variant impossible (DeviceFaer lacks complex LinAlgBackend) | Removed; `complex_u1()` returns RuntimeError |
| PyO3 `extension-module` vs test linking conflict | Moved behind feature flag; tests use `--no-default-features` |
| `DMRGConfig` not Clone | Config mirror pattern with `to_rust_config()` |
| `AndersonImpurityModel::new()` takes 5 args, not 4 | Constructor signature corrected |
| `LinearPredictionConfig` field names differ (`lp_order` → `prediction_order`) | Field names corrected |
| `DeviceAPI` has no `Default` impl | Uses `DeviceAPI::new(DeviceFaer, DeviceFaer)` explicitly |
| Nested config getter returns clone, not reference | PyO3 limitation; documented |

---

## Design decisions

1. **`dispatch_variant_mut!` borrow semantics** — Uses `match $inner` (not `match &mut $inner`) because the caller already passes `&mut`.
2. **Only `RealU1` and `RealZ2` variants** — `ComplexU1` removed as blocked on upstream.
3. **String-based enums for Python** — simpler than `#[pyclass]` wrappers, loses type safety.

---

## Lessons learned

1. Always verify upstream API signatures before writing bindings.
2. PyO3 `extension-module` is incompatible with `cargo test` — document the feature flag pattern.
3. Non-Clone types propagate through the entire binding layer.
4. PyO3 getter/setter semantics for nested `#[pyclass]` objects are surprising (returns clone).
5. `DeviceFaer`'s trait bounds gate which `DmftLoopVariant` arms are possible.
