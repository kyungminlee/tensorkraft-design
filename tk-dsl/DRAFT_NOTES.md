# tk-dsl Draft Implementation Notes

**Status:** Draft implementation — compiles, 37 tests pass, core abstractions functional.
**Date:** March 2026

---

## What Was Implemented

- **index.rs** — `Index` with `IndexId`, tag, dim, prime_level, direction; `IndexRegistry` with duplicate detection
- **operators.rs** — `SpinOp`, `FermionOp`, `BosonOp`, `CustomOp<T>`, `SiteOperator<T>` with matrix generation and adjoint
- **opterm.rs** — `OpTerm<T>`, `op()` constructor, `OpProduct<T>`, `ScaledOpProduct<T>` with operator overloading
- **opsum.rs** — `OpSum<T>`, `OpSumTerm<T>`, `HermitianConjugate` marker, `hc()`, `OpSumPair<T>`
- **indexed_tensor.rs** — `IndexedTensor<T>` with named-index contraction via `contract()`
- **lattice/** — `Lattice` trait, `Chain`, `Square`, `Triangular`, `BetheLattice`, `StarGeometry`, `snake_path()`
- **error.rs** — `DslError` enum with 8 variants

**Not implemented:** `tk-dsl-macros` proc-macro crate (`hamiltonian!{}`). This requires a separate `proc-macro = true` crate and depends on `syn 2.x` / `quote` / `proc_macro2`. The core library types are the foundation; the macro is syntactic sugar over them.

---

## Design Issues Discovered During Implementation

### 1. `DenseTensor` does not implement `Clone`

The spec's `IndexedTensor<T>` and `CustomOp<T>` both contain `DenseTensor<'static, T>` and are expected to be `Clone`. However, `DenseTensor` in `tk-core` does not derive or implement `Clone` (due to the `TensorStorage` enum with its borrowed variant).

**Workaround used:** Manual `Clone` implementations that copy via `DenseTensor::from_vec(shape.clone(), slice.to_vec())`. For `IndexedTensor`, a `clone_owned()` method was added instead of implementing `Clone` directly, since the derive macro can't work.

**Recommendation:** Either:
- Implement `Clone` for `DenseTensor<'static, T>` (only the owned variant) in `tk-core`
- Use `Arc<[T]>` in `TensorStorage::Owned` so Clone is cheap
- Document that `IndexedTensor` is not `Clone` by design and users should use `clone_owned()`

**Severity:** Medium — affects ergonomics of the entire DSL layer.

### 2. `Scalar` trait lacks an imaginary unit constructor

`SpinOp::Sy` requires the matrix `[0, -i/2; i/2, 0]`. The `Scalar` trait provides `from_real(r)`, `conj()`, `zero()`, `one()`, but no way to construct a purely imaginary value. For `Complex<f64>`, we need `Complex::new(0.0, 0.5)`, but `Scalar` doesn't expose this.

**Workaround used:** `SpinOp::Sy.matrix::<T>()` returns zeros for all types. This is correct for `T = f64` (Sy has no real representation) but wrong for `T = Complex<f64>`.

**Recommendation:** Either:
- Add `fn imaginary_unit() -> Option<Self>` to `Scalar` (returns `None` for real types, `Some(i)` for complex)
- Add `fn from_real_imag(re: Self::Real, im: Self::Real) -> Self` to `Scalar`
- Handle `Sy` specially in the macro/OpSum layer where the concrete type is known

**Severity:** High for complex-valued Hamiltonians (Sy terms silently produce zeros).

### 3. Operator overloading is restricted to `f64` due to Rust's orphan rules

The spec shows generic `T * OpTerm<T>` and `T * OpProduct<T>` operator overloads. In Rust, `impl Mul<OpTerm<T>> for T` requires either `T` or `OpTerm<T>` to be defined in the current crate. Since `T` is a foreign type (f64, Complex<f64>), we can only implement for specific concrete types.

**Workaround used:** Implemented `Mul` for `f64` only. Users with `Complex<f64>` must use `ScaledOpProduct { coeff, product }` explicitly or a helper function.

**Recommendation:** Either:
- Provide a `scaled(coeff, product)` helper function as the primary API
- Use a newtype wrapper `Coeff<T>(T)` that the user constructs: `Coeff(J) * op(Sz, i)`
- Accept the f64-only limitation for operator overloading and document it
- Consider a `scale()` method: `op(Sz, 0).scale(J)` which works for any `T`

**Severity:** Medium — blocks ergonomic complex-valued Hamiltonian construction.

### 4. `Lattice` trait requires `Clone` for `Box<dyn Lattice>` — needs helper trait

`OpSum<T>` stores `Option<Box<dyn Lattice>>` and needs to be `Clone`. But `dyn Lattice` is not `Clone` (clone is not object-safe).

**Workaround used:** Added a `LatticeClone` helper trait with `clone_box()` and a blanket impl, then implemented `Clone for Box<dyn Lattice>`. This pattern is well-known but adds boilerplate not mentioned in the spec.

**Recommendation:** Document this pattern in the spec, or consider using `Arc<dyn Lattice>` instead of `Box<dyn Lattice>` for zero-cost sharing (lattice geometry is immutable).

**Severity:** Low — the pattern works, just undocumented.

### 5. `IndexedTensor::contract()` performs naive GEMM — no backend delegation

The spec says `contract()` should delegate numerical work to `tk-contract` / `tk-linalg`. However, `tk-dsl` deliberately has no `tk-linalg` dependency. The `tk-contract` dependency is listed as "IndexId re-export only."

The draft implementation performs contraction via a naive triple-loop GEMM. This is correct but O(mnk) without BLAS acceleration. For the DSL layer (small operator matrices, not production tensor contractions), this is acceptable.

**Recommendation:** Either:
- Accept naive GEMM for `IndexedTensor::contract()` (it's for small matrices)
- Move `contract()` to `tk-dmrg` where `tk-linalg` is available
- Add an optional `backend` parameter: `contract_with_backend(a, b, &backend)`

**Severity:** Low — `IndexedTensor` contraction is for small matrices in the DSL layer.

### 6. `tk-core` parallel feature flag doesn't exist

The spec lists `parallel = ["tk-core/parallel"]` as a feature flag. However, `tk-core/Cargo.toml` has no `parallel` feature.

**Workaround used:** Changed to `parallel = []` (no-op feature).

**Recommendation:** Either add `parallel` feature to `tk-core` or remove the propagation from `tk-dsl`.

**Severity:** Low — no parallelism needed in the DSL layer.

### 7. `SmallString` from `smallstr` requires separate crate dependency

The spec uses `SmallString<[u8; 32]>` for tags and operator names. This requires the `smallstr` crate, which is listed in the spec's dependencies but is an additional dependency not present in any other tensorkraft crate. It works well but adds to the dependency tree.

**Assessment:** `SmallString` avoids heap allocation for the common case (tags < 32 bytes). Worth the dependency.

### 8. `AddAssign<ScaledOpProduct>` for `OpSum` uses `unwrap_or_default` pattern

The spec says `+=` syntax should not require `Result` handling. The implementation uses `expect()` in debug builds and silent skip in release. This means out-of-bounds site errors are panics in debug and silently ignored in release.

**Recommendation:** Consider always panicking (consistent behavior) or always returning an error. The current split behavior could hide bugs in release builds.

**Severity:** Low — matches the spec's intent but could surprise users.

---

## Gaps Between Spec and Reality

| Spec Section | Issue | Severity |
|:-------------|:------|:---------|
| §5 (operators) | `SpinOp::Sy` cannot produce correct imaginary matrix via `Scalar` trait | High |
| §5 (operators) | Operator overloading limited to `f64` due to orphan rules | Medium |
| §6 (IndexedTensor) | `DenseTensor` not `Clone`; `IndexedTensor` can't derive `Clone` | Medium |
| §6 (IndexedTensor) | `contract()` uses naive GEMM, not `tk-linalg` backend | Low |
| §8 (OpSum) | `Lattice` cloneability requires undocumented `LatticeClone` helper trait | Low |
| §14 (Cargo.toml) | `tk-core/parallel` feature doesn't exist | Low |
| §11 (proc-macro) | `tk-dsl-macros` not implemented in this draft | N/A (deferred) |

---

## Consistency Issues: Tech Spec vs. ARCHITECTURE.md

| Item | Spec | Architecture | Assessment |
|:-----|:-----|:-------------|:-----------|
| SpinOp variants | `SPlus, SMinus, Sz, Sx, Sy, Identity` | `SPlus, SMinus, Sz, Identity` | Spec is more complete |
| BosonOp variants | `BDag, B, N, NPairInteraction, Identity` | `BDag, B, N, Identity` | Spec adds NPairInteraction for Bose-Hubbard |
| Lattice trait | Has `local_dim() -> Option<usize>` | No `local_dim()` method | Spec adds useful optional method |
| Lattice structs | Have `d: usize` field | No `d` field | Spec is more complete |
| `contract()` return | `DslResult<IndexedTensor<T>>` | `IndexedTensor<T>` (no Result) | Spec is more correct |
| Macro operator names | `SPlus`/`SMinus` | `Sp`/`Sm` | Architecture uses shorthand; both are valid |
| Boson in SiteOperator | `Boson { op, n_max }` | `Boson(BosonOp)` | Spec correctly carries n_max at runtime |

---

## What Works Well

1. **Typed operator enums eliminate entire class of bugs** — compile-time operator name checking is the killer feature. No more runtime `"Sz"` vs `"sz"` mismatches.

2. **`op()` constructor + operator overloading is ergonomic** — `J * op(SpinOp::Sz, 0) * op(SpinOp::Sz, 1)` reads naturally and closely matches physics notation.

3. **`hc()` marker for Hermitian conjugate pairs** — `term + hc()` atomically adds both forward and backward terms, preventing the common bug of forgetting the h.c.

4. **Lattice abstraction with cached bonds** — Computing bonds once at construction and returning `&[(usize, usize)]` avoids repeated allocation.

5. **`snake_path()` for 2D→1D mapping** — Boustrophedon ordering is the standard choice for DMRG on 2D systems; having it built-in saves users from implementing it.

6. **Index prime level system** — The `prime()`/`unprime()`/`contracts_with()` pattern cleanly encodes bra/ket pairing without separate bra/ket types.

7. **`OpSum::with_lattice()` for bounds checking** — Catching out-of-bounds site indices at term insertion time prevents downstream errors in MPO compilation.

8. **`CustomOp<T>` escape hatch** — Users can define arbitrary operators while still using the same `OpSum` infrastructure, avoiding the need for a parallel "custom model" API.
