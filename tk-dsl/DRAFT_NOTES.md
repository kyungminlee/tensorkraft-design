# tk-dsl Draft Implementation Notes

**Status:** Draft implementation — compiles, 41 tests pass, core abstractions functional.
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

### 1. ~~`DenseTensor` does not implement `Clone`~~ **MITIGATED**

`DenseTensor` in `tk-core` still does not derive `Clone` (due to `TensorStorage` borrowed variant).

**Resolution:** `IndexedTensor<T>` now implements `Clone` directly via a manual impl that copies through `DenseTensor::from_vec(shape.clone(), slice.to_vec())`. `CustomOp<T>` already had a manual `Clone` impl. The old `clone_owned()` method is deprecated in favor of `.clone()`.

**Remaining:** `DenseTensor` itself still lacks `Clone` in `tk-core`. This could be addressed upstream if needed.

### 2. ~~`Scalar` trait lacks an imaginary unit constructor~~ **FIXED**

**Resolution:** Added `fn from_real_imag(re: Self::Real, im: Self::Real) -> Self` to the `Scalar` trait in `tk-core`. For real types (`f32`, `f64`), the imaginary part is ignored. For complex types, it constructs `re + im*i`. `SpinOp::Sy.matrix::<T>()` now correctly produces `[0, -i/2; i/2, 0]` for complex types and `[0, 0; 0, 0]` for real types (which is the correct lossy projection).

### 3. ~~Operator overloading is restricted to `f64` due to Rust's orphan rules~~ **MITIGATED**

**Resolution:** Added `scale(coeff: T) -> ScaledOpProduct<T>` methods to `OpTerm<T>` and `OpProduct<T>` that work for any `T: Scalar`. The `f64`-only `Mul` overloads remain for ergonomics with real-valued models. For complex-valued models, use `op(Sz, 0).scale(coeff)` or `(op(Sz, 0) * op(Sz, 1)).scale(coeff)`. Also made `OpSum * T` generic over `T: Scalar` instead of `f64`-only.

**Remaining limitation:** `coeff * op(Sz, 0)` syntax still only works for `f64` due to orphan rules. This is inherent to Rust and cannot be fixed without a newtype wrapper.

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

### 6. ~~`tk-core` parallel feature flag doesn't exist~~ **FIXED**

**Resolution:** Removed the no-op `parallel` feature and removed it from `default` features in `tk-dsl/Cargo.toml`. No parallelism is needed in the DSL layer.

### 7. `SmallString` from `smallstr` requires separate crate dependency

The spec uses `SmallString<[u8; 32]>` for tags and operator names. This requires the `smallstr` crate, which is listed in the spec's dependencies but is an additional dependency not present in any other tensorkraft crate. It works well but adds to the dependency tree.

**Assessment:** `SmallString` avoids heap allocation for the common case (tags < 32 bytes). Worth the dependency.

### 8. ~~`AddAssign<ScaledOpProduct>` for `OpSum` uses split debug/release behavior~~ **FIXED**

**Resolution:** `AddAssign` now always panics on out-of-bounds site indices, regardless of build profile. This is consistent behavior — bugs are never silently hidden in release builds. Users who want fallible insertion should use `push_term()` which returns `DslResult`.

---

## Gaps Between Spec and Reality

| Spec Section | Issue | Severity | Status |
|:-------------|:------|:---------|:-------|
| §5 (operators) | `SpinOp::Sy` cannot produce correct imaginary matrix via `Scalar` trait | High | **FIXED** — `from_real_imag` added to `Scalar` |
| §5 (operators) | Operator overloading limited to `f64` due to orphan rules | Medium | **MITIGATED** — `scale()` methods added |
| §6 (IndexedTensor) | `DenseTensor` not `Clone`; `IndexedTensor` can't derive `Clone` | Medium | **MITIGATED** — manual `Clone` impl added |
| §6 (IndexedTensor) | `contract()` uses naive GEMM, not `tk-linalg` backend | Low | Accepted (small matrices) |
| §8 (OpSum) | `Lattice` cloneability requires undocumented `LatticeClone` helper trait | Low | Accepted (standard pattern) |
| §8 (OpSum) | `AddAssign` split debug/release behavior | Low | **FIXED** — always panics |
| §14 (Cargo.toml) | `tk-core/parallel` feature doesn't exist | Low | **FIXED** — removed |
| §11 (proc-macro) | `tk-dsl-macros` not implemented in this draft | N/A | Deferred |

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
