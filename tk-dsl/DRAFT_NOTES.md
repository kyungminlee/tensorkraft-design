# tk-dsl Draft Implementation Notes

**Status:** Draft implementation — compiles, 49 tests pass (44 unit + 5 proptest), core abstractions functional.
**Date:** March 2026

---

## What is implemented

- **index.rs** — `Index` with `IndexId`, tag, dim, prime_level, direction; `IndexRegistry` with duplicate detection
- **operators.rs** — `SpinOp`, `FermionOp`, `BosonOp`, `CustomOp<T>`, `SiteOperator<T>` with matrix generation and adjoint
- **opterm.rs** — `OpTerm<T>`, `op()` constructor, `OpProduct<T>`, `ScaledOpProduct<T>` with operator overloading
- **opsum.rs** — `OpSum<T>`, `OpSumTerm<T>`, `HermitianConjugate` marker, `hc()`, `OpSumPair<T>`
- **indexed_tensor.rs** — `IndexedTensor<T>` with named-index contraction via `contract()`
- **lattice/** — `Lattice` trait, `Chain`, `Square`, `Triangular`, `BetheLattice`, `StarGeometry`, `snake_path()`
- **error.rs** — `DslError` enum with 8 variants

### Imaginary unit for Scalar (complete)

Added `fn from_real_imag(re: Self::Real, im: Self::Real) -> Self` to the `Scalar` trait in `tk-core`. `SpinOp::Sy.matrix::<T>()` correctly produces `[0, -i/2; i/2, 0]` for complex types and `[0, 0; 0, 0]` for real types (correct lossy projection).

### Operator overloading for all Scalar types (complete)

`Mul` overloads (`coeff * expr` and `expr * coeff`) implemented for all four concrete `Scalar` types (`f32`, `f64`, `Complex<f32>`, `Complex<f64>`) via a `impl_scalar_mul!` macro. A blanket `impl<T: Scalar>` is impossible due to Rust orphan rules. `scale(coeff: T)` methods on `OpTerm<T>` and `OpProduct<T>` remain as an alternative that works for any `T: Scalar`. `OpSum * T` is also generic.

### IndexedTensor Clone (complete)

`IndexedTensor<T>` implements `Clone` via manual impl that copies through `DenseTensor::from_vec(shape.clone(), slice.to_vec())`.

### Consistent AddAssign behavior (complete)

`AddAssign` for `OpSum` always panics on out-of-bounds site indices regardless of build profile. Users who want fallible insertion use `push_term()` which returns `DslResult`.

---

## Remaining limitations

1. **`tk-dsl-macros` proc-macro crate not implemented** — `hamiltonian!{}` macro requires a separate `proc-macro = true` crate with `syn 2.x` / `quote` / `proc_macro2`. The core library types are the foundation; the macro is syntactic sugar.

2. **`Lattice` cloneability requires `LatticeClone` helper trait** — `Box<dyn Lattice>` needs `clone_box()` blanket impl for `Clone`. Standard pattern but undocumented in spec. Consider `Arc<dyn Lattice>` for zero-cost sharing.

3. **`IndexedTensor::contract()` uses naive GEMM** — No `tk-linalg` dependency in `tk-dsl`. O(mnk) triple-loop is acceptable for small operator matrices in the DSL layer.

4. **`SmallString` requires `smallstr` dependency** — Avoids heap allocation for tags < 32 bytes. Worth the dependency.

### Spec-vs-reality gaps

| Spec Section | Issue | Severity | Status |
|:-------------|:------|:---------|:-------|
| §5 (operators) | `SpinOp::Sy` imaginary matrix | High | **Resolved** — `from_real_imag` added |
| §5 (operators) | Operator overloading limited to `f64` | Medium | **Resolved** — `impl_scalar_mul!` for all 4 types |
| §6 (IndexedTensor) | `DenseTensor` not `Clone` | Medium | **Mitigated** — manual `Clone` impl |
| §6 (IndexedTensor) | `contract()` uses naive GEMM | Low | Accepted (small matrices) |
| §8 (OpSum) | `Lattice` cloneability undocumented | Low | Accepted (standard pattern) |
| §11 (proc-macro) | `tk-dsl-macros` not implemented | N/A | Deferred |

---

## Consistency issues: tech spec vs. architecture

| Item | Spec | Architecture | Assessment |
|:-----|:-----|:-------------|:-----------|
| SpinOp variants | `SPlus, SMinus, Sz, Sx, Sy, Identity` | `SPlus, SMinus, Sz, Identity` | Spec is more complete |
| BosonOp variants | `BDag, B, N, NPairInteraction, Identity` | `BDag, B, N, Identity` | Spec adds NPairInteraction |
| Lattice trait | Has `local_dim() -> Option<usize>` | No `local_dim()` | Spec adds useful method |
| `contract()` return | `DslResult<IndexedTensor<T>>` | `IndexedTensor<T>` | Spec is more correct |
| Macro operator names | `SPlus`/`SMinus` | `Sp`/`Sm` | Both valid |
| Boson in SiteOperator | `Boson { op, n_max }` | `Boson(BosonOp)` | Spec correctly carries n_max |

---

## What works well

1. **Typed operator enums** — compile-time operator name checking eliminates runtime mismatches.
2. **`op()` constructor + overloading** — `J * op(Sz, 0) * op(Sz, 1)` reads naturally.
3. **`hc()` marker** — `term + hc()` atomically adds forward and backward terms.
4. **Lattice with cached bonds** — compute once, return `&[(usize, usize)]`.
5. **`snake_path()`** — boustrophedon ordering built-in for 2D→1D DMRG.
6. **Index prime level system** — `prime()`/`unprime()`/`contracts_with()` for bra/ket pairing.
7. **`OpSum::with_lattice()`** — bounds checking at term insertion time.
8. **`CustomOp<T>` escape hatch** — arbitrary operators via same `OpSum` infrastructure.
