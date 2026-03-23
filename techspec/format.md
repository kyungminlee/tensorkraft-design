# Tech Spec Format and Content Requirements

This document defines the canonical formatting and content requirements for all technical specification documents in the `techspec/` directory. All tech specs must follow this structure to ensure consistency and navigability across the tensorkraft workspace.

---

## File Naming Convention

Tech spec files are named with the pattern:

```
{N}_tech-spec_{crate-name}.md
```

where `{N}` is the crate's position in the dependency graph (leaf = 1, topmost = highest number), and `{crate-name}` matches the Cargo crate name (e.g., `tk-core`, `tk-symmetry`).

---

## Document Structure

Every tech spec document must contain the following sections in the order listed below. Sections marked **(required)** must always be present. Sections marked **(conditional)** are included only when the crate has relevant content for that section.

### Header Block (required)

```markdown
# Technical Specification: `{crate-name}`

**Crate:** `tensorkraft/crates/{crate-name}`
**Version:** {version} ({status-qualifier})
**Status:** {Specification | Draft | Approved | Implemented}
**Last Updated:** {Month Year}

---
```

- **Version** includes a parenthetical qualifier such as `(Pre-Implementation)`.
- **Last Updated** uses `{Month} {Year}` format (e.g., `March 2026`).

---

### Section Ordering

The following table defines the canonical section order. Section numbers are assigned sequentially starting from 1, skipping any omitted conditional sections.

| Order | Section Title                              | Status      | Description |
|:------|:-------------------------------------------|:------------|:------------|
| 1     | Overview                                   | Required    | Crate purpose, core responsibilities, position in dependency graph, key dependencies |
| 2     | Module Structure                           | Required    | Directory/module tree as a code block showing the crate's internal file layout |
| 3+    | Core Technical Content                     | Required    | One or more sections detailing the crate's primary types, traits, algorithms, and APIs (see [Core Content Sections](#core-content-sections) below) |
| N-9   | Error Handling                             | Required    | Error enum definition, `Result` type alias, error propagation strategy |
| N-8   | Public API Surface                         | Required    | `lib.rs` re-exports showing the crate's public interface |
| N-7   | Feature Flags                              | Required    | Table of feature flags and their effects within this crate |
| N-6   | Build-Level Concerns                       | Conditional | Compile-time checks, mutual exclusivity enforcement, `build.rs` logic |
| N-5   | Internal Helpers                           | Conditional | Internal (`pub(crate)`) utility functions not part of the public API |
| N-4   | Data Structures and Internal Representations | Conditional | Memory layouts, tensor leg conventions, checkpoint formats, and other internal data organization |
| N-3   | Dependencies and Integration               | Required    | `Cargo.toml` sketch, upstream/downstream crate relationships, external dependency listing |
| N-2   | Testing Strategy                           | Required    | Unit tests, property-based tests, compile-fail tests, integration tests, benchmarks |
| N-1   | Performance Invariants                     | Conditional | Table of operations with their performance guarantees, validated by CI benchmarks |
| N     | Implementation Notes and Design Decisions  | Conditional | Numbered notes explaining non-obvious design choices with traceability to the design document |
| N+1   | Security Considerations                    | Conditional | Security-relevant concerns (e.g., `unsafe` usage, FFI boundaries, user-supplied input handling) |
| N+2   | Out of Scope                               | Required    | Bullet list of what this crate explicitly does **not** implement, with arrows indicating which crate owns each responsibility |
| N+3   | Open Questions                             | Required    | Table of unresolved design questions with status |
| N+4   | Future Considerations                      | Conditional | Forward-looking items planned for later phases that are not yet specified |

> **Note:** "N-9" through "N+4" are relative labels to indicate ordering — actual section numbers are assigned sequentially (e.g., if core content spans sections 3 through 8, then Error Handling is section 9, Public API Surface is section 10, and so on).

---

### Core Content Sections

The core technical content (sections 3 through the section before Error Handling) varies per crate and constitutes the bulk of the specification. These sections document:

- **Type and trait definitions** — Full Rust struct/enum/trait signatures in fenced code blocks
- **Method signatures and contracts** — Public methods with parameter semantics, return types, and invariants
- **Algorithm descriptions** — Step-by-step algorithmic procedures with complexity analysis
- **Configuration types** — Builder patterns, config structs, and their field semantics
- **Phase-gated extensions** — Future capabilities gated behind feature flags, with forward-compatibility notes

#### Formatting rules for core content

1. **Subsection hierarchy**: Use `### N.M` for subsections and `#### N.M.K` for sub-subsections. Do not nest deeper than three levels.
2. **Code blocks**: Use fenced Rust code blocks (` ```rust `) for all type definitions, method signatures, and implementation sketches. Include `#[derive(...)]` attributes and `#[cfg(...)]` guards as they would appear in the actual source.
3. **Feature-gated code**: Precede feature-gated blocks with `#[cfg(feature = "...")]` annotations. Group all feature-gated items for the same feature together when possible.
4. **Phase annotations**: When a section describes functionality deferred to a future phase, state the target phase explicitly (e.g., "Phase 5", "Phase 5+") and note any forward-compatibility provisions in the current design.

---

## Formatting Rules

### General

- Use `---` (horizontal rule) to separate top-level sections (between `## N` headings).
- Number all top-level sections sequentially starting from 1.
- Use ATX-style headers only (`#`, `##`, `###`, `####`).
- All section titles use **Title Case**.
- Backtick-wrap all code identifiers inline: type names, function names, crate names, feature flags, file paths.

### Tables

Tables are used in the following sections with these column formats:

**Feature Flags:**
```markdown
| Flag | Effect in {crate-name} |
|:-----|:-----------------------|
| `flag-name` | Description of what this flag enables or changes |
```

**Unit Tests:**
```markdown
| Test | Description |
|:-----|:------------|
| `test_name` | What the test verifies |
```

**Performance Invariants:**
```markdown
| Operation | Invariant |
|:----------|:----------|
| `operation_name` | Performance guarantee (e.g., zero allocations, O(log N), < 10 ns) |
```

**Open Questions:**
```markdown
| # | Question | Status |
|:--|:---------|:-------|
| 1 | Question text | Open / Deferred / Resolved |
```

All table columns use left-alignment (`:-----`).

### Code Blocks

- Use ` ```rust ` for Rust source code (struct definitions, trait definitions, impl blocks, test examples).
- Use ` ```toml ` for `Cargo.toml` excerpts.
- Use bare ` ``` ` (no language tag) for directory trees and non-code text blocks.
- Include doc comments (`///`) on public types and methods in code blocks when they clarify intent.
- Include `#[cfg(...)]` and `#[derive(...)]` attributes as they would appear in actual source.

### Cross-References

- Reference the design/architecture document as "design document" or "architecture document" followed by the section (e.g., "design document \S3.2").
- Reference other tech specs by crate name (e.g., "see `tk-linalg` tech spec \S7").
- Reference sections within the same document by section number (e.g., "see \S5.2").

---

## Section Content Requirements

### 1. Overview (required)

Must include:
- One-sentence summary of the crate's purpose.
- **Core responsibilities** as a bulleted or bold-prefixed list.
- Position in the dependency graph (which crates depend on it, which crates it depends on).
- **Dependencies** listing (direct crate and external dependencies with version/purpose).
- Scope boundary statement: what this crate does **not** do (brief; detailed version goes in Out of Scope).

### 2. Module Structure (required)

Must include:
- A directory tree in a plain code block showing the `src/` layout:
  ```
  {crate-name}/
    src/
      lib.rs
      module_a.rs
      module_b/
        mod.rs
        sub.rs
    Cargo.toml
  ```
- One-line description of each module's responsibility after or below the tree.

### Error Handling (required)

Must include:
- **Error enum definition** — Full `#[derive(Debug, thiserror::Error)]` enum with all variants and `#[error("...")]` messages.
- **Result type alias** — e.g., `pub type FooResult<T> = Result<T, FooError>;`
- **Error propagation strategy** — How errors flow across crate boundaries (e.g., `#[from]` conversions from upstream error types).

### Public API Surface (required)

Must include:
- The complete `lib.rs` file content showing:
  - `pub mod` declarations
  - `pub use` re-exports for ergonomic downstream usage
  - `#[cfg(feature = "...")]` conditional exports
- Comments indicating the purpose of each group of re-exports.

### Feature Flags (required)

Must include:
- A table listing every feature flag relevant to this crate.
- For each flag: the flag name and its effect **within this specific crate** (not upstream effects).
- Note any flags that are pass-through to upstream crates.

### Dependencies and Integration (required)

Must include:
- **`Cargo.toml` sketch** — A representative `[dependencies]`, `[dev-dependencies]`, and `[features]` section.
- **Upstream dependencies** — Which workspace crates this crate depends on and what it uses from them.
- **Downstream consumers** — Which workspace crates depend on this crate.

May include:
- **External dependencies by functionality** — Grouped listing of third-party crates by purpose.
- **Integration points** — How this crate connects to adjacent crates (e.g., compilation pipeline from `tk-dsl` to `tk-dmrg`).

### Testing Strategy (required)

Must include:

- **Unit Tests** — Table of test cases with names and descriptions. Each test name should be descriptive enough to understand without reading the implementation.
- **Property-Based Tests** — `proptest` examples with bounded strategies (never unbounded random dimensions). Show the `proptest!` macro usage with representative test cases.

May include (as applicable):

- **Compile-Fail Tests** — `trybuild` test cases verifying borrow-checker or type-system enforcement.
- **Integration Tests** — Cross-crate or end-to-end test scenarios.
- **Snapshot/Reference Tests** — Tests comparing against known-correct reference values.
- **Cross-Backend Equivalence Tests** — Tests verifying identical results across different backend implementations.
- **Performance Benchmarks** — Criterion/iai benchmark descriptions with CI gate thresholds.
- **Invariant Checks** — Debug-mode `assert_invariants()` methods called at construction and mutation boundaries.

### Performance Invariants (conditional)

Must include (when present):
- A table of operations with their performance guarantees.
- Each invariant must be verifiable by CI benchmarks (Criterion with instruction-counting mode or wall-clock thresholds).
- State the benchmark tool and mode used for validation.

### Implementation Notes and Design Decisions (conditional)

Must include (when present):
- Numbered notes (e.g., `### Note 1 — Title`), each explaining a non-obvious design decision.
- Each note should be self-contained: state the decision, the alternatives considered, and why this choice was made.
- Include traceability to the design/architecture document where applicable (e.g., "This decision is specified in design doc \S8.2").

### Out of Scope (required)

Must include:
- A bulleted list of functionalities explicitly **not** implemented in this crate.
- Each bullet must indicate the responsible crate with an arrow notation: `(-> crate-name)`.
- Items deferred to future phases should note the phase: `(-> Phase 5+)`.

### Open Questions (required)

Must include:
- A table with columns: `#`, `Question`, `Status`.
- Status values: `Open`, `Deferred`, `Resolved` (optionally with brief rationale or target phase).
- Questions should be specific and actionable, not vague.

### Future Considerations (conditional)

Must include (when present):
- Bulleted list of forward-looking items with brief descriptions.
- Each item should indicate the target phase or condition under which it becomes relevant.
- This section is for items that are **not yet specified** but are anticipated for future phases.

---

## Style Guide

### Language

- Use present tense for specifications ("The trait defines...", "The method returns...").
- Use imperative mood for requirements ("Must validate...", "The caller must ensure...").
- Use passive voice sparingly; prefer active constructions.
- Avoid first person ("I", "we") — use the crate name or component name as the subject.

### Specificity

- Every public type must have a full Rust signature in a code block.
- Every public method must document its parameters, return type, and error conditions.
- Algorithmic descriptions must include complexity (time and space) where non-trivial.
- Numerical thresholds and magic numbers must be justified (e.g., "16x16 tile size balances L1 cache capacity").

### Traceability

- Design decisions must reference the architecture/design document section that motivates them.
- Phase-gated features must state the target phase explicitly.
- Cross-crate dependencies must reference the specific types or traits consumed.
