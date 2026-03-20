Reviewing a design document for computational physics requires an LLM to look past generic software engineering principles and focus on the unique constraints of numerical simulations, such as mathematical stability, memory bandwidth, and hardware acceleration.

Here is a comprehensive prompt structured to extract a highly technical and domain-specific review. You can copy and paste this directly:

---

**Copy and Paste:**

Act as a Principal Software Engineer and Computational Physicist. I want you to review the following software design document for a computational physics software library.
As you review, please evaluate the architecture and design choices against the following criteria:
**1. Numerical Robustness & Algorithms:**
* Are the chosen numerical methods appropriate for the physical models being simulated?
* Identify potential risks regarding numerical stability, precision loss, or error propagation.
**2. High-Performance Computing & Architecture:**
* Evaluate the parallelization strategy. Are there potential bottlenecks in CPU/GPU workload distribution, synchronization overhead, or thread contention?
* If hardware acceleration (like CUDA) is utilized, assess the efficiency of data transfers between host and device memory.
**3. Memory & Resource Management:**
* Critically analyze the data structures and memory access patterns. Are they optimized for cache locality, especially within tight computational loops?
* Given typical constraints in low-level, high-performance languages (like C++), point out any architectural choices that might lead to memory leaks or inefficient allocations.
**4. Scalability & Extensibility:**
* How well does the design support adding new physical models, boundary conditions, or numerical solvers in the future without major refactoring?

Please format your review with a brief executive summary of the design's strengths, followed by categorized, actionable feedback for each of the areas above. If any part of the design document lacks sufficient detail to make a judgment, explicitly state what information is missing.

---

### Why this structure works:

* **Persona Assignment:** Giving the LLM the dual role of "Software Engineer" and "Computational Physicist" prevents it from focusing entirely on standard web-dev concepts like microservices when you need advice on algorithmic stability.
* **Targeted Hardware Constraints:** By explicitly mentioning cache locality, memory access patterns, and host-to-device transfers, you force the AI to look for the specific performance bottlenecks that plague high-performance simulations.
* **Actionable Formatting:** It ensures the output isn't just a wall of text, but a categorized checklist of concrete issues to address before you start writing code.

Would you like to paste a specific section of your design document here to test-drive this review framework?