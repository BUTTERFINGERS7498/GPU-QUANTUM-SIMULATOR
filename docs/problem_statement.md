# Problem Statement: Fit More Qubits in Less Memory

**Category:** Non-AI/ML (high-performance computing, numerical methods, quantum simulation)
**Format:** Hackathon, team of 1-4 (duration and level: see "Assumptions" at the end)

## 1. Background

A quantum computer with n qubits has a state of 2^n complex numbers. A classical computer that simulates it must store all of them. Memory doubles with each added qubit.

With 64-bit complex numbers (16 bytes each), the memory is:

| Qubits (n) | State vector size |
|-----------:|------------------:|
| 20 | 16 MiB |
| 26 | 1 GiB |
| 28 | 4 GiB |
| 30 | 16 GiB |

A typical laptop has 6 GiB of GPU memory and 8 GB of RAM. Such a laptop reaches about 28 qubits with a plain state-vector simulator. Real research on quantum chemistry and error correction needs more.

The reference setup in this repository measures the limit:

- Simulator: NVIDIA CUDA-Q 0.13.0, targets `qpp-cpu` (CPU, fp64) and `nvidia` (GPU, fp32).
- Hardware: RTX 4050 Laptop GPU (6141 MiB), WSL2 with 7 GB RAM.
- Measured: `qpp-cpu` on a GHZ circuit takes a median of 119.6 s at n=26 and uses 1.00 GiB.

## 2. The Problem

Build a quantum circuit simulator, or an extension to the reference simulator, that **runs larger circuits on the same laptop** or **runs the same circuits faster and in less memory**. Do not use machine learning.

You choose one track.

### Track A: Memory
Reach a higher qubit count than the reference setup for the benchmark circuits in section 3. Examples of methods:

- Reduced-precision storage (fp32, fp16 or block floating point) with a proven error bound.
- Chunked state vectors that move parts between disk, CPU RAM and GPU memory.
- Compressed state storage (for example, exploit the structure of GHZ-type states).

### Track B: Speed
Keep the same n as the reference setup and reduce the run time. Examples of methods:

- Gate fusion (merge adjacent gates into one matrix).
- Better qubit ordering and memory access patterns.
- Better use of the GPU (batching, streams, kernel tuning).

### Track C: Structure
Use a different simulation method (for example, tensor networks or stabilizer simulation) for circuits where it is exact and cheaper than the state vector. The solution must detect automatically when the method applies, and fall back to the state vector when it does not.

## 3. Benchmark Circuits

Each team is scored on these circuits.

1. **GHZ state, n = 4 to the maximum that the team reaches.** Correct output: only `00...0` and `11...1`, each close to 50% of the shots. The reference solution checks this at n=8 with 4000 shots.
2. **H2 molecule VQE.** The Hamiltonian has 15 Pauli terms. The exact (FCI) energy is the reference. The reference prototype reaches an error of 0.0028 mHa. A solution is correct when its error is at most 1.6 mHa (chemical accuracy).

The organizers add one hidden circuit on the day of the event, to prevent a solution that fits only the two known circuits.

## 4. Rules and Constraints

1. Hardware: one laptop-class machine with at most 6 GiB of GPU memory and 8 GB of RAM. No cloud GPU for the scored runs.
2. Language: Python for the interface. Native extensions (C, C++, CUDA) are allowed.
3. No machine learning, no neural networks and no trained models in the solution.
4. Results must be reproducible. The run must start from one command (for example, `make benchmark`) and use pinned dependencies (`requirements.txt`).
5. Report the run time as the median of at least 5 runs, with the standard deviation.
6. Do not hard-code the answers of the benchmark circuits.

## 5. Deliverables

1. Source code in a repository, with a README that has the run commands.
2. A results table: n, run time (median and standard deviation), peak memory, correctness check.
3. One plot: run time against n for the reference setup and for the team solution.
4. A one-page description of the method, with the main trade-off (for example, speed against accuracy).
5. A 5-minute demonstration.

## 6. Scoring (100 points)

| Criterion | Points | How it is measured |
|-----------|-------:|--------------------|
| Correctness | 30 | All benchmark circuits give correct output, including the hidden circuit |
| Improvement | 30 | Track A: extra qubits. Track B: speedup. Track C: number of circuit types handled exactly |
| Rigor | 15 | Error bounds, repeated measurements, honest limits |
| Reproducibility | 10 | A clean machine runs the result with one command |
| Code quality and tests | 10 | Readable code, tests that pass, coverage of at least 80% |
| Presentation | 5 | Clarity of the demonstration and the one-page description |

Tie-breaker: lower peak memory.

## 7. Starter Kit

The repository provides:

- `src/hardware.py`: hardware detection and target selection.
- `src/ghz_benchmark.py` and `src/ghz_plots.py`: GHZ benchmark and plots.
- `src/h2_vqe.py` and `src/h2_plots.py`: H2 VQE and plots.
- `tests/`: unit tests for the reference code.
- `Makefile` and `requirements.txt`: build and run commands.
- `notebooks/colab_quickstart.ipynb`: a quick start for teams without a local GPU.

## Assumptions

I did not know the hackathon duration or the participant level. This draft assumes a 24-hour event with student teams. If the duration is shorter, keep one track and remove the hidden circuit. If the participants are beginners, remove Track C.
