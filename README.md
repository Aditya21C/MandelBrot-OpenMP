# Parallel Mandelbrot Renderer (C + OpenMP)

A Mandelbrot set renderer in C, parallelized with OpenMP. It includes a sequential baseline, a multithreaded version, and a benchmark tool that times both and checks that their output is pixel-for-pixel identical.

<!-- Add a rendered image here, e.g. ![Seahorse Valley](images/seahorse.png) -->

## Highlights

- **Multithreaded rendering** with `#pragma omp parallel for`, one image row per task.
- **Dynamic load balancing.** Rows near the set's boundary need far more iterations than rows outside it, so rows are handed out with `schedule(dynamic, 1)` instead of being split evenly up front.
- **Built-in correctness check.** The benchmark renders the same frame serially and in parallel, then compares the two images pixel by pixel using OpenMP reductions.
- **Smooth colouring** using a normalized iteration count and a six-stop gradient palette.
- **No external dependencies** apart from OpenMP. PNG output uses the single-header [stb](https://github.com/nothings/stb) libraries.

## Files

| File | Purpose |
|------|---------|
| `mandelbrot_seq.c` | Sequential baseline renderer |
| `mandelbrot_par.c` | OpenMP parallel renderer (schedule configurable at runtime) |
| `mandelbrot_compare.c` | Benchmark: serial vs parallel timing, speedup and pixel-diff check |
| `seq_to_par.diff` | The exact changes needed to parallelize the sequential version |
| `stb_image_write.h`, `stb_image.h` | Public-domain PNG writer and reader by Sean Barrett |

## Build

Requires GCC (or Clang with OpenMP support).

```bash
gcc -O2 -o mandelbrot_seq mandelbrot_seq.c -lm
gcc -O2 -fopenmp -o mandelbrot_par mandelbrot_par.c -lm
gcc -O2 -fopenmp -o mandelbrot_compare mandelbrot_compare.c -lm
```

On macOS with Apple Clang, install `libomp` (e.g. `brew install libomp`) or use Homebrew GCC.

## Usage

Each program accepts arguments or, if run without them, prompts for input.

```bash
# Sequential or parallel render
./mandelbrot_seq <width> <height> <view> [output.png]
./mandelbrot_par <width> <height> <view> [output.png]

# Benchmark and correctness check
./mandelbrot_compare <width> <height> <view> <threads> [output_prefix]
```

**Views:** `1` Full View, `2` Seahorse Valley, `3` Elephant Valley

**Examples:**

```bash
./mandelbrot_par 1920 1080 2 seahorse.png

# Try other scheduling strategies (default is dynamic,1)
OMP_SCHEDULE="static" ./mandelbrot_par 1920 1080 1
OMP_SCHEDULE="guided" OMP_NUM_THREADS=4 ./mandelbrot_par 1920 1080 1

# Compare serial vs parallel on 8 threads
./mandelbrot_compare 1920 1080 2 8 bench
```

`mandelbrot_compare` writes three images (`<prefix>_seq.png`, `<prefix>_par.png`, and `<prefix>_diff.png`, a mask of any differing pixels) and prints a report like:

```
Seahorse Valley | 1920x1080 | 5000 iterations | 2 threads | schedule dynamic,1

               Compute (ms)       PNG (ms)     Total (ms)
Serial             15923.16         284.10       16207.26
Parallel            7966.22         275.46        8241.68

Speedup (compute): 2.00x
Speedup (end to end): 1.97x
Differing pixels: 0 of 2073600 (0.0000%)
Result: PASS, images are identical
```

The program exits with code `0` on PASS and `2` if the images differ, so it can be used in scripts or CI.

## Results

1920x1080, 5,000 max iterations, `schedule(dynamic, 1)`:

| View | Threads | Serial compute | Parallel compute | Speedup | Pixel check |
|------|---------|----------------|------------------|---------|-------------|
| Full View | 2 | 7.82 s | 3.95 s | 1.98x | PASS (0 differing) |
| Seahorse Valley | 2 | 15.92 s | 7.97 s | 2.00x | PASS (0 differing) |

Near-linear scaling (about 99% parallel efficiency). Each pixel is computed independently and rows are balanced dynamically, so threads rarely sit idle.

<!-- Replace or extend this table with runs on your own machine, e.g. 4 and 8 threads. -->

## How it works

1. **Mapping pixels to the complex plane.** Each view has a centre `(cx, cy)` and a width `span`. Pixel `(px, py)` maps to `c = (x0 + px*scale) + i(y0 - py*scale)`.
2. **Escape-time iteration.** For each `c`, iterate `z = z² + c` from `z = 0` until `|z|² ≥ 4` or 5,000 iterations. The loop keeps `x²` and `y²` from the previous step to save multiplications.
3. **Colouring.** Points that never escape are black. Others use a smooth iteration count, `μ = n + 1 − log₂(log|z|)`, mapped onto a gradient palette so there's no visible banding.
4. **Parallelism.** Rows are independent and each writes only to its own part of the image buffer, so no locks are needed. Only the outer row loop is parallelized. This keeps each task large enough to be worth scheduling while still balancing load.
5. **Verification.** The compare tool reads both PNGs back, using two threads via `omp parallel sections`. A parallel loop then computes the number of differing pixels, the maximum channel difference and the first mismatch, using `reduction(+)`, `reduction(max)` and `reduction(min)`.
