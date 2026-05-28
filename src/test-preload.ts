/**
 * Test preload file to ensure proper cleanup of native resources.
 *
 * Uses bun:test afterAll to properly dispose of llama.cpp Metal
 * resources before the process exits, avoiding GGML_ASSERT failures.
 */

// Mirror bin/qmd's darwin Metal residency mitigation so `bun test` and
// `vitest` runs exit cleanly. The test runners load node-llama-cpp directly
// without going through the launcher, so the libggml-metal static destructor
// asserts on a non-empty residency set during __cxa_finalize_ranges and dumps
// a multi-kB backtrace at process exit (ggml-org/llama.cpp#17869). Opt back
// in with QMD_METAL_KEEP_RESIDENCY=1 if you're triaging the upstream Metal
// teardown bug.
//
// Two-step propagation, because:
//   - Native code in libggml-metal reads via C getenv() at module load time.
//   - Node syncs process.env mutations to libc via uv_os_setenv automatically.
//   - Bun does NOT — `process.env.X = "1"` only updates the JS-level object,
//     so getenv() in the C++ binding still sees nothing (verified empirically
//     with bun:ffi). We have to call setenv() ourselves on Bun.
if (process.platform === "darwin" && process.env.QMD_METAL_KEEP_RESIDENCY !== "1") {
  process.env.GGML_METAL_NO_RESIDENCY = process.env.GGML_METAL_NO_RESIDENCY || "1";

  if (typeof (globalThis as { Bun?: unknown }).Bun !== "undefined") {
    try {
      const { dlopen, FFIType, suffix } = await import("bun:ffi");
      const libc = dlopen(`libSystem.${suffix}`, {
        setenv: { args: [FFIType.cstring, FFIType.cstring, FFIType.i32], returns: FFIType.i32 },
      });
      libc.symbols.setenv(
        Buffer.from("GGML_METAL_NO_RESIDENCY\0", "utf8"),
        Buffer.from("1\0", "utf8"),
        1,
      );
    } catch {
      // FFI unavailable on this Bun build — the backtrace dump at exit is
      // cosmetic; tests still pass.
    }
  }
}

import { afterAll } from "bun:test";
import { disposeDefaultLlamaCpp } from "./llm";

// Global afterAll runs after all test files complete
afterAll(async () => {
  await disposeDefaultLlamaCpp();
});
