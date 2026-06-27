/**
 * mcp-embed-dimension.test.ts - Regression for the MCP/SDK "Dimension
 * mismatch for query vector" bug.
 *
 * Symptom: the MCP server (and any SDK consumer of the public createStore)
 * threw "Dimension mismatch for query vector ... Expected 1024 ... received
 * 768" when the index was built with a remote embedding model, while the CLI
 * `vsearch` worked fine with the same model.
 *
 * Root cause: src/index.ts:createStore() unconditionally bound a concrete
 * local LlamaCpp to internal.llm. store.ts:getLlm() prefers store.llm over
 * the HybridLLM singleton, and resolveEmbedModelForStore() short-circuits on
 * store.llm.embedModelName. Both forced the LOCAL embeddinggemma model
 * (768-dim) for the query vector, while the index column was built with the
 * REMOTE model (1024-dim). The CLI's getStore() never binds store.llm and
 * routes through getDefaultLLM() (HybridLLM), so it embedded at the remote
 * dimension and matched the index.
 *
 * Fix: createStore() only binds internal.llm when the embed backend is LOCAL.
 * When remote embed is active it leaves internal.llm unset (registering the
 * sanitized local LlamaCpp as the default backend for tokenization) so embed
 * routes through the HybridLLM, matching the remote-built index dimension.
 *
 * These tests assert the routing invariant only — they do not hit the network.
 */

import { describe, test, expect, beforeAll, afterAll, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../src/index.js";
import { setDefaultLLM } from "../src/llm.js";

let testDir: string;

// Snapshot the remote-embed env vars we mutate so each test starts clean and
// we never leak state into sibling suites.
const REMOTE_ENV_KEYS = [
  "QMD_REMOTE_API_KEY",
  "QMD_REMOTE_BASE_URL",
  "QMD_REMOTE_EMBED_MODEL",
  "QMD_EMBED_BACKEND",
  "QMD_GENERATE_BACKEND",
  "QMD_RERANK_BACKEND",
] as const;
const savedEnv: Record<string, string | undefined> = {};

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-mcp-dim-test-"));
  for (const k of REMOTE_ENV_KEYS) savedEnv[k] = process.env[k];
});

afterAll(async () => {
  for (const k of REMOTE_ENV_KEYS) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k]!;
  }
  await rm(testDir, { recursive: true, force: true });
});

afterEach(() => {
  // Clear the default-LLM singleton and the remote env between tests.
  try {
    setDefaultLLM(null);
  } catch {
    // ignore
  }
  for (const k of REMOTE_ENV_KEYS) delete process.env[k];
});

function freshDbPath(): string {
  return join(testDir, `test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
}

describe("createStore embed-model routing (MCP/SDK dimension-mismatch regression)", () => {
  test("remote embed active: does NOT bind a local LlamaCpp to internal.llm", async () => {
    // With a remote embed backend, binding a concrete local LlamaCpp to
    // internal.llm makes getLlm()/resolveEmbedModelForStore() force the local
    // 768-dim model for the query vector — the exact cause of the 1024-vs-768
    // dimension mismatch. The store must instead route through getDefaultLLM()
    // (HybridLLM) so the query embeds at the remote dimension.
    process.env.QMD_REMOTE_API_KEY = "sk-test";
    process.env.QMD_REMOTE_EMBED_MODEL = "perplexity/pplx-embed-v1-0.6b";
    process.env.QMD_EMBED_BACKEND = "remote";
    process.env.QMD_GENERATE_BACKEND = "local";
    process.env.QMD_RERANK_BACKEND = "local";

    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    // The invariant: store.internal.llm must be unset so embed dispatches
    // through the HybridLLM (remote), not a bound local LlamaCpp.
    expect(store.internal.llm).toBeUndefined();
    await store.close();
  });

  test("local embed (no remote key): binds the per-store LlamaCpp as before", async () => {
    // Original behaviour must be preserved when no remote backend is set:
    // a local LlamaCpp is bound so the local embed/index path keeps working.
    delete process.env.QMD_REMOTE_API_KEY;
    delete process.env.QMD_EMBED_BACKEND;
    delete process.env.QMD_REMOTE_EMBED_MODEL;

    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    expect(store.internal.llm).toBeDefined();
    await store.close();
  });

  test("remote key set but QMD_EMBED_BACKEND=local: still binds local LlamaCpp", async () => {
    // A remote API key is present (e.g. for remote generate/rerank) but embed
    // is explicitly local. The embed path is local, so binding the local
    // LlamaCpp is correct — only embed=remote must skip the binding.
    process.env.QMD_REMOTE_API_KEY = "sk-test";
    process.env.QMD_EMBED_BACKEND = "local";
    process.env.QMD_GENERATE_BACKEND = "remote";
    delete process.env.QMD_REMOTE_EMBED_MODEL;

    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    expect(store.internal.llm).toBeDefined();
    await store.close();
  });

  test("searchVector keys sqlite-vec with the REMOTE model name when remote embed is active", async () => {
    // The low-level searchVector() SDK call previously hardcoded the local
    // LlamaCpp's embedModelName. With remote embed active, the sqlite-vec
    // lookup must be keyed by the remote model name (matching the dimension
    // the index column was built with). We assert this by injecting a fake
    // default LLM and capturing the model name passed to internal.searchVec.
    process.env.QMD_REMOTE_API_KEY = "sk-test";
    process.env.QMD_REMOTE_EMBED_MODEL = "perplexity/pplx-embed-v1-0.6b";
    process.env.QMD_EMBED_BACKEND = "remote";

    const store = await createStore({
      dbPath: freshDbPath(),
      config: { collections: {} },
    });

    // Capture the model name searchVector forwards to the internal store.
    let capturedModel: string | undefined;
    const origSearchVec = store.internal.searchVec.bind(store.internal);
    store.internal.searchVec = (async (q: string, model: string, ...rest: unknown[]) => {
      capturedModel = model;
      // Return empty results; we only care about the model-name keying here.
      return [];
    }) as typeof store.internal.searchVec;

    await store.searchVector("hello world");
    expect(capturedModel).toBe("perplexity/pplx-embed-v1-0.6b");

    store.internal.searchVec = origSearchVec;
    await store.close();
  });
});
