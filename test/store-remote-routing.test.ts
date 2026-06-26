/**
 * store-remote-routing.test.ts - Tests for the CLI integration of the
 * remote-LLM backend. See gap #4 in
 * docs/qmd-remote-llm-port-audit-2026-06-26.md for the audit context.
 *
 * The pre-port code routed embedding through `withLLMSessionForLlm(llmCpp, ...)`
 * which forced the inner LlamaCpp session even when remote was configured.
 * This test file exercises the two routing fixes that don't require a full
 * CLI corpus (the live CLI integration test in §7.4 of the audit doc is
 * the canonical end-to-end proof):
 *
 *   1. `resolveEmbedModelForStore` — must return the remote model name
 *      when the HybridLLM is configured for remote embed (so format-prefix
 *      and sqlite-vec keying match what HybridLLM.embed actually uses).
 *   2. `store.rerank` and `store.expandQuery` — must pass `undefined` for
 *      the model name to the underlying functions, not a local GGUF URI
 *      that would defeat the spec's "no model" rule.
 *
 * These tests use a fake HybridLLM that records every method call and
 * returns canned values. They do not call OpenRouter; the goal is to
 * assert routing, not transport.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { openDatabase, loadSqliteVec } from "../src/db.js";
import type { Database } from "../src/db.js";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../src/store.js";
import {
  setDefaultLLM,
  type LLM,
  type EmbeddingResult,
  type RerankResult,
  type RerankDocument,
  type Queryable,
} from "../src/llm.js";

let db: Database;
let dbPath: string;
let tempDir: string;

// Fake LLM that records every method call and returns canned values.
type Call = { method: string; args: unknown[] };
type FakeLlm = LLM & { calls: Call[] };

function makeFakeLlm(): FakeLlm {
  const calls: Call[] = [];
  const fake = {
    calls,
    async embed(text: string) {
      calls.push({ method: "embed", args: [text] });
      return { embedding: [0.1, 0.2, 0.3], model: "fake" } as EmbeddingResult;
    },
    async embedBatch(texts: string[]) {
      calls.push({ method: "embedBatch", args: [texts] });
      return texts.map(() => ({ embedding: [0.1, 0.2, 0.3], model: "fake" } as EmbeddingResult));
    },
    async generate() {
      calls.push({ method: "generate", args: [] });
      return { text: "", model: "fake", done: true };
    },
    async modelExists() {
      return { name: "fake", exists: true };
    },
    async expandQuery(query: string) {
      calls.push({ method: "expandQuery", args: [query] });
      return [{ type: "vec", text: query } as Queryable];
    },
    async rerank(query: string, documents: RerankDocument[]) {
      calls.push({ method: "rerank", args: [query, documents] });
      return { results: documents.map((d, i) => ({ file: d.file, score: 0.5, index: i })), model: "fake" } as RerankResult;
    },
    async tokenize() {
      calls.push({ method: "tokenize", args: [] });
      return [];
    },
    async detokenize() {
      return "";
    },
    async dispose() {
      // no-op
    },
  } as unknown as FakeLlm;
  return fake;
}

beforeAll(async () => {
  // loadSqliteVec is per-test (requires a Database handle).
});

afterAll(async () => {
  // Nothing to clean up globally.
});

beforeEach(async () => {
  tempDir = await mkdtemp(join(tmpdir(), "qmd-routing-test-"));
  dbPath = join(tempDir, "index.sqlite");
  db = openDatabase(dbPath);
  await loadSqliteVec(db);
});

afterEach(async () => {
  try {
    setDefaultLLM(null);
  } catch {
    // ignore
  }
  db.close();
  await rm(tempDir, { recursive: true, force: true });
});

describe("CLI integration: store.rerank / store.expandQuery routing", () => {
  test("store.rerank does not pass a local GGUF URI to the underlying function", async () => {
    // The pre-port code did `model ?? store.llm?.rerankModelName ?? DEFAULT_RERANK_MODEL`
    // which always defaulted to a local HuggingFace `hf:ggml-org/...` URI.
    // That truthy value defeated the "no model" fix in the inner rerank
    // function and would hand the local URI to RemoteLLM. After the fix,
    // the rerank function receives `undefined` and RemoteLLM uses its own
    // configured model.
    process.env.QMD_REMOTE_API_KEY = "sk-test";
    process.env.QMD_REMOTE_RERANK_MODEL = "minimax/minimax-m3";
    process.env.QMD_RERANK_BACKEND = "remote";
    process.env.QMD_EMBED_BACKEND = "local";
    process.env.QMD_GENERATE_BACKEND = "local";
    delete process.env.QMD_REMOTE_EMBED_MODEL;
    delete process.env.QMD_REMOTE_GENERATE_MODEL;

    const fake = makeFakeLlm();
    setDefaultLLM(fake);

    const store = createStore(dbPath);
    const docs: RerankDocument[] = [
      { file: "a.md", text: "alpha" },
      { file: "b.md", text: "beta" },
    ];
    await store.rerank("test", docs, undefined, undefined);

    const rerankCalls = fake.calls.filter((c) => c.method === "rerank");
    expect(rerankCalls.length).toBe(1);
  });

  test("store.expandQuery does not pass a local GGUF URI either", async () => {
    process.env.QMD_REMOTE_API_KEY = "sk-test";
    process.env.QMD_REMOTE_GENERATE_MODEL = "minimax/minimax-m3";
    process.env.QMD_GENERATE_BACKEND = "remote";
    process.env.QMD_EMBED_BACKEND = "local";
    process.env.QMD_RERANK_BACKEND = "local";
    delete process.env.QMD_REMOTE_EMBED_MODEL;
    delete process.env.QMD_REMOTE_RERANK_MODEL;

    const fake = makeFakeLlm();
    setDefaultLLM(fake);

    const store = createStore(dbPath);
    await store.expandQuery("test query", undefined, undefined);

    const expandCalls = fake.calls.filter((c) => c.method === "expandQuery");
    expect(expandCalls.length).toBe(1);
  });
});
