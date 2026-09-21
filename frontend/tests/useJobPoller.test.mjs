import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

// Exercise the hook's effect with deterministic state and timer substitutes.
for (const hookName of ["useJobPoller", "useFacePoller"]) {
const source = readFileSync(new URL("../src/hooks/" + hookName + ".js", import.meta.url), "utf8")
  .replace(/^import .*;$/gm, "")
  .replaceAll("useFacePoller", "useJobPoller")
  .replaceAll("getFaceStatus", "getStatus")
  .replaceAll("getFaceResults", "getResults")
  .replace("export function useJobPoller", "function useJobPoller");
const flush = () => new Promise((resolve) => setImmediate(resolve));

function setup(getStatus, getResults = async () => ["result"]) {
  const state = [];
  const timers = new Map();
  let nextId = 0;
  let effect;
  const context = {
    Error, getStatus, getResults,
    useState(value) {
      const index = state.length;
      state.push(value);
      return [value, (next) => { state[index] = next; }];
    },
    useEffect(callback) { effect = callback; },
    window: {
      setTimeout(callback, delay) {
        const id = ++nextId;
        timers.set(id, { callback, delay });
        return id;
      },
      clearTimeout(id) { timers.delete(id); },
    },
  };
  vm.runInNewContext(source + '; useJobPoller("job");', context);
  const cleanup = effect();
  return {
    state, timers, cleanup,
    async tick() {
      assert.equal(timers.size, 1);
      const [id, timer] = timers.entries().next().value;
      timers.delete(id);
      timer.callback();
      await flush();
    },
  };
}

test("timeout retries, preserves progress, and recovers through completion", async () => {
  let calls = 0;
  const hook = setup(async () => {
    calls++;
    if (calls === 2) throw new Error("signal timed out");
    return { status: calls === 4 ? "completed" : "processing", progress_pct: 40 };
  });
  await flush();
  await hook.tick();
  assert.equal(hook.state[0].progress_pct, 40);
  assert.equal(hook.state[2], true);
  assert.match(hook.state[3], /retrying automatically/);
  assert.equal([...hook.timers.values()][0].delay, 3000);
  await hook.tick();
  assert.equal(hook.state[3], "");
  await hook.tick();
  assert.deepEqual(hook.state[1], ["result"]);
  assert.equal(hook.state[2], false);
  assert.equal(hook.timers.size, 0);
});

test("no overlapping requests while status is pending", async () => {
  let resolve;
  const hook = setup(() => new Promise((done) => { resolve = done; }));
  assert.equal(hook.timers.size, 0);
  resolve({ status: "processing" });
  await flush();
  assert.equal(hook.timers.size, 1);
  hook.cleanup();
  assert.equal(hook.timers.size, 0);
});

test("failed and missing jobs stop polling", async () => {
  for (const missing of [false, true]) {
    const hook = setup(async () => {
      if (missing) throw Object.assign(new Error("missing"), { status: 404 });
      return { status: "failed", error: "OCR failed" };
    });
    await flush();
    assert.equal(hook.state[2], false);
    assert.equal(hook.timers.size, 0);
    assert.match(hook.state[3], missing ? /no longer available/ : /OCR failed/);
  }
});

test("temporary result-fetch failure retries without restarting extraction", async () => {
  let calls = 0;
  const hook = setup(async () => ({ status: "completed" }), async () => {
    if (++calls === 1) throw new Error("connection lost");
    return ["saved"];
  });
  await flush();
  assert.equal(hook.state[2], true);
  await hook.tick();
  assert.deepEqual(hook.state[1], ["saved"]);
  assert.equal(hook.state[3], "");
  assert.equal(hook.state[2], false);
});

test("cleanup ignores late errors and schedules no retries", async () => {
  let reject;
  const hook = setup(() => new Promise((_, fail) => { reject = fail; }));
  hook.cleanup();
  reject(new Error("late timeout"));
  await flush();
  assert.equal(hook.timers.size, 0);
  assert.equal(hook.state[3], "");
});
}
