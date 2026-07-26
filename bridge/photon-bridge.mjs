/**
 * Outbound iMessage sends, on behalf of the Python robot.
 *
 * Photon's Spectrum SDK is TypeScript and there is no documented REST endpoint for
 * SENDING a message, so `robot/messaging.py` POSTs here instead and this process makes
 * the SDK call. Inbound does not need a sidecar at all: that arrives as a signed
 * webhook straight to FastAPI.
 *
 * The load-bearing idea is that this thing tells the truth about itself. The Python
 * side polls GET /health to decide whether to use Photon or fall back to the Messages
 * app, so a failed Spectrum init answers 200 with ready:false and the reason rather
 * than exiting -- a dead sidecar that is still listening is far more useful than one
 * that took its diagnosis to the grave. Only missing credentials exit non-zero, because
 * that one cannot be fixed by waiting.
 */

import { createServer } from "node:http";

const PORT = Number(process.env.PORT || 8787);
// A text is a text. Anything larger is a mistake or an attack, so read no further.
const MAX_BODY_BYTES = 64 * 1024;
const LOG_TEXT_CHARS = 40;
// Spectrum's iMessage provider needs a real phone number; Apple ID emails are out.
const E164 = /^\+[1-9]\d{6,14}$/;

const projectId = process.env.SPECTRUM_PROJECT_ID || "";
const projectSecret = process.env.SPECTRUM_PROJECT_SECRET || "";

if (!projectId || !projectSecret) {
  console.error(
    "Set SPECTRUM_PROJECT_ID and SPECTRUM_PROJECT_SECRET in the environment, " +
      "then start the bridge again.",
  );
  process.exit(1);
}

// --- The only three places that touch spectrum-ts -------------------------
// The SDK's exact surface cannot be verified from here, so every call into it lives in
// initSpectrum, resolveSpace and sendText. If the real shape differs, these are the
// only functions to change; everything below them is plain node:http. Each one probes
// the accessors the docs imply, in order, and throws a readable message if the SDK
// offers none of them -- that message travels back to Python as a 502 body instead of
// killing the process.

let sdk = null; // the awaited Spectrum app, or null
let readyReason = "not initialised yet";
let initInFlight = null;

async function initSpectrum() {
  const mod = await import("spectrum-ts");
  const Spectrum = mod.Spectrum ?? mod.default;
  const imessage = mod.imessage ?? mod.providers?.imessage;
  if (typeof Spectrum !== "function") {
    throw new Error("spectrum-ts did not export a callable Spectrum()");
  }
  if (!imessage || typeof imessage.config !== "function") {
    throw new Error("spectrum-ts did not export imessage.config()");
  }
  return await Spectrum({ projectId, projectSecret, providers: [imessage.config()] });
}

async function resolveSpace(app, { spaceId, to }) {
  if (spaceId) {
    if (typeof app.getSpace === "function") return await app.getSpace(spaceId);
    if (typeof app.spaces?.get === "function") return await app.spaces.get(spaceId);
    if (typeof app.space === "function") return await app.space(spaceId);
    // Nothing space-shaped on offer: hand the raw id to app.send(space, ...content).
    return spaceId;
  }
  if (typeof app.getSpaceByPhone === "function") return await app.getSpaceByPhone(to);
  if (typeof app.spaces?.findByPhone === "function") return await app.spaces.findByPhone(to);
  if (typeof app.spaces?.create === "function") {
    return await app.spaces.create({ platform: "iMessage", phone: to });
  }
  return to;
}

async function sendText(app, space, text) {
  // Documented form first: `await space.send("hello world")`.
  if (space && typeof space.send === "function") {
    await space.send(text);
    return;
  }
  if (typeof app.send === "function") {
    await app.send(space, text);
    return;
  }
  throw new Error("spectrum-ts exposed neither space.send() nor app.send()");
}

// --- Plain HTTP from here down --------------------------------------------

async function ensureApp() {
  if (sdk) return sdk;
  // One init at a time, but a later request may retry: the bridge is often started
  // before the network is up, and a permanent no would be a lie.
  if (!initInFlight) {
    initInFlight = initSpectrum()
      .then((app) => {
        sdk = app;
        readyReason = "";
        return app;
      })
      .catch((err) => {
        readyReason = describe(err);
        throw err;
      })
      .finally(() => {
        initInFlight = null;
      });
  }
  return await initInFlight;
}

function describe(err) {
  return (err && (err.message || String(err))) || "unknown error";
}

function reply(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        req.pause();
        reject(Object.assign(new Error("body too large"), { status: 413 }));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function logSend(spaceLabel, text, outcome) {
  const head = text.slice(0, LOG_TEXT_CHARS).replace(/\s+/g, " ");
  console.log(`send space=${spaceLabel} "${head}" ${outcome}`);
}

async function handleSend(req, res) {
  let raw;
  try {
    raw = await readBody(req);
  } catch (err) {
    // The rest of an oversized body is still in flight and we stopped reading it, so
    // drop the socket once the refusal is out rather than parsing the tail as a request.
    res.on("finish", () => req.destroy());
    return reply(res, err.status || 400, { ok: false, error: describe(err) });
  }

  let payload;
  try {
    payload = JSON.parse(raw || "{}");
  } catch {
    return reply(res, 400, { ok: false, error: "body is not JSON" });
  }
  if (!payload || typeof payload !== "object") {
    return reply(res, 400, { ok: false, error: "body must be a JSON object" });
  }

  const spaceId = typeof payload.spaceId === "string" ? payload.spaceId.trim() : "";
  const to = typeof payload.to === "string" ? payload.to.trim() : "";
  const text = typeof payload.text === "string" ? payload.text : "";
  if (!text.trim()) {
    return reply(res, 400, { ok: false, error: "text is required" });
  }
  if (!spaceId && !to) {
    return reply(res, 400, { ok: false, error: "spaceId or to is required" });
  }
  if (!spaceId && !E164.test(to)) {
    return reply(res, 400, {
      ok: false,
      error: `to must be a phone number in E.164 form, like +15551234567; got ${to}`,
    });
  }

  const label = spaceId || to;
  try {
    const app = await ensureApp();
    const space = await resolveSpace(app, { spaceId, to });
    if (!space) throw new Error(`no conversation for ${label}`);
    await sendText(app, space, text);
  } catch (err) {
    // Includes an SDK shape mismatch: report it and stay up, so the Python side can
    // read the real reason and switch backends.
    const error = describe(err);
    logSend(label, text, `failed: ${error}`);
    return reply(res, 502, { ok: false, error });
  }
  logSend(label, text, "ok");
  return reply(res, 200, { ok: true });
}

const server = createServer((req, res) => {
  const path = (req.url || "/").split("?")[0];
  if (req.method === "POST" && path === "/send") {
    handleSend(req, res).catch((err) => reply(res, 502, { ok: false, error: describe(err) }));
    return;
  }
  if (req.method === "GET" && path === "/health") {
    const ready = sdk !== null;
    return reply(res, 200, {
      ok: true,
      project: projectId,
      ready,
      reason: ready ? "" : readyReason,
    });
  }
  reply(res, 404, { ok: false, error: `no route for ${req.method} ${path}` });
});

server.listen(PORT, () => {
  console.log(`photon bridge listening on ${PORT}, project ${projectId}`);
});

// Warm the SDK up now so /health has an honest answer before the first send, and
// swallow the failure: the reason is already recorded and /health reports it.
ensureApp().catch(() => {});
