// Ops console. Everything the server sends is inserted as text, never as markup:
// human_text is built by formatting a user-supplied item label into a phrase, so the
// phase log is a straight path from a text message to this page.

const logEl = document.getElementById("log");
const camEl = document.getElementById("cam");
const rigEl = document.getElementById("rig");
const truthEl = document.getElementById("truth");
const targetEl = document.getElementById("target");
const basketEl = document.getElementById("basket");
const routesEl = document.getElementById("routes");
const routerBackendEl = document.getElementById("router-backend");
const messagesEl = document.getElementById("messages");
const outboxStatsEl = document.getElementById("outbox-stats");
const walkerStateEl = document.getElementById("walker-state");
const walkerStatsEl = document.getElementById("walker-stats");

let mock = false;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function line(parts, bad = false) {
  const div = el("div", bad ? "bad" : "");
  for (const part of parts) div.append(part);
  logEl.append(div);
  logEl.scrollTop = logEl.scrollHeight;
}

function stamp(at) {
  return new Date((at || Date.now() / 1000) * 1000).toLocaleTimeString();
}

async function post(path, payload) {
  let res;
  let text;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    text = await res.text();
  } catch (err) {
    line([el("span", "at", `→ ${path} `), el("span", undefined, String(err))], true);
    return null;
  }
  line([el("span", "at", `→ ${path} `), el("span", undefined, text.slice(0, 400))], !res.ok);
  return res;
}

async function boot() {
  let health;
  try {
    health = await (await fetch("/api/health")).json();
  } catch (err) {
    rigEl.textContent = "server unreachable";
    line([el("span", "at", "boot "), el("span", undefined, String(err))], true);
    return;
  }
  mock = health.mock;
  rigEl.textContent = [
    health.mock ? "MOCK" : "LIVE",
    `drive ${health.drive_connected ? "up" : "down"}`,
    `arm ${health.arm_present ? "up" : "none"}`,
    `vision ${health.vision_backend || "none"}`,
    `router ${(health.router && health.router.backend) || "none"}`,
    `texts ${(health.messaging && health.messaging.backend) || "none"}`,
  ].join("  ·  ");
  routerBackendEl.textContent = (health.router && health.router.backend) || "";
  if (health.camera_present) camEl.src = "/api/camera/stream.mjpg";
  for (const note of health.notes || []) {
    line([el("span", "at", "boot "), el("span", undefined, note)]);
  }
  if (mock) setInterval(pollTruth, 700);
}

async function pollTruth() {
  try {
    const world = await (await fetch("/api/debug/world")).json();
    const rows = world.objects.map(
      (o) =>
        `${o.label.padEnd(14)} d=${String(o.distance_m).padStart(6)}m  ` +
        `bearing=${String(o.bearing_deg).padStart(6)}°${o.held ? "  HELD" : ""}`
    );
    if (world.basket && world.basket.length) rows.push(`basket: ${world.basket.join(", ")}`);
    truthEl.textContent = rows.join("\n");
  } catch {
    truthEl.textContent = "";
  }
}

// -- router ---------------------------------------------------------------
function renderRoute(intent) {
  const row = el("div", "row");
  row.append(el("span", "kind", intent.kind));
  row.append(el("span", "said", ` ${intent.item ? `“${intent.item}” ` : ""}`));
  row.append(el("span", "said", `— ${intent.raw || ""}`));

  const r = intent.route || {};
  const bits = [
    `conf ${intent.confidence}`,
    `${r.backend}/${r.tier}`,
    r.model || "",
    r.served_by ? `served by ${r.served_by}` : "",
    r.service_tier ? `tier ${r.service_tier}` : "",
    r.latency_ms ? `${r.latency_ms}ms` : "",
  ].filter(Boolean);
  const meta = el("span", "meta", bits.join("  ·  "));
  if (r.escalated) meta.append(el("span", "flag", "  ESCALATED"));
  if (r.fell_back) meta.append(el("span", "flag", "  FELL BACK"));
  if (intent.needs_clarification) meta.append(el("span", "flag", "  NEEDS CLARIFYING"));
  if (r.note) meta.append(el("span", "mute", `  ${r.note}`));
  row.append(meta);

  routesEl.prepend(row);
  while (routesEl.childElementCount > 40) routesEl.lastElementChild.remove();
}

document.getElementById("preview-go").addEventListener("click", async () => {
  const text = document.getElementById("preview").value.trim();
  if (!text) return;
  try {
    const res = await fetch("/api/router/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderRoute(await res.json());
  } catch (err) {
    line([el("span", "at", "→ /api/router/preview "), el("span", undefined, String(err))], true);
  }
});

// -- imessage -------------------------------------------------------------
const seenMessages = new Set();

function renderMessage(dir, text, at, ok, detail) {
  const key = `${dir}:${at}:${text}`;
  if (seenMessages.has(key)) return;
  seenMessages.add(key);

  const row = el("div", `row ${dir === "out" ? "out" : "in"}${ok ? "" : " dropped"}`);
  row.append(el("span", "dir", dir === "out" ? "robot →" : "← person"));
  row.append(el("span", "said", text));
  const bits = [stamp(at)];
  if (!ok) bits.push(detail || "not sent");
  row.append(el("span", "meta", bits.join("  ·  ")));
  messagesEl.prepend(row);
  while (messagesEl.childElementCount > 60) messagesEl.lastElementChild.remove();
}

async function pollMessages() {
  try {
    const log = await (await fetch("/api/imessage/log")).json();
    for (const m of log.inbound || []) {
      const text = m.text || (m.message && m.message.text) || "";
      if (text) renderMessage("in", text, m.at, true, "");
    }
    for (const m of log.outbound || []) {
      renderMessage("out", m.text, m.at, m.ok, m.detail);
    }
    const s = log.stats || {};
    outboxStatsEl.textContent = `${s.backend || "?"} · sent ${s.sent} · dropped ${
      s.dropped
    } · ${s.remaining}/${s.budget} left today`;
  } catch {
    /* next poll */
  }
}

document.getElementById("text-go").addEventListener("click", async () => {
  const text = document.getElementById("text-in").value.trim();
  if (!text) return;
  const res = await post("/api/imessage/simulate", { text });
  if (res && res.ok) pollMessages();
});

// -- walker ---------------------------------------------------------------
async function pollWalker() {
  try {
    const state = await (await fetch("/api/walker/state")).json();
    walkerStateEl.textContent = JSON.stringify(state, null, 2);
    walkerStatsEl.textContent = state.active
      ? `active · ${state.moving ? state.direction : "still"} · dead-man ${
          state.deadman_ms_left
        }ms · ${state.session_s}s`
      : `off · ${state.reason || "—"}`;
  } catch {
    walkerStatsEl.textContent = "";
  }
}

for (const button of document.querySelectorAll("button[data-walk]")) {
  button.addEventListener("click", async () => {
    await post(`/api/walker/${button.dataset.walk}`);
    pollWalker();
  });
}

for (const button of document.querySelectorAll("button[data-nudge]")) {
  button.addEventListener("click", async () => {
    await post("/api/walker/nudge", { direction: button.dataset.nudge });
    pollWalker();
  });
}

// -- manual driving and skills -------------------------------------------
document.querySelectorAll(".jog button[data-linear]").forEach((button) => {
  button.addEventListener("click", () =>
    post("/api/drive/velocity", {
      linear: Number(button.dataset.linear),
      angular: Number(button.dataset.angular),
      ms: 600,
    })
  );
});

document.getElementById("halt").addEventListener("click", () => post("/api/drive/stop"));
document.getElementById("estop").addEventListener("click", () => post("/api/estop"));
document.getElementById("clear").addEventListener("click", () => logEl.replaceChildren());

document.querySelectorAll(".skills button[data-skill]").forEach((button) => {
  button.addEventListener("click", () => {
    const skill = button.dataset.skill;
    const payload = skill === "fetch" ? { label: targetEl.value.trim() } : {};
    post(`/api/skills/${skill}`, payload);
  });
});

async function pollState() {
  try {
    const state = await (await fetch("/api/state")).json();
    const basket = (state.robot && state.robot.basket) || [];
    basketEl.textContent = basket.length ? basket.join(", ") : "—";
  } catch {
    /* next poll */
  }
}

// -- events ---------------------------------------------------------------
let socket = null;
let backoff = 1000;

function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/api/events`);

  socket.onopen = () => {
    backoff = 1000;
  };
  socket.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }
    line(
      [
        el("span", "at", `${stamp(data.at)} `),
        el("span", "phase", data.phase || ""),
        el("span", undefined, ` ${data.human_text || ""}`),
      ],
      data.ok === false
    );
    if (data.intent) renderRoute(data.intent);
  };
  socket.onclose = () => {
    socket = null;
    backoff = Math.min(backoff * 2, 15000);
    setTimeout(connect, backoff);
  };
  socket.onerror = () => {
    if (socket) socket.close();
  };
}

boot();
connect();
pollMessages();
pollWalker();
pollState();
setInterval(pollMessages, 2000);
setInterval(pollWalker, 700);
setInterval(pollState, 1200);
