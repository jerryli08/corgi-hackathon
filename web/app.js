// The resident's page. Everything the person sees is built with createElement and
// textContent -- their own message text comes back around through the robot, and it is
// never going anywhere near innerHTML.

const form = document.getElementById("text-form");
const input = document.getElementById("message");
const statusEl = document.getElementById("status");
const threadEl = document.getElementById("thread");
const liveEl = document.getElementById("live");
const livePhaseEl = document.getElementById("live-phase");
const liveBarEl = document.getElementById("live-bar");
const basketEl = document.getElementById("basket");
const camEl = document.getElementById("cam");
const rigEl = document.getElementById("rig");
const walkStateEl = document.getElementById("walk-state");

// How often a held walk button re-asks the robot to keep going. Comfortably inside the
// robot's dead-man window, so a held button never lapses -- and a released one stops
// within it, because nothing is left re-arming the timer.
const NUDGE_REPEAT_MS = 200;
const BUSY = new Set(["queued", "running"]);

// Robot messages are keyed so the poll below can add new ones without duplicating the
// ones already on screen.
const thread = [];
const seen = new Set();

function setStatus(text, bad = false) {
  statusEl.textContent = text || "";
  statusEl.classList.toggle("bad", Boolean(bad));
}

function renderThread() {
  threadEl.replaceChildren();
  if (!thread.length) {
    const li = document.createElement("li");
    li.className = "bubble robot empty";
    li.textContent = "Text me what you need — for example, bring me my water bottle.";
    threadEl.append(li);
    return;
  }
  for (const entry of thread) {
    const li = document.createElement("li");
    li.className = `bubble ${entry.who}${entry.pending ? " pending" : ""}`;
    li.textContent = entry.text;
    threadEl.append(li);
  }
  threadEl.scrollTop = threadEl.scrollHeight;
}

function addMessage(who, text, at, key) {
  if (key) {
    if (seen.has(key)) return false;
    seen.add(key);
  }
  thread.push({ who, text, at: at || Date.now() / 1000 });
  thread.sort((a, b) => a.at - b.at);
  renderThread();
  return true;
}

// The robot's side of the conversation. Only texts that actually went out are shown --
// the log also records the ones the robot deliberately withheld, and those belong in
// the ops console, not in someone's message thread.
async function refreshLog() {
  try {
    const res = await fetch("/api/imessage/log");
    if (!res.ok) return;
    const log = await res.json();
    for (const sent of log.outbound || []) {
      if (!sent.ok) continue;
      addMessage("robot", sent.text, sent.at, `out:${sent.at}:${sent.text}`);
    }
  } catch {
    /* the next poll picks it up */
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  const button = form.querySelector("button");
  addMessage("you", text, Date.now() / 1000);
  input.value = "";
  button.disabled = true;
  setStatus("Sending…");

  try {
    const res = await fetch("/api/imessage/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setStatus("");
    await refreshLog();
  } catch {
    setStatus("Could not reach the robot. Is the server running?", true);
  } finally {
    button.disabled = false;
    input.focus();
  }
});

// -- what the robot is doing ----------------------------------------------
function showPhase(text, progress) {
  liveEl.hidden = false;
  if (text) livePhaseEl.textContent = text;
  if (typeof progress === "number") {
    liveBarEl.style.width = `${Math.round(progress * 100)}%`;
  }
}

async function refreshState() {
  try {
    const state = await (await fetch("/api/state")).json();
    const basket = (state.robot && state.robot.basket) || [];
    basketEl.hidden = basket.length === 0;
    basketEl.textContent = basket.length ? `In my basket: ${basket.join(", ")}` : "";

    const walker = state.walker || {};
    walkStateEl.classList.toggle("on", Boolean(walker.active));
    if (!walker.active) {
      walkStateEl.textContent = "Not walking.";
    } else if (walker.moving) {
      walkStateEl.textContent = `Walking with you — going ${walker.direction}.`;
    } else {
      walkStateEl.textContent = "Here with you, standing still. Hold a button to move.";
    }

    const order = state.current_order;
    if (order && BUSY.has(order.status)) {
      showPhase(order.message || order.phase.toLowerCase(), order.progress);
    } else if (!walker.active && !order) {
      // Nothing on: keep the last phase visible rather than yanking the panel away
      // mid-sentence, but stop the bar from implying work is still happening.
      liveBarEl.style.width = "0%";
    }
  } catch {
    /* the next poll picks it up */
  }
}

// -- walking alongside ----------------------------------------------------
let holding = null;

async function nudge(direction) {
  try {
    await fetch("/api/walker/nudge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ direction }),
    });
  } catch {
    releaseHold();
  }
}

function releaseHold() {
  if (!holding) return;
  clearInterval(holding.timer);
  holding.button.classList.remove("holding");
  holding = null;
  // Nothing else to send: the robot stops on its own once the nudges stop arriving,
  // which is the whole point of the dead-man.
  refreshState();
}

function beginHold(button) {
  if (holding) releaseHold();
  const direction = button.dataset.direction;
  button.classList.add("holding");
  holding = { button, timer: setInterval(() => nudge(direction), NUDGE_REPEAT_MS) };
  nudge(direction);
}

for (const button of document.querySelectorAll(".jog")) {
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    beginHold(button);
  });
  for (const evt of ["pointerup", "pointercancel", "pointerleave", "blur"]) {
    button.addEventListener(evt, releaseHold);
  }
  // Holding a button and pressing space/enter should not latch it on.
  button.addEventListener("keydown", (event) => {
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      nudge(button.dataset.direction);
    }
  });
}

// A tab switch or a lost window must not leave a repeat running.
window.addEventListener("blur", releaseHold);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) releaseHold();
});

document.getElementById("walk-start").addEventListener("click", async () => {
  await fetch("/api/walker/start", { method: "POST" });
  refreshState();
});

document.getElementById("walk-stop").addEventListener("click", async () => {
  releaseHold();
  await fetch("/api/walker/stop", { method: "POST" });
  refreshState();
});

// -- boot -----------------------------------------------------------------
async function loadRig() {
  try {
    const health = await (await fetch("/api/health")).json();
    if (health.camera_present) {
      camEl.src = "/api/camera/stream.mjpg";
    } else {
      const figure = camEl.closest("figure");
      if (figure) figure.hidden = true;
    }
    rigEl.textContent = [
      health.mock ? "simulated robot" : "live robot",
      health.drive_connected ? "wheels ready" : "wheels offline",
      health.arm_present ? "arm ready" : "no arm",
      `texting via ${(health.messaging && health.messaging.backend) || "?"}`,
      `router: ${(health.router && health.router.backend) || "?"}`,
    ].join(" · ");
  } catch {
    rigEl.textContent = "robot offline";
  }
}

// The phase stream is what makes the wait feel like watching rather than waiting.
// Polling runs underneath it, so a dropped socket only costs responsiveness.
let socket = null;
let backoff = 1000;

function connectEvents() {
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
    if (data.type !== "phase") return;
    showPhase(data.human_text || data.phase.toLowerCase(), data.progress);
    // A milestone phase is also a text, so pick it up without waiting for the poll.
    if (["PRESENTING", "ARRIVED", "NEEDS_HELP", "FAILED"].includes(data.phase)) {
      refreshLog();
    }
    refreshState();
  };
  socket.onclose = () => {
    socket = null;
    // Capped backoff, and one socket at a time: a server restart used to leave a
    // browser tab opening a new socket every two seconds forever.
    backoff = Math.min(backoff * 2, 15000);
    setTimeout(connectEvents, backoff);
  };
  socket.onerror = () => {
    if (socket) socket.close();
  };
}

renderThread();
loadRig();
refreshLog();
refreshState();
connectEvents();
setInterval(refreshLog, 3000);
setInterval(refreshState, 1500);
