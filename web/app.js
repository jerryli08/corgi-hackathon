const form = document.getElementById("order-form");
const itemInput = document.getElementById("item");
const statusEl = document.getElementById("status");
const ordersEl = document.getElementById("orders");
const liveEl = document.getElementById("live");
const livePhaseEl = document.getElementById("live-phase");
const liveBarEl = document.getElementById("live-bar");
const camEl = document.getElementById("cam");
const rigEl = document.getElementById("rig");

const BUSY = new Set(["queued", "running"]);

function setStatus(text) {
  statusEl.textContent = text || "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function loadRig() {
  try {
    const health = await (await fetch("/api/health")).json();
    if (health.camera_present) {
      camEl.src = "/api/camera/stream.mjpg";
    } else {
      camEl.closest("figure").hidden = true;
    }
    const bits = [
      health.mock ? "simulated robot" : "live robot",
      health.drive_connected ? "wheels ready" : "wheels offline",
      health.arm_present ? "arm ready" : "no arm",
    ];
    rigEl.textContent = bits.join(" · ");
  } catch {
    rigEl.textContent = "robot offline";
  }
}

function renderOrders(orders) {
  ordersEl.innerHTML = orders
    .slice(0, 8)
    .map((o) => {
      // A run that ended in NEEDS_HELP is a failure the customer can actually fix,
      // so say that rather than just "failed".
      const help = o.phase === "NEEDS_HELP";
      const label = help ? "needs you" : o.status;
      const title = help ? ` title="${escapeHtml(o.message)}"` : "";
      return (
        `<li${title}><span class="item">${escapeHtml(o.item)}</span>` +
        `<span class="badge ${help ? "help" : escapeHtml(o.status)}">` +
        `${escapeHtml(label)}</span></li>`
      );
    })
    .join("");
}

function renderLive(orders) {
  const active = orders.find((o) => BUSY.has(o.status));
  if (!active) {
    liveEl.hidden = true;
    return;
  }
  liveEl.hidden = false;
  livePhaseEl.textContent = active.message || active.phase.toLowerCase();
  liveBarEl.style.width = `${Math.round((active.progress || 0) * 100)}%`;
}

async function refreshOrders() {
  try {
    const res = await fetch("/api/orders");
    if (!res.ok) return;
    const orders = await res.json();
    renderOrders(orders);
    renderLive(orders);
  } catch {
    /* the poll below will pick it back up */
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const item = itemInput.value.trim();
  if (!item) return;

  setStatus("Sending…");
  form.querySelector("button").disabled = true;
  try {
    const res = await fetch("/api/orders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ item }),
    });
    if (!res.ok) throw new Error("Order failed");
    const order = await res.json();
    itemInput.value = "";
    setStatus(`Queued “${order.item}”`);
    await refreshOrders();
  } catch {
    setStatus("Could not reach the robot. Is the server running?");
  } finally {
    form.querySelector("button").disabled = false;
    itemInput.focus();
  }
});

// The phase stream is what makes the wait feel like watching rather than waiting.
// Polling still runs underneath it, so a dropped socket only costs responsiveness.
function connectEvents() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/events`);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type !== "phase") return;
    liveEl.hidden = false;
    livePhaseEl.textContent = data.human_text || data.phase.toLowerCase();
    if (typeof data.progress === "number") {
      liveBarEl.style.width = `${Math.round(data.progress * 100)}%`;
    }
  };
  ws.onclose = () => setTimeout(connectEvents, 2000);
}

loadRig();
refreshOrders();
connectEvents();
setInterval(refreshOrders, 2000);
