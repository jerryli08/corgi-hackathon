const logEl = document.getElementById("log");
const camEl = document.getElementById("cam");
const rigEl = document.getElementById("rig");
const truthEl = document.getElementById("truth");
const targetEl = document.getElementById("target");

let mock = false;

function line(html, bad = false) {
  const div = document.createElement("div");
  div.className = bad ? "bad" : "";
  div.innerHTML = html;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

function stamp(at) {
  return new Date((at || Date.now() / 1000) * 1000).toLocaleTimeString();
}

async function post(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  const text = await res.text();
  line(
    `<span class="at">→ ${path}</span> ${text.slice(0, 400)}`,
    !res.ok
  );
  return res;
}

async function boot() {
  const health = await (await fetch("/api/health")).json();
  mock = health.mock;
  rigEl.textContent = [
    health.mock ? "MOCK" : "LIVE",
    `drive ${health.drive_connected ? "up" : "down"}`,
    `arm ${health.arm_present ? "up" : "none"}`,
    `vision ${health.vision_backend || "none"}`,
  ].join("  ·  ");
  if (health.camera_present) camEl.src = "/api/camera/stream.mjpg";
  health.notes.forEach((note) => line(`<span class="at">boot</span> ${note}`));
  if (mock) setInterval(pollTruth, 700);
}

async function pollTruth() {
  try {
    const world = await (await fetch("/api/debug/world")).json();
    truthEl.textContent = world.objects
      .map(
        (o) =>
          `${o.label.padEnd(14)} d=${String(o.distance_m).padStart(6)}m  ` +
          `bearing=${String(o.bearing_deg).padStart(6)}°${o.held ? "  HELD" : ""}`
      )
      .join("\n");
  } catch {
    truthEl.textContent = "";
  }
}

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
document.getElementById("clear").addEventListener("click", () => (logEl.innerHTML = ""));

document.querySelectorAll(".skills button[data-skill]").forEach((button) => {
  button.addEventListener("click", () => {
    const skill = button.dataset.skill;
    const payload = skill === "fetch" ? { label: targetEl.value.trim() } : {};
    post(`/api/skills/${skill}`, payload);
  });
});

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/events`);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    line(
      `<span class="at">${stamp(data.at)}</span> ` +
        `<span class="phase">${data.phase}</span> ${data.human_text || ""}`,
      data.ok === false
    );
  };
  ws.onclose = () => setTimeout(connect, 2000);
}

boot();
connect();
