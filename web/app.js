const form = document.getElementById("order-form");
const itemInput = document.getElementById("item");
const statusEl = document.getElementById("status");
const ordersEl = document.getElementById("orders");

function setStatus(text) {
  statusEl.textContent = text || "";
}

async function refreshOrders() {
  const res = await fetch("/api/orders");
  if (!res.ok) return;
  const orders = await res.json();
  ordersEl.innerHTML = orders
    .slice(0, 8)
    .map(
      (o) =>
        `<li><span class="item">${escapeHtml(o.item)}</span><span class="badge">${escapeHtml(
          o.status
        )}</span></li>`
    )
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
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
  } catch (err) {
    setStatus("Could not reach the robot. Is the server running?");
  } finally {
    form.querySelector("button").disabled = false;
    itemInput.focus();
  }
});

refreshOrders();
setInterval(refreshOrders, 2000);
