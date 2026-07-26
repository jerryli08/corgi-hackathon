# bridge — the Photon sidecar

## 1. What this is for

The robot is one Python process, but Photon's send path is a TypeScript SDK
(`spectrum-ts`) with no documented REST equivalent, so `robot/messaging.py` cannot call it
directly. `photon-bridge.mjs` is a small Node process that exists only to make that one
SDK call: the Python side POSTs `{spaceId, text, to}` to it and it puts the text on
iMessage. It is **outbound only**. Nothing inbound goes through it, because inbound needs
no sidecar at all — Photon delivers a signed webhook straight to FastAPI. If you skip this
whole file the robot still boots and still receives texts through the browser simulator;
what you lose is real outgoing iMessage.

```
inbound:   phone --iMessage--> Photon Spectrum --POST /api/imessage/webhook--> FastAPI :8000
outbound:  FastAPI :8000 --POST /send--> sidecar :8787 --spectrum-ts--> Photon Spectrum
delivery:  Photon Spectrum --iMessage--> phone
```

## 2. Get a projectId and a projectSecret

Sign in at <https://app.photon.codes>, create a project, and enable the iMessage provider
on it. The project page gives you two values:

* `projectId` — stable, safe to paste around, and it is also the HTTP username below.
* `projectSecret` — shown once. Copy it now. If you lose it, rotate it from the same page
  and redo step 4, because rotating the secret invalidates the webhook registration call.

Keep both in your shell for the rest of this document:

```bash
export PROJECT_ID='...'
export PROJECT_SECRET='...'
```

These two names are the sidecar's own environment variables (`SPECTRUM_PROJECT_ID`,
`SPECTRUM_PROJECT_SECRET`) and they are **not** `CORGI_` variables — the robot never sees
them. Only the sidecar holds the credentials.

## 3. Expose FastAPI so Photon can reach it

Photon has to POST into the Mac, so the robot needs a public HTTPS URL. Start the robot
first, on its normal port 8000:

```bash
bash scripts/run_server.sh
curl -sS localhost:8000/api/health
```

Then, in a second terminal:

```bash
ngrok http 8000
```

ngrok prints a `Forwarding` line like `https://a1b2c3d4.ngrok-free.app -> http://localhost:8000`.
Take the https half:

```bash
export PUBLIC_URL='https://a1b2c3d4.ngrok-free.app'
curl -sS "$PUBLIC_URL/api/health"
```

That URL changes every time you restart ngrok on a free account, and every restart means
redoing step 4. Leave the tunnel up for the whole demo.

## 4. Register the webhook

```bash
curl -sS -X POST "https://spectrum.photon.codes/projects/$PROJECT_ID/webhooks/" \
  -u "$PROJECT_ID:$PROJECT_SECRET" \
  -H 'content-type: application/json' \
  -d "{\"webhookUrl\": \"$PUBLIC_URL/api/imessage/webhook\"}"
```

The path matters: `/api/imessage/webhook`, on the ngrok host, not on localhost.

The response carries a `signingSecret`. **It is shown once, in this response only.** Every
webhook delivery is signed with it, and FastAPI rejects deliveries it cannot verify, so put
it in `.env` before you close the terminal:

```
CORGI_PHOTON_WEBHOOK_SECRET=<the signingSecret from the response>
```

List what is currently registered:

```bash
curl -sS "https://spectrum.photon.codes/projects/$PROJECT_ID/webhooks/" \
  -u "$PROJECT_ID:$PROJECT_SECRET"
```

Delete a stale one — you will want this after an ngrok restart, since a dead URL keeps
getting retried:

```bash
export WEBHOOK_ID='...'   # the id from the list call
curl -sS -X DELETE "https://spectrum.photon.codes/projects/$PROJECT_ID/webhooks/$WEBHOOK_ID" \
  -u "$PROJECT_ID:$PROJECT_SECRET"
```

Each registration mints its own `signingSecret`, so re-registering means updating
`CORGI_PHOTON_WEBHOOK_SECRET` again.

## 5. Start the sidecar

```bash
cd bridge
npm install
SPECTRUM_PROJECT_ID="$PROJECT_ID" SPECTRUM_PROJECT_SECRET="$PROJECT_SECRET" npm start
```

It listens on 8787 (`PORT` overrides that; if you change it, change
`CORGI_PHOTON_BRIDGE_URL` to match). Missing credentials are the one thing that exits
non-zero, with the two variable names in the message. Then, in another terminal:

```bash
curl -sS localhost:8787/health
```

```json
{"ok": true, "project": "prj_...", "ready": true, "reason": ""}
```

`ready: true` means the Spectrum SDK is up and sends will be attempted. `ready: false`
comes back with a `reason` string and is still a 200 — the sidecar deliberately stays
listening after a failed init, because it retries on the next send and a process that
reports its own diagnosis is more use than one that exited. Read the `reason` before you
start debugging anything else. Every send logs one line to this terminal, so keep it
visible during the demo.

## 6. Point the robot at it

In `.env` at the repo root:

```
CORGI_MESSAGING_BACKEND=photon
CORGI_PHOTON_BRIDGE_URL=http://127.0.0.1:8787
CORGI_PHOTON_WEBHOOK_SECRET=<from step 4>
```

No trailing slash on the bridge URL: the robot POSTs to `{CORGI_PHOTON_BRIDGE_URL}/send`.
Optional, all with working defaults: `CORGI_PHOTON_BRIDGE_TIMEOUT_S` (6.0),
`CORGI_PHOTON_WEBHOOK_TOLERANCE_S` (300), `CORGI_PHOTON_REQUIRE_SIGNATURE` (1),
`CORGI_ALLOWED_SENDERS` (blank means answer anyone),
`CORGI_DAILY_MESSAGE_BUDGET` (200). `.env` is read once at import, so restart the robot
after editing it, then confirm the backend took:

```bash
curl -sS localhost:8000/api/health          # "messaging": {"backend": "photon", ...}
```

If that says `"backend": "log"`, the backend degraded on purpose rather than failing the
boot; the reason is in the `notes` list in the same response.

## 7. The no-cloud fallback

```
CORGI_MESSAGING_BACKEND=applescript
```

This sends through the Messages app already signed in on this Mac, via `osascript`. No
Photon account, no keys, no tunnel, and the texts are genuinely iMessage. macOS will ask
the terminal for permission the first time; if the send fails silently, grant it under
System Settings → Privacy & Security → Automation (the terminal controlling **Messages**)
and Full Disk Access for the same terminal, then restart the terminal.

Reach for it when the sidecar will not start (`npm install` fails, or `/health` stays
`ready: false`) or when there is no ngrok. What it cannot do is **receive**: there is no
inbound path through Messages. Inbound still needs either the Photon webhook from step 4 or
the browser phone simulator at <http://localhost:8000/>, which posts to
`/api/imessage/simulate` and is gated by `CORGI_ALLOW_SIMULATED_TEXTS` (on by default).
The simulator plus this backend is a working demo: you type into the browser, the reply
arrives on a real phone.

It also needs a recipient string, since Messages resolves a buddy rather than a Spectrum
conversation id. A send with no `to` raises instead of guessing.

## 8. Troubleshooting

**The webhook returns 401.** The signature did not verify. Either
`CORGI_PHOTON_WEBHOOK_SECRET` is not the `signingSecret` from the *current* registration
(re-registering mints a new one — list the webhooks, delete the extras, re-register once and
copy carefully), or the Mac's clock is more than `CORGI_PHOTON_WEBHOOK_TOLERANCE_S` seconds
off, which the verifier treats as a replay. Check with `date -u` against
`curl -sI https://spectrum.photon.codes | grep -i ^date`. To poke the endpoint by hand
without a signature, set `CORGI_PHOTON_REQUIRE_SIGNATURE=0` — for debugging only, and put it
back before the demo.

**The webhook returns 200 with `{"ignored": true}`.** Normal, and not a failure. The
envelope was not an inbound text: a delivery receipt, a reaction, an outbound echo of the
robot's own reply, an attachment with no text. Photon retries any non-2xx, so an event we do
not act on has to answer 200. Send yourself a plain "hello" and watch for a non-ignored
response before suspecting anything.

**Sends fail with 502.** That is the sidecar saying it could not hand the text to Spectrum,
and the error string in the body is the real reason. Check `curl -sS localhost:8787/health`
for `ready` and `reason`, and the sidecar's own log line for that send. Usual causes: no
network from the Mac, a rotated `projectSecret`, or a `spectrum-ts` version whose surface
does not match — the three functions at the top of `photon-bridge.mjs` are the only places
that touch the SDK, and a shape mismatch reports itself in that error string rather than
killing the process. The robot records the failure as a dropped text; see
`curl -sS localhost:8000/api/imessage/log`. While you fix it, switch to
`CORGI_MESSAGING_BACKEND=applescript`.

**The recipient is an Apple ID email, not a phone number.** Photon's iMessage provider needs
a real phone number, and the sidecar refuses anything that is not E.164 with a 400 naming
the value it got (`+15551234567` is the shape). Replies into an existing conversation are
unaffected, because those carry a `spaceId` and never need the number. Only starting a
thread does. Messages on this Mac does resolve Apple ID emails, so the
`applescript` backend is the workaround for an email-only contact.

**The 5000/day quota.** Photon allows 5000 messages per server per day by default. The
robot is nowhere near it, and `CORGI_DAILY_MESSAGE_BUDGET` (200) caps us far below on
purpose — an elderly user must not get nine texts about one water bottle. Over-budget and
too-soon texts are dropped with a reason, not raised as errors, so a suspiciously quiet
robot is a `stats` question first:

```bash
curl -sS localhost:8000/api/imessage/log    # stats.remaining, and the drop reasons
```
