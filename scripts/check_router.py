#!/usr/bin/env python3
"""What can the router actually reach, and what does it do with a sentence?

Two things this answers that nothing else does:

  1. Which model ids your Merge key can really serve. The ones in config.py are a guess
     at a catalogue that moves, and a wrong id is a silent fallback to keywords rather
     than an error you would notice.
  2. What the configured router does with real messages, including which model served
     each one and whether it had to escalate.

    python scripts/check_router.py               # list models, then route the samples
    python scripts/check_router.py "get my pills"  # route one message of your own
"""

from __future__ import annotations

import asyncio
import sys

from robot.brain import RouterContext, make_router
from robot.config import (
    MERGE_API,
    MERGE_API_KEY,
    MERGE_BASE_URL,
    ROUTER_BACKEND,
    ROUTER_DEEP_MODEL,
    ROUTER_FAST_MODEL,
)

try:
    from merge_gateway import MergeGateway
except ImportError:
    MergeGateway = None  # type: ignore[assignment,misc]

SAMPLES = [
    "can you bring me my water bottle please",
    "banana",
    "bring me the water bottle and a banana",
    "come here",
    "walk with me to the kitchen",
    "no no stop",
    "whats going on",
    "get me the thing on the counter",
    "i've fallen and i need help",
    "thank you dear that was kind",
]


async def list_models() -> None:
    """Ask Merge what it can serve, and say plainly whether our configured ids are in it."""
    if not MERGE_API_KEY:
        print("MERGE_API_KEY is not set, so there is nothing to list.")
        print("The keyword router needs no key and the demo runs on it.\n")
        return
    if MergeGateway is None:
        print("the merge_gateway package is not installed (pip install merge-gateway-python).")
        print("The openai-compatible shim (CORGI_MERGE_API=openai) does not need it.\n")
        return

    client = MergeGateway(api_key=MERGE_API_KEY, base_url=MERGE_BASE_URL, timeout=15.0)
    print(f"GET {MERGE_BASE_URL}/models")
    try:
        listing = await asyncio.to_thread(client.models.list)
    except ValidationError as exc:
        # The live catalogue moves faster than a pinned SDK's schema -- a vendor
        # capability the installed version has never heard of (e.g. a new input type)
        # fails validation for every model in the same response, not just the new one.
        # That is a version-skew problem, not a routing problem, so say so briefly
        # instead of dumping pydantic's full field-by-field report.
        print(f"  {exc.error_count()} model(s) did not match the installed SDK's schema.")
        print("  Try: pip install -U merge-gateway-python\n")
        return
    except Exception as exc:  # this script exists to report failures
        print(f"  could not list models: {exc}\n")
        return
    finally:
        await asyncio.to_thread(client.close)

    names = sorted(m.id for m in listing.data)
    print(f"  {len(names)} models reachable")
    for name in names:
        print(f"    {name}")

    print()
    for label, want in (("fast", ROUTER_FAST_MODEL), ("deep", ROUTER_DEEP_MODEL)):
        mark = "ok" if want in names else "NOT IN THE LIST -- set it to one of the above"
        print(f"  {label:<5} {want:<45} {mark}")
    print()


async def route_all(texts: list[str]) -> int:
    router, notes = make_router()
    for note in notes:
        print(f"note: {note}")
    print(f"router backend: {router.name}  (CORGI_ROUTER_BACKEND={ROUTER_BACKEND})")
    if router.name == "merge":
        print(f"merge api: {MERGE_API}  base: {MERGE_BASE_URL}")
    print()

    ctx = RouterContext(
        known_items=["strawberries", "banana", "granola bar", "water bottle"],
        last_item="water bottle",
    )

    problems = 0
    for text in texts:
        intent = await router.route(text, ctx)
        r = intent.route
        detail = f"{r.backend}/{r.tier}"
        if r.served_by:
            detail += f" served_by={r.served_by}"
        if r.service_tier:
            detail += f" tier={r.service_tier}"
        if r.latency_ms:
            detail += f" {r.latency_ms}ms"
        if r.escalated:
            detail += " ESCALATED"
        if r.fell_back:
            detail += " FELL-BACK"
            problems += 1

        item = f" item={intent.item!r}" if intent.item else ""
        also = f" also={intent.also}" if intent.also else ""
        ask = " NEEDS-CLARIFYING" if intent.needs_clarification else ""
        print(f"{text!r}")
        print(f"    -> {intent.kind}{item}{also}{ask} conf={intent.confidence:.2f}  [{detail}]")
        if r.note:
            print(f"       {r.note}")
        if intent.reply:
            print(f"       reply: {intent.reply}")

    await router.aclose()

    if problems:
        print(
            f"\n{problems} of {len(texts)} fell back to keywords. "
            "That is the safe outcome, not a crash -- check the notes above for why."
        )
    return 0


async def main() -> int:
    texts = sys.argv[1:] or SAMPLES
    if not sys.argv[1:]:
        await list_models()
    return await route_all(texts)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
