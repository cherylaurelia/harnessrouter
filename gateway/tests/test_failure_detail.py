"""A failed turn's error names the base that ran and the tool that failed (#504).

The message says the turn failed and why; the detail says WHICH harness and WHICH tool, so the
debugging does not stop at the router. The runner's account of the failure (the tool's name) comes
in with the turn's record; the base is the harness's own name for the family that ran it.
"""
import asyncio
import json
import os
import pathlib
import sys
import time

import jsonschema
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("HR_BACKING", "local")
import app as gw  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCHEMA = os.path.join(_ROOT, "protocol", "schema", "uhp-2026-08-11.schema.json")
ORG = "org_e"


def _rec(**kw) -> dict:
    rec = {"base": "claude-code", "backend": "claude",
           "tried": [{"connection": "c", "status": "failed", "error": "boom"}],
           "error_detail": {"base": "claude", "tool": "Bash",
                            "exception": "ValueError: the tool blew up"}}
    rec.update(kw)
    return rec


def _failed(rec: dict) -> dict:
    tr = gw._RespTranslator("resp_test", "some-model", None, False, 0.0)
    gw._fail_events(tr, rec)
    return tr.error


def test_the_error_names_the_base_and_the_tool():
    err = _failed(_rec())
    assert err["type"] == "harness_error" and err["code"] == "turn_failed"
    assert err["detail"] == {"base": "claude-code", "tool": "Bash"}
    assert "boom" in err["message"]


def test_the_error_still_validates_against_the_spec():
    s = json.load(open(_SCHEMA))
    err = _failed(_rec())
    jsonschema.validate(err, {"$defs": s["$defs"], **s["$defs"]["Error"]})


def test_a_turn_with_no_failing_tool_names_the_base_only():
    rec = _rec(error_detail={"base": "claude", "tool": "", "exception": "boom"})
    err = _failed(rec)
    assert err["detail"] == {"base": "claude-code"}


def test_the_harnesses_own_base_is_named_when_it_says_one():
    rec = _rec(base="", backend="codex")
    err = _failed(rec)
    assert err["detail"] == {"base": "codex", "tool": "Bash"}


def test_the_errors_own_detail_is_kept_where_the_turn_has_one():
    """A turn the server cut short (interrupted, capped) carries its own detail; the runner's
    account of a failure does not overwrite it."""
    rec = _rec(error=gw._cap_error())
    err = _failed(rec)
    assert err["detail"] == {"reason": "timeout"}


# ── the whole path, from the runner's answer to the response and the trace ─────────────────────

def _drive_a_failed_turn(monkeypatch, start=None):
    """Run _resp_execute against a runner that answers a failed turn the way the runner answers
    one now: its own account of the failure (the base, the tool, the exception) beside the one
    line of reason. `start` answers the call that starts the turn instead, for one that never does.
    Returns (status, the record, the translator, the events written to the trace).
    """
    detail = {"base": "claude", "tool": "Bash",
              "exception": "ValueError: the tool blew up\n  File \"/app/tools.py\", line 42, in run"}
    traced: list = []

    async def _sandbox_json(path, sid, method="POST", body=None, **kw):
        if path == "/turn":
            if start is not None:
                raise start
            return {"turn_id": "t1", "status": "running"}
        if path.startswith("/turn/"):
            return {"done": True, "status": "failed", "error": "the CLI own sentence",
                    "error_detail": detail,
                    "events": [{"type": "result", "subtype": "error", "is_error": True,
                                "result": "the CLI own sentence", "error_detail": detail}],
                    "n_total": 1}
        return {}

    async def _nothing(*a, **kw):
        return None

    async def _get_connection(org, name):
        return ({"provider": "anthropic", "base_url": "https://proxy.example/v1"}, "tenant-1")

    async def _harness_plugins(*a, **kw):
        return ([], [], [], [], [])

    async def _harness_env(*a, **kw):
        return ({}, [])

    async def _image_auth(*a, **kw):
        return None

    def _auth_from_conn(*a, **kw):
        return {"token": "t", "base_url": "https://proxy.example/v1"}

    async def _turn_cancelled(*a, **kw):
        return False

    async def _collect_produced(*a, **kw):
        return []

    async def _trace_flush(tr, s):
        traced.extend(s.get("events") or [])
        return True

    for name, fn in (("_sandbox_json", _sandbox_json), ("_vertex_get", _nothing),
                     ("_vertex_upsert", _nothing), ("_hydrate", _nothing),
                     ("_harness_plugins", _harness_plugins), ("_harness_env", _harness_env),
                     ("_image_auth", _image_auth), ("_get_connection", _get_connection),
                     ("_auth_from_conn", _auth_from_conn), ("_turn_cancelled", _turn_cancelled),
                     ("_collect_produced", _collect_produced), ("_blob_put", _nothing),
                     ("_checkpoint", _nothing), ("_trace_finalize", _nothing),
                     ("_browser_close", _nothing), ("_recover_trace_cursor", _nothing),
                     ("_trace_flush", _trace_flush)):
        monkeypatch.setattr(gw, name, fn)
    monkeypatch.setattr(gw, "RESP_POLL_S", 0)
    monkeypatch.setattr(gw.control_store, "enabled", lambda: False)

    tr = gw._RespTranslator("resp_fail", "m", None, True, time.time())
    status, _produced, rec = asyncio.run(gw._resp_execute(
        tr, org=ORG, member="me@local", sid="sess_fail", backend="claude", chain=["anthropic"],
        prompt="hi", files_in=[], resume=None, emit=lambda ev: asyncio.sleep(0)))
    gw._fail_events(tr, rec)      # what create_response does with a failed turn
    return status, rec, tr, traced


def test_the_whole_path_names_the_base_and_the_tool_and_keeps_the_exception(monkeypatch):
    status, rec, tr, traced = _drive_a_failed_turn(monkeypatch)
    assert status == "failed", rec.get("tried")
    err = tr.error
    assert err["type"] == "harness_error" and err["code"] == "turn_failed"
    assert err["detail"] == {"base": "claude", "tool": "Bash"}
    assert "the CLI own sentence" in err["message"]
    assert not [o for o in tr.output if o.get("type") == "message"]   # said once, as the error
    # the trace keeps the harness's own account of the failure, exception and stack frames whole
    result_ev = [ev for ev in traced if ev.get("type") == "result"]
    assert result_ev and "ValueError: the tool blew up" in result_ev[-1]["error_detail"]["exception"]
    assert 'line 42, in run' in result_ev[-1]["error_detail"]["exception"]


def test_a_turn_the_runner_never_took_does_not_blame_the_connections(monkeypatch):
    """The connection was usable and the runner failed every try. With no status on its entry this read as
    a connection skipped: "No connection on this account serves <model>", after five minutes (0.38.0)."""
    status, rec, tr, _ = _drive_a_failed_turn(monkeypatch, RuntimeError(
        "sandbox /turn failed after 28 of 28 tries (HTTP 500, 21B body)"))
    assert status == "failed"
    said = gw._turn_failure_message(rec)
    assert said.startswith("The turn failed: it couldn't start here.") and "Try again" in said
    assert "No connection" not in said and "sandbox" not in said and "HTTP 500" not in said
    assert not rec.get("error_message")           # not read as the provider refusing the key


def test_a_session_tree_verdict_reaches_the_person_as_said(monkeypatch):
    why = "a file in this task's workspace can't be written, so it can't go on. Start a new task."
    status, rec, tr, _ = _drive_a_failed_turn(monkeypatch, gw._SandboxRefused("sandbox /turn failed", 409, why))
    assert status == "failed" and gw._turn_failure_message(rec) == f"The turn failed: {why}"
    assert not rec.get("error_message")


def test_a_result_with_the_failure_as_error_adds_no_answer():
    """The runner's own result for a CLI that died before it said anything carries the reason as
    `error` (runner/tests/test_failure_detail.py): fed to the translator it closes the turn and says
    nothing, where a reason in `result` was drawn as the agent's reply."""
    made = {"type": "result", "subtype": "error", "is_error": True, "result": "",
            "error": "ValueError: the tool blew up", "error_detail": {"base": "claude", "tool": ""}}
    tr = gw._RespTranslator("resp_made", "m", None, False, 0.0)
    tr.feed(made)
    assert not tr.any_text
    assert not [o for o in tr.output if o.get("type") == "message"]
    said = gw._RespTranslator("resp_said", "m", None, False, 0.0)
    said.feed({**made, "result": "ValueError: the tool blew up", "error": None})
    assert not said.any_text   # the reason in `result` is not an answer the agent gave either
    assert not [o for o in said.output if o.get("type") == "message"]


# ── a CLI's own failed result, as each base's normaliser hands it over ─────────────────────────
# The result events of three failed turns, read back from a Community Edition instance's trace
# (`GET /v1/traces/{sid}/all`, image of 2026-11-20) with the provider answering 500 or 400.
# Codex's is its retry log, Claude Code's the "API Error" line _strip_claude_error_text keeps out
# of the answer, OpenCode's the provider's message.
_CODEX_RETRIES = "\n".join(
    [f"Reconnecting... {i}/5 (We're currently experiencing high demand, which may cause temporary errors.)"
     for i in range(1, 6)] + ["We're currently experiencing high demand, which may cause temporary errors."])
_CLI_FAILED_RESULTS = {
    "codex": {"type": "result", "subtype": "error", "is_error": True, "result": _CODEX_RETRIES, "usage": {}},
    "claude": {"type": "result", "subtype": "success", "is_error": True,
               "result": "API Error: 400 stub forced 400", "stop_reason": "stop_sequence"},
    "opencode": {"type": "result", "subtype": "error", "is_error": True, "result": "stub forced 500", "usage": {}},
}


def _failed_turn(result: dict, before: list[dict] = ()) -> tuple[gw._RespTranslator, list[dict]]:
    """Feed a failed turn's events and close it the way create_response closes a failed turn."""
    tr = gw._RespTranslator("resp_cli", "m", None, False, 0.0)
    events = tr.start()
    for ev in [*before, result]:
        events += tr.feed(ev)
    events += gw._fail_events(tr, {"backend": "codex", "tried": [
        {"connection": "c", "status": "failed", "error": str(result.get("result") or "")}]})
    return tr, events


def _answer(tr: gw._RespTranslator) -> str:
    """What a client reads as the reply: the text of the response's message items (output_text)."""
    return "".join(c.get("text") or "" for o in tr.output if o.get("type") == "message"
                   for c in o.get("content") or [])


@pytest.mark.parametrize("base", sorted(_CLI_FAILED_RESULTS))
def test_a_cli_failed_result_is_not_drawn_as_the_answer(base):
    result = _CLI_FAILED_RESULTS[base]
    tr, events = _failed_turn(result)
    assert _answer(tr) == ""
    assert not [e for e in events if e["type"].startswith(("response.output_text", "response.content_part"))]
    assert events[-1]["type"] == "response.failed" and events[-1]["response"]["output"] == []
    assert result["result"].splitlines()[0] in tr.error["message"]   # the reason is said once, as the error


def test_what_the_agent_said_before_it_failed_is_kept():
    """A turn that answered and then failed keeps the answer it streamed; the reason is the error's."""
    said = {"type": "assistant", "message": {"content": [{"type": "text", "text": "Half of the summary"}]}}
    tr, _ = _failed_turn(_CLI_FAILED_RESULTS["opencode"], [said])
    assert _answer(tr) == "Half of the summary"
    assert "stub forced 500" in tr.error["message"]


def test_a_successful_result_with_nothing_said_before_is_still_the_answer():
    """Claude's stream-json can carry the final text on the result alone; that stays the answer."""
    tr = gw._RespTranslator("resp_ok", "m", None, False, 0.0)
    tr.feed({"type": "result", "subtype": "success", "is_error": False, "result": "it works.", "usage": {}})
    tr.complete("completed")
    assert _answer(tr) == "it works."
