"""Candidate agent for real AppWorld, driven entirely through the FLYWHEEL MCP surface.

It never touches AppWorld in-process: it reaches the environment only via FLYWHEEL_MCP_URL's
tools (search_apis, api_doc, call_api, complete_task), the model via FLYWHEEL_PROXY_URL, and
its skill store via FLYWHEEL_MEMORY_URL. Each turn the model emits ONE json tool call; we
dispatch it over MCP and feed the result back. Procedural memory (which apps it has logged
into this stream) is read at start and written at end, so the memory on/off ablation is real.
"""
import json
import os
import re
import urllib.request

MCP_URL = os.environ["FLYWHEEL_MCP_URL"]
MEMORY_URL = os.environ["FLYWHEEL_MEMORY_URL"]
PROXY_URL = os.environ["FLYWHEEL_PROXY_URL"]
PROXY_TOKEN = os.environ.get("FLYWHEEL_PROXY_TOKEN", "")
INSTRUCTION = os.environ["FLYWHEEL_TASK_INSTRUCTION"]
MAX_STEPS = int(os.environ.get("FLYWHEEL_MAX_STEPS", "20"))

SYSTEM = (
    "You solve a task in the AppWorld environment by calling tools over an MCP surface.\n"
    "Each turn output ONE json object and nothing else, one of:\n"
    '  {"tool":"search_apis","query":"..."}                      discover APIs by keyword\n'
    '  {"tool":"api_doc","app":"...","api":"..."}                read one API\'s params\n'
    '  {"tool":"call_api","app":"...","api":"...","arguments":{}} call apis.<app>.<api>(**arguments)\n'
    '  {"tool":"complete_task","answer":"..."}                   finish (omit answer if none asked)\n'
    "Login flow: call_api supervisor.show_account_passwords (list of {account_name,password}),\n"
    "call_api supervisor.show_profile (has email, phone_number), then call_api <app>.login with\n"
    "username+password to get access_token; pass access_token in later call_api arguments.\n"
    "Inspect each result before indexing it. Page list APIs with page_index=0,1,2,... Keep the\n"
    "final answer concise (a number, name, or comma-separated list). Output ONLY the json object."
)

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


def rpc(method, params):
    return _post(MCP_URL, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).get("result") or {}


def call_tool(name, args):
    return rpc("tools/call", {"name": name, "arguments": args})


def memory_read():
    return _post(MEMORY_URL + "/read", {})


def memory_write(key, value):
    return _post(MEMORY_URL + "/write", {"key": key, "value": value})


def chat(messages):
    body = {"model": "gemini-3.1-flash-lite", "messages": messages}
    req = urllib.request.Request(PROXY_URL.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json",
                                          "authorization": f"Bearer {PROXY_TOKEN}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


def parse_action(text):
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def dispatch(action):
    tool = action.get("tool")
    if tool == "search_apis":
        return call_tool("search_apis", {"query": action.get("query", "")})
    if tool == "api_doc":
        return call_tool("api_doc", {"app": action.get("app", ""), "api": action.get("api", "")})
    if tool == "call_api":
        return call_tool("call_api", {"app": action.get("app", ""), "api": action.get("api", ""),
                                      "arguments": action.get("arguments") or {}})
    if tool == "complete_task":
        return call_tool("complete_task", {"answer": action.get("answer")} if "answer" in action else {})
    return {"error": f"unknown tool {tool}"}


def main():
    skills = memory_read()
    known = sorted(skills.get("apps", []))
    hint = ""
    if known:
        hint = ("\n(Session note: earlier tasks logged into these apps: " + ", ".join(known[:8]) +
                ". Re-login and re-fetch fresh for THIS task; treat the note as context only.)")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Task:\n{INSTRUCTION}{hint}\n\nBegin. Output your first json tool call."},
    ]
    apps_seen = set()
    for _ in range(MAX_STEPS):
        try:
            content = chat(messages)
        except Exception as e:
            messages.append({"role": "user", "content": f"model error: {e}. Output one json tool call."})
            continue
        action = parse_action(content)
        if not action:
            messages.append({"role": "user", "content": "Output exactly one json tool call."})
            continue
        if action.get("tool") == "call_api" and action.get("app") not in ("supervisor", "api_docs"):
            apps_seen.add(action.get("app"))
        result = dispatch(action)
        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": f"Result:\n{json.dumps(result, default=str)[:1500]}\n\nNext json tool call, or complete_task if done."})
        if action.get("tool") == "complete_task":
            break

    if apps_seen:
        merged = sorted(set(known) | apps_seen)
        memory_write("apps", merged)


if __name__ == "__main__":
    main()
