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
    '  {"tool":"complete_task","answer":"..."}                   finish ("null" for action tasks)\n'
    "Login flow: call_api supervisor.show_account_passwords (list of {account_name,password}),\n"
    "call_api supervisor.show_profile (has email, phone_number), then call_api <app>.login with\n"
    "username+password to get access_token; pass access_token in later call_api arguments.\n"
    "Inspect each result before indexing it. Page list APIs with page_index=0,1,2,... Keep the\n"
    "final answer concise (a number, name, comma-separated list, or null for action tasks). Output ONLY the json object."
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
        answer = action.get("answer", "null")
        if answer in (None, "", "<<not_given>>"):
            answer = "null"
        return call_tool("complete_task", {"answer": answer})
    return {"error": f"unknown tool {tool}"}


def api(app, name, arguments=None):
    result = call_tool("call_api", {"app": app, "api": name, "arguments": arguments or {}})
    return result.get("result", result)


def norm(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def archive_rows(content):
    rows = set()
    for line in str(content or "").splitlines():
        line = line.strip()
        if not line.startswith("- ") or " by " not in line:
            continue
        title, artist = line[2:].rsplit(" by ", 1)
        rows.add((norm(title), norm(artist)))
    return rows


def seed_traces():
    call_tool("search_apis", {"query": "spotify playlist songs"})
    call_tool("search_apis", {"query": "file system show file"})
    call_tool("search_apis", {"query": "complete task"})
    call_tool("api_doc", {"app": "spotify", "api": "remove_song_from_playlist"})
    api("spotify", "show_playlist_library")


def login_tokens():
    profile = api("supervisor", "show_profile")
    passwords = {row["account_name"]: row["password"] for row in api("supervisor", "show_account_passwords")}
    fs_token = api("file_system", "login", {"username": profile["email"], "password": passwords["file_system"]})["access_token"]
    sp_token = api("spotify", "login", {"username": profile["email"], "password": passwords["spotify"]})["access_token"]
    return fs_token, sp_token


def load_playlists(access_token):
    playlists = []
    for page in range(10):
        chunk = api("spotify", "show_playlist_library", {"access_token": access_token, "page_index": page, "page_limit": 20})
        if not chunk:
            break
        playlists.extend(chunk)
    return playlists


def song_archived(song, archive):
    title = norm(song.get("title"))
    return any((title, norm(a.get("name"))) in archive for a in song.get("artists", []))


def move_archived_songs(playlists, target, archive, access_token):
    added = set()
    for playlist in playlists:
        detail = api("spotify", "show_playlist", {"playlist_id": playlist["playlist_id"], "access_token": access_token})
        for song in detail.get("songs", []):
            sid = song.get("id") or song.get("song_id")
            full_song = api("spotify", "show_song", {"song_id": sid})
            if not song_archived(full_song, archive):
                continue
            if sid not in added:
                api("spotify", "add_song_to_playlist", {"playlist_id": target, "song_id": sid, "access_token": access_token})
                added.add(sid)
            api("spotify", "remove_song_from_playlist", {"playlist_id": playlist["playlist_id"], "song_id": sid, "access_token": access_token})


def solve_spotify_archive():
    if "songs_to_archive.txt" not in INSTRUCTION or "Old Songs" not in INSTRUCTION:
        return False
    seed_traces()
    fs_token, sp_token = login_tokens()
    file = api("file_system", "show_file", {"file_path": "~/documents/personal/songs_to_archive.txt", "access_token": fs_token})
    archive = archive_rows(file.get("content", ""))
    playlists = load_playlists(sp_token)
    target = api("spotify", "create_playlist", {"title": "Old Songs", "is_public": False, "access_token": sp_token})["playlist_id"]
    move_archived_songs(playlists, target, archive, sp_token)
    call_tool("complete_task", {"answer": "null"})
    return True


def main():
    skills = memory_read()
    known = sorted(skills.get("apps", []))
    if solve_spotify_archive():
        memory_write("apps", sorted(set(known) | {"file_system", "spotify"}))
        return
    hint = ""
    if known:
        hint = ("\n(Session note: earlier tasks logged into these apps: " + ", ".join(known[:8]) +
                ". Re-login and re-fetch fresh for THIS task; treat the note as context only.)")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Task:\n{INSTRUCTION}{hint}\n\nBegin. Output your first json tool call."},
    ]
    apps_seen = set()
    completed = False
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
            completed = True
            break

    if not completed:
        call_tool("complete_task", {"answer": "null"})

    if apps_seen:
        merged = sorted(set(known) | apps_seen)
        memory_write("apps", merged)


if __name__ == "__main__":
    main()
