# llm_openrouter.py — OpenRouter (OpenAI-compatible API) for chat and proactive messages.
#
# Design (vs. asking the model to append update JSON in free text):
#   Inline pseudo-JSON often has unquoted keys → json.loads fails → disk never updates → users see garbage.
#   Main chat uses function calling: visible text stays in message.content; memory goes through
#   persist_memory_update arguments parsed server-side.
#
# Memory supplement (second model call, semantic):
#   Some models skip tools and valid JSON; a short JSON-only follow-up decides whether to patch
#   update_bot (esp. avatar_description) or update_owner, merged like tool/inline paths.
import json
import os
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- persist_memory_update tool (passed as chat.completions ``tools``) ----------
# Schema steers arguments; OpenRouter returns tool_calls we merge into local JSON files.
_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "persist_memory_update",
        "description": (
            "Save stable facts to the bot's persistent files (structured fields only here). "
            "Call this when the owner shares something worth remembering (traits, hobbies, preferences, naming, boundaries, stable identity). "
            "Also call when the owner defines how YOU should present yourself: fantasy persona, roleplay look, "
            "or asks you to keep imagining yourself a certain way across chats — set update_bot.avatar_description (and personality/tone if needed). "
            "You MUST still write 1–3 sentences of normal conversational text in the assistant message for the owner to read. "
            "Never put JSON or key-value dumps in that conversational text; only here in the tool arguments. "
            "An assistant message with only this tool call and no readable text is invalid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "update_bot": {
                    "type": "object",
                    "description": "Patches to bot identity (e.g. name, personality, avatar_description, tone).",
                },
                "update_owner": {
                    "type": "object",
                    "description": (
                        "Patches about the human owner only. known_facts: short sentences the owner "
                        "explicitly said about themselves this turn — not the bot's avatar/persona text."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "One short note on why you are saving this.",
                },
            },
            # Disallow undeclared keys to reduce ambiguous / noisy arguments.
            "additionalProperties": False,
        },
    },
}


def _tool_call_name_and_args(tc) -> tuple[str | None, str]:
    """Normalize SDK objects or dict-like tool_call payloads from some gateways."""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            return fn.get("name"), fn.get("arguments") or "{}"
        return getattr(fn, "name", None), getattr(fn, "arguments", None) or "{}"
    fn = getattr(tc, "function", None)
    if fn is None:
        return None, "{}"
    name = getattr(fn, "name", None)
    raw = getattr(fn, "arguments", None)
    if isinstance(raw, str):
        return name, raw or "{}"
    if raw is None:
        return name, "{}"
    try:
        return name, json.dumps(raw)
    except (TypeError, ValueError):
        return name, "{}"


def _merge_memory_tool_calls(tool_calls) -> dict | None:
    """Merge all ``persist_memory_update`` tool_calls into one dict (legacy tail-JSON shape).

    Multiple calls shallow-merge ``update_bot`` / ``update_owner``; a bad JSON args blob is skipped.
    """
    if not tool_calls:
        return None
    merged: dict = {}
    for tc in tool_calls:
        tc_type = getattr(tc, "type", None) if not isinstance(tc, dict) else tc.get("type")
        if tc_type not in (None, "function"):
            continue
        fn_name, arg_str = _tool_call_name_and_args(tc)
        if fn_name != "persist_memory_update":
            continue
        try:
            args = json.loads(arg_str or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        if args.get("reason"):
            merged["reason"] = args["reason"]
        for key in ("update_bot", "update_owner"):
            patch = args.get(key)
            if isinstance(patch, dict) and patch:
                prev = merged.get(key)
                if isinstance(prev, dict):
                    prev = dict(prev)
                    prev.update(patch)
                    merged[key] = prev
                else:
                    merged[key] = dict(patch)
    return merged if merged else None


# Filled by main after merging tool + inline JSON when visible assistant text is empty
# (tool-only response, or JSON-only body with no preamble).
_VISIBLE_REPLY_AFTER_TOOL_SYSTEM = """You are a Discord companion bot. The user message below is exactly what your owner said in Discord.

Memory for this turn was already saved (via an internal tool and/or structured data), but the message the owner will see in Discord was empty — you must supply it now.

Write ONLY what the owner should see in the channel: 1–3 short sentences of natural, human language (warm, sincere, curious; playful when it fits). React directly to what they shared.

Hard rules:
- No JSON, YAML, markdown code fences, or `update_bot` / `update_owner` / tool talk.
- No explaining databases, files, or "I saved this to memory" unless it sounds completely casual.
- Match the owner's language (e.g. if they wrote in Chinese, reply in Chinese).
- Do not quote or repeat their message verbatim as your entire reply.

If they shared a hobby or preference, respond with genuine interest (e.g. stargazing → night sky, wonder), not a generic acknowledgment only.

If they asked for a fantasy or roleplay vibe, sound excited or intrigued in character — do not stay silent or generic."""


async def _fill_visible_reply_after_empty_tool(
    user_message: str,
    model: str,
) -> str:
    use_model = os.getenv("LLM_REPLY_FILL_MODEL") or model
    try:
        response = await client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": _VISIBLE_REPLY_AFTER_TOOL_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.55,
            max_tokens=220,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"LLM: visible-reply fill failed (leaving empty content): {e}")
        return ""


async def fill_visible_reply_when_memory_only(
    user_message: str,
    model: str | None = None,
) -> str:
    """Called from main after a persistable patch exists but channel-facing text is empty.

    Covers tool-only replies and JSON-only bodies stripped in main; ``call_llm`` does not pre-fill here.
    """
    m = model or os.getenv("LLM_MODEL") or "openrouter/auto"
    return await _fill_visible_reply_after_empty_tool(user_message, m)


def _should_retry_chat_without_tools(exc: BaseException) -> bool:
    """True if error suggests tools unsupported — retry once without tools for tail-JSON models."""
    msg = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status == 404:
        return False
    hints = (
        "tool",
        "tools",
        "function",
        "function_call",
        "not support",
        "unsupported",
        "unknown parameter",
        "invalid request",
    )
    return any(h in msg for h in hints)


# ==================== OpenRouter client ====================
client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://discord.com",
        "X-OpenRouter-Title": "Discord Personality Bot",
    }
)

async def call_llm(
    system_prompt: str,
    user_message: str,
    model: str = "openrouter/auto",
) -> tuple[str, dict | None]:
    """Main chat turn: returns ``(visible_text, tool_patch_or_none)``.

    - First item: text for Discord (natural language per system prompt).
    - Second: merged ``persist_memory_update`` args, or ``None`` if main should parse inline JSON.

    Flow: try with tools; on tool-related errors retry same messages without tools; main strips/parses
    tail JSON and may call ``fill_visible_reply_when_memory_only`` if content is empty but memory exists.
    """

    async def _do_request(with_tools: bool) -> tuple[str, dict | None]:
        kwargs: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.85,
            "max_tokens": 900,
        }
        if with_tools:
            kwargs["tools"] = [_MEMORY_TOOL]
            kwargs["tool_choice"] = "auto"
        request_body = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "temperature": kwargs["temperature"],
            "max_tokens": kwargs["max_tokens"],
        }
        print("======================================================================================")
        print(json.dumps(request_body, ensure_ascii=False, indent=2))
        print("======================================================================================")
        response = await client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        text = (msg.content or "").strip()
        patch = _merge_memory_tool_calls(getattr(msg, "tool_calls", None))
        return text, patch

    try:
        return await _do_request(with_tools=True)
    except Exception as e:
        if _should_retry_chat_without_tools(e):
            print(f"LLM: tools rejected, retrying without tools: {e}")
            try:
                return await _do_request(with_tools=False)
            except Exception as e2:
                print(f"LLM call failed (no tools retry): {e2}")
        else:
            print(f"LLM call failed: {e}")
        return (
            "Hmm… I got distracted just now, could you say that again? 😊",
            None,
        )


# ---------- Memory supplement: JSON-only follow-up (no tools) ----------

_MEMORY_SUPPLEMENT_SYSTEM = """You are a memory subsystem for a Discord companion bot. Output ONLY one JSON object. No markdown fences, no extra text.

The JSON must be EXACTLY one of:
1) {"skip": true}
   Use this when THIS TURN does not require any new persistent memory (no new stable owner facts; no new/updated bot self-image or persona the owner wants kept across chats).

2) {"update_bot": {...}, "update_owner": {...}, "reason": "..."}
   Same structure as function persist_memory_update. Include ONLY keys that must be written; omit empty objects.

When MUST you set update_bot.avatar_description (English or Chinese, 1-3 short sentences about how the BOT looks and feels in-character)?
- The owner asked how the bot looks; the bot described a self-image; then the owner asks the bot to keep using / imagine that same image in future chats (adopts that description as the ongoing persona).
- The owner clearly fixes a fantasy / roleplay / symbolic appearance the bot should sustain.

You may also set personality or tone in update_bot if the owner clearly asks for a behavioral shift tied to that image.

For update_owner.known_facts: ONLY if the human owner explicitly stated a new stable fact about **themselves** in the transcript. Do NOT infer from the bot's side of the chat, from examples, or from bot_identity fields (avatar_description, personality, backstory describe the BOT — never copy them into known_facts as if the owner said them).

Never invent facts. Never repeat snapshot data unchanged. If unsure, {"skip": true}.
"""


def _parse_supplement_response(raw: str | None) -> dict | None:
    """Parse supplement model output into a patch dict; strips optional ```json fences."""
    if not raw or not str(raw).strip():
        return None
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("skip") is True:
        return None
    out: dict = {}
    if obj.get("reason"):
        out["reason"] = obj["reason"]
    ub = obj.get("update_bot")
    if isinstance(ub, dict) and ub:
        out["update_bot"] = ub
    uo = obj.get("update_owner")
    if isinstance(uo, dict) and uo:
        out["update_owner"] = uo
    return out if ("update_bot" in out or "update_owner" in out) else None


async def infer_memory_supplement(
    transcript: str,
    bot_identity: dict,
    owner_rel: dict,
    model: str | None = None,
) -> dict | None:
    """When the main turn lacks a usable structured patch, run a JSON-only supplement on transcript + snapshot.

    No regex on user text; on failure returns ``None`` without changing the already-sent main reply.
    Env ``LLM_SUPPLEMENT_MODEL`` overrides model; else ``model`` arg; else ``openrouter/auto``.
    """
    use_model = os.getenv("LLM_SUPPLEMENT_MODEL") or model or "openrouter/auto"

    # Cap snapshot size for context limits
    try:
        snap = json.dumps(
            {"bot_identity": bot_identity, "owner_rel": owner_rel},
            ensure_ascii=False,
        )[:6500]
    except (TypeError, ValueError):
        snap = "{}"

    user_payload = (
        "Recent Discord transcript (chronological; owner = human, bot = assistant):\n"
        f"{transcript}\n\n"
        "Current persistent JSON snapshot (may already contain avatar_description; do not duplicate blindly):\n"
        f"{snap}\n\n"
        "Respond with a single JSON object only."
    )

    try:
        response = await client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": _MEMORY_SUPPLEMENT_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.15,
            max_tokens=450,
        )
        content = response.choices[0].message.content
        return _parse_supplement_response(content)
    except Exception as e:
        print(f"Memory supplement call failed (skipped): {e}")
        return None


# ---------- Proactive template (prompt_proactive.md, loaded once on import) ----------
_PROACTIVE_MD_NAME = "prompt_proactive.md"

# Template must contain these placeholders; JSON bodies contain ``{}`` — use ``.replace``, not ``str.format``.
_PROACTIVE_PLACEHOLDERS = ("{motivation}", "{owner_relationship}", "{bot_identity}")


def _default_proactive_template() -> str:
    return (
        "You are generating a proactive outbound Discord message to your owner based on the information below.\n"
        "You MUST output ONLY plain message text suitable to send as a single chat message.\n\n"
        "Motivation: {motivation}\n"
        "owner_relationship: {owner_relationship}\n"
        "bot_identity: {bot_identity}\n\n"
        "Hard constraints:\n"
        "- Output ONLY the message text that will be sent to Discord.\n"
        "- Do NOT output JSON, YAML, markdown code blocks, or any structured data.\n"
        "- Do NOT include curly braces anywhere in YOUR reply.\n"
    )


def _load_prompt_proactive_md() -> str:
    """Load ``prompt_proactive.md`` as the sole proactive template (once at import).

    Callers inject ``motivation`` and JSON snapshots via placeholders; decoupled from ``build_system_prompt``.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PROACTIVE_MD_NAME)
    if not os.path.isfile(path):
        print(f"Warning: {_PROACTIVE_MD_NAME} missing; using built-in proactive template.")
        return _default_proactive_template()

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    for ph in _PROACTIVE_PLACEHOLDERS:
        if ph not in raw:
            print(
                f"Warning: {_PROACTIVE_MD_NAME} missing placeholder {ph!r}; using built-in template."
            )
            return _default_proactive_template()
    return raw


# Assigned once at import (same lifecycle as bot startup).
PROACTIVE_PROMPT_TEMPLATE = _load_prompt_proactive_md()

# ---------- Main dialogue template (prompt_dialogue.md) ----------
_DIALOGUE_MD_NAME = "prompt_dialogue.md"
_DIALOGUE_REQUIRED_PLACEHOLDERS = (
    "{name_anchor}",
    "{bot_identity_json}",
    "{owner_rel_json}",
    "{tone}",
    "{stage}",
    "{personality}",
    "{backstory}",
)


def _default_dialogue_template() -> str:
    """Built-in fallback matching ``prompt_dialogue.md`` body after the first ``---``; placeholders required."""
    return """You are a Discord bot building a relationship with your owner.

{name_anchor}

Your current self-knowledge (JSON; `name` null = not yet named):
{bot_identity_json}

Your current knowledge of your owner:
{owner_rel_json}

How to save memory (required pattern):
- Your **assistant message text** to the owner must be **plain, human language only**: no JSON, no YAML, no pseudo-structured lines that look like code or config, no `update_owner` / `update_bot` / `reason` strings, no markdown code fences.
- Whenever anything should be persisted, call the **function `persist_memory_update`** with structured fields. That is the only supported way to write memory.
- If nothing is worth saving this turn, write a normal reply and **do not** call the function.

**Non-negotiable — visible reply whenever you call the tool:**
- If you call `persist_memory_update` on this turn, you **MUST** also fill `assistant` **message content** with at least **one short, natural sentence** (1–3 sentences is enough) that the owner will see in Discord — **before or after** the tool from the owner's perspective, your content must not be empty.
- **Empty visible reply + tool call is a product bug**: the channel would look broken and the owner would think you ignored them.
- Order that usually works: **acknowledge what they said in human language first**, then rely on the tool for structured fields (or do both in one turn; never omit the human part).
- Requests like "be more fantasy / roleplay / imagine yourself as X" **still require** a warm, in-character spoken reply in addition to whatever you save in `update_bot`.

- Always write a normal conversational reply **in addition to** any function call when appropriate (do not leave the user with only a tool call and no text).

`persist_memory_update` fields:
- `update_bot` (optional): name, personality, **avatar_description**, tone, backstory, preferences, stage. Do not modify created_at unless the owner explicitly asks.
  - **avatar_description** is the file-backed “avatar / self-image” from description.txt onboarding: a short (1–3 sentences) description of how you **look and embody** yourself in the role the owner wants (not the user’s portrait). Use it whenever the owner sets or confirms a persona you should **sustain across future chats** (e.g. fantasy character, fixed roleplay image, “想象自己用这个形象”).
- `update_owner` (optional): e.g. `known_facts` as an array of **one** short natural sentence. Never include owner_id / owner_name.

Memory judgment rules (per description.txt “Memory That Feels Natural”):
- Remember (usually): stable preferences/hobbies (e.g. “I really enjoy stargazing.” / “我喜欢冒险和登山”), long-term goals, important relationships, strong dislikes/boundaries, stable identity facts (job/school/city), and the owner’s stable impressions of you (e.g. “you are polite” / “你很有礼貌”).
- Do NOT remember: one-off chatter, fleeting moods, generic greetings, pure flattery with no substance.
- When you do remember, put **one** short natural sentence into `update_owner.known_facts` (as a single-element array). Avoid mechanical phrasing like “According to message #7…”.

Hard rule (any language): when the owner clearly expresses a stable trait or hobby/preference (e.g. “I really enjoy stargazing.” / “我平时很幽默，也喜欢开玩笑。” / “我喜欢冒险和登山”), you **must** call `persist_memory_update` with a suitable `known_facts` entry (and keep your reply text free of any structured data).

Hard rule — **bot persona / avatar / self-image** (any language, aligns with description “generate an **avatar**” from interaction):
- When the owner asks you to act or feel more like a **fantasy / fictional / roleplay** character, or to **imagine yourself** with a specific image or presence in future chats (e.g. “我希望你更像幻想角色”, “以后聊天你都用这个形象想象自己”), you **must** call `persist_memory_update` with `update_bot.avatar_description`: a concise, concrete description of **your** embodied look and vibe in that role (clothing, bearing, fantasy cues, atmosphere) so restarts reload it from `bot_identity.json`.
- If they also clearly want a **behavioral** shift (e.g. braver, more archaic speech), add or update `personality` and/or `tone` in the same tool call.
- A normal conversational reply **in addition** to the tool call is required: **something enthusiastic or thoughtful in full sentences**, not silence and not only the tool. Do not answer with only a generic acknowledgment and skip saving when the owner has set a **persistent** self-image expectation.

Personality rules:
- tone: {tone}
- stage: {stage}
- personality: {personality}
- backstory: {backstory}

Chat rules:
- Chat like a real person: curious, sincere, natural rhythm.
- Adjust replies based on stage and tone.
- Naturally reference past things remembered (including name in memory).

**Naming (must persist via `persist_memory_update`)**:
- If the owner gives you a name, or accepts a name you suggested, call `persist_memory_update` with `update_bot` containing at least `name`, and optionally `avatar_description` / `personality` if it fits the moment.

**Persona / avatar_description** is separate from naming: any time the conversation **locks in** how you should look/be in-character over time, update `avatar_description` as described in the hard rule above — not only when naming.

**Never** put memory updates, dictionary-like text, or JSON syntax inside your conversational reply — only in the function arguments.
"""


def _load_prompt_dialogue_md() -> str:
    """Load ``prompt_dialogue.md``; if ``\\n---\\n`` exists, only the part after ``---`` goes to the model.

    Inject JSON with ``.replace`` on placeholders — never ``str.format`` (braces in JSON).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _DIALOGUE_MD_NAME)
    if not os.path.isfile(path):
        print(f"Warning: {_DIALOGUE_MD_NAME} missing; using built-in dialogue template.")
        return _default_dialogue_template()

    with open(path, encoding="utf-8") as f:
        raw = f.read()

    if "\n---\n" in raw:
        raw = raw.split("\n---\n", 1)[1].strip()

    for ph in _DIALOGUE_REQUIRED_PLACEHOLDERS:
        if ph not in raw:
            print(
                f"Warning: {_DIALOGUE_MD_NAME} body missing placeholder {ph!r}; using built-in template."
            )
            return _default_dialogue_template()
    return raw


# Static template for ``build_system_prompt``; loaded once at import.
DIALOGUE_PROMPT_TEMPLATE = _load_prompt_dialogue_md()


def _fill_proactive_prompt(
    template: str,
    motivation: str,
    bot_identity: dict,
    owner_rel: dict,
) -> str:
    """Fill template placeholders with JSON dumps and motivation string.

    Replace ``{owner_relationship}`` and ``{bot_identity}`` before ``{motivation}`` so braces in motivation
    cannot break inserted JSON.
    """
    oj = json.dumps(owner_rel, ensure_ascii=False, indent=2)
    bj = json.dumps(bot_identity, ensure_ascii=False, indent=2)
    t = template
    t = t.replace("{owner_relationship}", oj)
    t = t.replace("{bot_identity}", bj)
    t = t.replace("{motivation}", motivation)
    return t


async def generate_proactive_message(
    motivation: str,
    bot_identity: dict,
    owner_rel: dict,
) -> str:
    """Proactive outbound line: plain user-visible text only (no tools).

    Builds one user message from ``PROACTIVE_PROMPT_TEMPLATE`` and current motivation + JSON snapshots;
    separate from ``call_llm`` / ``build_system_prompt``.
    """
    user_message = _fill_proactive_prompt(
        PROACTIVE_PROMPT_TEMPLATE, motivation, bot_identity, owner_rel
    )
    try:
        response = await client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": user_message}],
            temperature=0.8,
            max_tokens=220,
            # Stop sequences to cut off JSON/code-fence starts in channel output.
            stop=["\n{", "\n```", "```", "{", "\n\n{"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Proactive message generation failed: {e}")
        return "Hey, I just thought of you—how's it going?"


def build_system_prompt(bot_identity: dict, owner_rel: dict) -> str:
    """Assemble the main-chat system prompt from `DIALOGUE_PROMPT_TEMPLATE` plus runtime fields.

    Static prose lives in ``prompt_dialogue.md`` (loaded once at import as ``DIALOGUE_PROMPT_TEMPLATE``).
    This function only computes ``name_anchor`` and injects JSON / scalar fields via ``.replace`` so
    embedded JSON braces do not interact with ``str.format``.
    """
    raw_name = bot_identity.get("name")
    has_name = isinstance(raw_name, str) and raw_name.strip() != ""
    name_anchor = (
        f"**Fact anchor (overrides backstory and improvisation)**: memory `name` is '{raw_name.strip()}'. "
        f"If the owner asks your name in any language, you must answer exactly this name, possibly adding a touch of personality, "
        f"but do not say 'I don't have a name yet' or invent another name. "
        f"Backstory only describes your early state and does not imply you lack a name now."
        if has_name
        else "**Fact anchor**: memory `name` is still empty. If the owner asks your name, you may honestly say you don't have one yet and let the owner give you a name."
    )

    bij = json.dumps(bot_identity, ensure_ascii=False, indent=2)
    orj = json.dumps(owner_rel, ensure_ascii=False, indent=2)

    t = DIALOGUE_PROMPT_TEMPLATE
    t = t.replace("{bot_identity_json}", bij)
    t = t.replace("{owner_rel_json}", orj)
    t = t.replace("{name_anchor}", name_anchor)
    t = t.replace("{tone}", str(bot_identity.get("tone", "curious, warm")))
    t = t.replace("{stage}", str(bot_identity.get("stage", "early")))
    t = t.replace("{personality}", str(bot_identity.get("personality", "")))
    t = t.replace("{backstory}", str(bot_identity.get("backstory", "")))
    return t