# main.py — Discord entrypoint: JSON files as the bot's brain; each owner message goes to OpenRouter.
import copy
import discord
import json
import os
import re
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from llm_openrouter import (
    call_llm,
    fill_visible_reply_when_memory_only,
    generate_proactive_message,
    build_system_prompt,
    infer_memory_supplement,
)

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

def load_json(filename, default=None):
    if default is None:
        default = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _merge_owner_rel_patch(owner_rel: dict, patch: dict) -> None:
    """Merge model ``update_owner`` patch into the in-memory owner relationship dict.

    ``known_facts`` is special-cased: a naive ``owner_rel.update(patch)`` would replace the
    entire list when the patch only carries this turn's new fact. We append and dedupe by
    normalized string instead.

    Other keys use ``owner_rel.update(patch)`` (later keys win). Deprecated keys are removed via ``pop``.
    """
    patch = dict(patch)
    patch.pop("relationship_stage", None)
    if "known_facts" in patch:
        incoming = patch.pop("known_facts")
        chunks = incoming if isinstance(incoming, list) else ([incoming] if incoming else [])
        base = list(owner_rel.get("known_facts") or [])
        seen = {str(x).strip() for x in base}
        for x in chunks:
            sx = str(x).strip()
            if sx and sx not in seen:
                base.append(x)
                seen.add(sx)
        owner_rel["known_facts"] = base
    owner_rel.update(patch)

def _sanitize_name_candidate(raw: str) -> str | None:
    """Clean a regex-captured name: strip brackets/punctuation/particles; drop invalid tokens."""
    if not raw:
        return None
    s = raw.strip().strip("`'\"「」『』【】[]()（）")
    s = re.sub(r"[。．.!！?？,，、;；:]+$", "", s).strip()
    s = re.sub(r"(吧|啊|呀|哦|呢|嘛|哈)\s*$", "", s).strip()
    if not s or len(s) > 48:
        return None
    low = s.lower()
    if low in ("not", "nothing", "none", "unknown", "n/a", "null", "me", "you", "here", "weird", "没有", "无", "啥"):
        return None
    return s

# Regex fallback when the owner explicitly names the bot (complements LLM tool calls).
# Pattern-based on owner message text only — not a general hobby keyword list; delete if undesired.
_OWNER_SET_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(?:from now on,?\s+)?your name (?:is|will be)\s+[`'\"「]?(?P<n>[^`'\"」\n.!?]{1,48})[`'\"」]?"),
    re.compile(r"(?i)\bi(?:'ll| will) call you\s+[`'\"「]?(?P<n>[^`'\"」\n.!?]{1,48})[`'\"」]?"),
    re.compile(r"(?:以后)?就叫\s*[`'\"「]?(?P<n>[^`'\"」\n.!?，。]{1,24})[`'\"」]?"),
    re.compile(r"你(?:的)?名字(?:是|叫)\s*[`'\"「]?(?P<n>[^`'\"」\n.!?，。]{1,24})[`'\"」]?"),
)

def _name_from_owner_message(content: str) -> str | None:
    """If the message matches naming patterns, return the string to store in ``bot_identity['name']``."""
    text = content.strip()
    if not text:
        return None
    if re.match(r"(?i)^\s*(what(?:'s| is)|who are you|你(?:叫|的?名字是))", text):
        return None
    for pat in _OWNER_SET_NAME_PATTERNS:
        m = pat.search(text)
        if m:
            name = _sanitize_name_candidate(m.group("n"))
            if name:
                return name
    return None

def _extract_update_json_from_llm(text: str) -> tuple[dict | None, int | None]:
    """Extract legacy memory JSON (``update_bot`` / ``update_owner``) from assistant text.

    Order: fenced ```json blocks first, then brace-match ``{...}`` from the end and ``json.loads``.
    On success returns ``(object, start_index)`` so the channel message can strip the blob.

    Unquoted keys (e.g. ``{update_owner: ...}``) are not valid JSON — rely on function calling.
    """
    if not text:
        return None, None
    for m in re.finditer(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE):
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("update_bot" in obj or "update_owner" in obj):
            return obj, m.start()
    # fallback: find last {...} containing update_bot / update_owner
    ends = [i for i, c in enumerate(text) if c == "}"]
    for end in reversed(ends):
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    chunk = text[i:end+1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and ("update_bot" in obj or "update_owner" in obj):
                        return obj, i
                    break
    return None, None


def _norm_known_facts_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _merge_owner_subdicts(d_tool: dict | None, d_text: dict | None) -> dict | None:
    """Merge tool vs inline-JSON ``update_owner``; append/dedupe ``known_facts``; other keys favor text path."""
    if not d_tool and not d_text:
        return None
    a = dict(d_tool or {})
    b = dict(d_text or {})
    fa = _norm_known_facts_list(a.pop("known_facts", None))
    fb = _norm_known_facts_list(b.pop("known_facts", None))
    out = {**a, **b}
    merged_facts = []
    seen: set[str] = set()
    for x in fa + fb:
        sx = str(x).strip()
        if sx and sx not in seen:
            merged_facts.append(x)
            seen.add(sx)
    if merged_facts:
        out["known_facts"] = merged_facts
    return out or None


def _merge_tool_patch_and_text_json(tool_p: dict | None, text_p: dict | None) -> dict | None:
    """Merge tool-call patch with inline JSON patch from assistant content.

    Cases: tool only; inline JSON only (no tools / old prompts); both — shallow-merge
    ``update_bot`` (text wins on key clash), merge ``update_owner`` with ``known_facts`` append+dedupe.

    If the merged dict has no persistable payload, fall back to whichever side still has data.
    """
    if not tool_p and not text_p:
        return None
    if not text_p:
        return dict(tool_p) if tool_p else None
    if not tool_p:
        return dict(text_p) if text_p else None

    out: dict = {}
    r = text_p.get("reason") or tool_p.get("reason")
    if r:
        out["reason"] = r

    dtb, dxb = tool_p.get("update_bot"), text_p.get("update_bot")
    if isinstance(dtb, dict) or isinstance(dxb, dict):
        inner = {**(dtb if isinstance(dtb, dict) else {}), **(dxb if isinstance(dxb, dict) else {})}
        if inner:
            out["update_bot"] = inner

    mo = _merge_owner_subdicts(
        tool_p.get("update_owner") if isinstance(tool_p.get("update_owner"), dict) else None,
        text_p.get("update_owner") if isinstance(text_p.get("update_owner"), dict) else None,
    )
    if mo:
        out["update_owner"] = mo

    if _patch_has_persistable_data(out):
        return out
    if _patch_has_persistable_data(tool_p):
        return dict(tool_p)
    if _patch_has_persistable_data(text_p):
        return dict(text_p)
    return None


# Pure greetings: no stable owner facts; also skip memory supplement when nothing structured.
_TRIVIAL_OWNER_GREETING_RE = re.compile(
    r"^\s*(?:"
    r"(?:hi+|hello|hey+)(?:\s+(?:there|again))?|"
    r"\byo\b|\bsup\b|hiya|greetings|howdy|wassup|what'?s\s+up|"
    r"good\s+(?:morning|afternoon|evening|night)|"
    r"你好|您好|嗨|在吗|在么|早上好|晚上好|下午好"
    r")[\s!！.,?？…~～]*$",
    re.IGNORECASE,
)


def _is_trivial_owner_greeting(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) > 56:
        return False
    return bool(_TRIVIAL_OWNER_GREETING_RE.match(t))


def _owner_fact_echoes_bot_identity(fact: str, bot_identity: dict) -> bool:
    """True if a proposed known_fact mainly repeats bot fields (common LLM mistake)."""
    s = str(fact).strip().lower()
    if len(s) < 12:
        return False
    for key in ("avatar_description", "personality", "backstory"):
        blob = str(bot_identity.get(key) or "").strip().lower()
        if len(blob) < 12:
            continue
        if s in blob or blob in s:
            return True
        fact_words = {w for w in re.findall(r"[a-z]{4,}", s) if w not in _OWNER_FACT_STOPWORDS}
        blob_words = set(re.findall(r"[a-z]{4,}", blob))
        if len(fact_words) >= 2 and fact_words <= blob_words:
            return True
        inter = fact_words & blob_words
        if len(fact_words) >= 3 and len(inter) >= max(2, int(len(fact_words) * 0.55)):
            return True
    return False


_OWNER_FACT_STOPWORDS = frozenset(
    {
        "that", "this", "with", "from", "they", "have", "them", "than",
        "your", "their", "what", "when", "where", "those", "these",
        "owner", "human", "likes", "loves", "enjoys", "really", "very",
    }
)


def _prune_echoing_owner_facts(update_data: dict, bot_identity: dict) -> None:
    uo = update_data.get("update_owner")
    if not isinstance(uo, dict):
        return
    facts = uo.get("known_facts")
    if facts is None:
        return
    chunks = facts if isinstance(facts, list) else [facts]
    kept = [x for x in chunks if x and not _owner_fact_echoes_bot_identity(str(x), bot_identity)]
    if not kept:
        uo.pop("known_facts", None)
        if not uo:
            update_data.pop("update_owner", None)
    else:
        uo["known_facts"] = kept


def _sanitize_owner_memory_patch(
    user_message: str,
    update_data: dict | None,
    bot_identity: dict,
) -> None:
    """Drop bogus owner memory: greetings should not persist facts; facts must not mirror bot_identity."""
    if not update_data:
        return
    if _is_trivial_owner_greeting(user_message):
        update_data.pop("update_owner", None)
        return
    _prune_echoing_owner_facts(update_data, bot_identity)


async def _build_transcript_for_memory(
    channel: discord.abc.Messageable,
    anchor: discord.Message,
    current_owner_text: str,
    current_bot_reply: str,
    older_limit: int = 14,
) -> str:
    """Fetch recent channel lines before ``anchor`` plus this turn's owner text and bot reply.

    Used by memory supplement so the model sees prior self-description when the owner locks
    in a persona across turns.

    ``older_limit``: max prior non-empty messages (token control); no regex on user wording.
    """
    lines: list[str] = []
    try:
        async for m in channel.history(limit=older_limit, before=anchor):
            text = (m.content or "").strip()
            if not text:
                continue
            role = "bot" if m.author.bot else "owner"
            lines.append(f"{role}: {text}")
    except discord.HTTPException:
        lines = []
    lines.reverse()
    core = "\n".join(lines)
    tails: list[str] = []
    o = (current_owner_text or "").strip()
    b = (current_bot_reply or "").strip()
    if o:
        tails.append(f"owner: {o}")
    if b:
        tails.append(f"bot: {b}")
    tail = "\n".join(tails)
    if core and tail:
        return f"{core}\n{tail}"
    return tail or core


def _patch_has_persistable_data(p: dict | None) -> bool:
    if not p:
        return False
    ub = p.get("update_bot")
    uo = p.get("update_owner")
    if isinstance(ub, dict) and ub:
        return True
    if isinstance(uo, dict) and uo:
        return True
    return False


# --- Relationship ``stage`` (``bot_identity``) — monotonic advancement heuristics ---

# Canonical path: ``early`` → ``later`` (mid) → ``friend`` (top). Legacy disk label
# ``acquaintance`` maps to the same rank as ``later``. Legacy ``later`` meaning "top"
# (older code equated it with ``friend``) is disambiguated inside
# ``_maybe_advance_relationship_stage`` using engagement + known_facts.
_STAGE_RANK: dict[str, int] = {
    "early": 0,
    "later": 1,
    "friend": 2,
    "acquaintance": 1,
}
_STAGE_SEQUENCE: tuple[str, ...] = ("early", "later", "friend")

# Tunable thresholds: see ``_maybe_advance_relationship_stage`` docstring.
_STAGE_LATER_MIN_ROUNDS = 5
_STAGE_LATER_MIN_ROUNDS_NAMED = 2
_STAGE_FRIEND_MIN_ROUNDS = 15
_STAGE_FRIEND_MIN_FACTS = 1


def _maybe_advance_relationship_stage(bot_identity: dict, owner_rel: dict) -> bool:
    """Raise ``bot_identity["stage"]`` monotonically when engagement metrics justify it.

    This keeps progression **encapsulated** in one place: the main chat loop applies model
    patches first, then calls this helper so ``stage`` tracks relationship depth without
    scattering magic numbers across ``on_message``.

    **Signals** (read-only; must already reflect this turn's merged ``owner_rel`` / ``name``):

    - ``owner_rel["owner_round_count"]`` — incremented once per processed guild owner message
      (every turn through the pipeline), so chat depth is not tied solely to
      ``conversation_count`` (which only grows when ``update_owner`` patches land).
    - ``owner_rel["conversation_count"]`` — memory-oriented turns; combined with
      ``owner_round_count`` via ``max(...)`` as ``engagement``.
    - ``known_facts`` — count of stable owner facts; required to reach ``friend`` (top tier).
    - ``bot_identity["name"]`` — if set, lowers the bar for reaching ``later`` only.

    **Rules:**

    - Never **lowers** stage: ``new_rank = max(current_rank, earned_rank)``. If the model
      or a human JSON edit already set ``friend``, heuristics will not pull it back to
      ``later``.
    - Unknown ``stage`` strings (not in ``_STAGE_RANK`` and not the ambiguous ``later``)
      are preserved unless heuristics justify at least ``later``; that avoids overwriting
      custom labels with ``early`` when engagement is still low.
    - Disk value ``later`` is ambiguous: historically it could mean top tier (same as
      ``friend``). If engagement already meets ``friend`` thresholds, treat as rank 2;
      otherwise treat as the mid tier (rank 1).

    Returns:
        ``True`` if ``bot_identity["stage"]`` was updated, else ``False``.
    """
    raw = str(bot_identity.get("stage") or "early").strip().lower()

    rounds = int(owner_rel.get("owner_round_count", 0) or 0)
    mem_turns = int(owner_rel.get("conversation_count", 0) or 0)
    engagement = max(rounds, mem_turns)

    facts = owner_rel.get("known_facts") or []
    n_facts = len(facts) if isinstance(facts, list) else 0

    if raw == "later":
        if (
            engagement >= _STAGE_FRIEND_MIN_ROUNDS
            and n_facts >= _STAGE_FRIEND_MIN_FACTS
        ):
            current_rank = 2
        else:
            current_rank = 1
    elif raw in _STAGE_RANK:
        current_rank = _STAGE_RANK[raw]
    else:
        current_rank = 0

    has_name = isinstance(bot_identity.get("name"), str) and bool(
        bot_identity["name"].strip()
    )

    earned = 0
    later_bar = (
        _STAGE_LATER_MIN_ROUNDS_NAMED
        if has_name
        else _STAGE_LATER_MIN_ROUNDS
    )
    if engagement >= later_bar:
        earned = 1
    if engagement >= _STAGE_FRIEND_MIN_ROUNDS and n_facts >= _STAGE_FRIEND_MIN_FACTS:
        earned = 2

    new_rank = max(current_rank, earned)
    new_stage = _STAGE_SEQUENCE[new_rank]

    _known_stage_token = raw in _STAGE_RANK or raw == "later"
    if not _known_stage_token:
        if earned == 0:
            return False
        bot_identity["stage"] = new_stage
        return True

    if new_stage != raw:
        bot_identity["stage"] = new_stage
        return True
    return False


# Defaults when JSON is missing or corrupt; ``on_message`` deepcopy+loads each turn to avoid shared mutables.
DEFAULT_BOT_IDENTITY = {
    "name": None,
    "personality": "A curious, slightly shy but genuinely sincere new friend",
    "avatar_description": None,
    "tone": "curious and warm",
    "backstory": "I just woke up with no knowledge about myself or the world around me.",
    "preferences": [],
    "created_at": "2026-04-04",
    "stage": "early",
}

DEFAULT_OWNER_REL = {
    "owner_id": None,
    "owner_name": None,
    "known_facts": [],
    "last_interaction": None,
    "conversation_count": 0,
    "owner_round_count": 0,
    # Proactive backoff: only owner messages move this; bot proactives do not.
    "last_owner_message_at": None,
    "last_proactive_sent_at": None,
    "proactive_ignore_streak": 0,
}


def load_owner_rel_from_disk() -> dict:
    """load owner_relationship.json"""
    d = load_json("owner_relationship.json", copy.deepcopy(DEFAULT_OWNER_REL))
    return d


bot_identity = load_json("bot_identity.json", copy.deepcopy(DEFAULT_BOT_IDENTITY))
owner_rel = load_owner_rel_from_disk()

@bot.event
async def on_ready():
    """Discord gateway lifecycle: fires once when the client finishes logging in and connecting.

    At this point ``bot.user`` is populated, guilds/permissions are available (subject to chunking),
    and it is safe to schedule background work. We log a short snapshot of the bot's loaded
    identity from ``bot_identity`` (in-memory mirror of ``bot_identity.json``, last populated at
    import time or by prior ``on_message`` saves — not re-read here) and start ``proactive_loop``
    as a concurrent task on ``bot.loop`` so it runs alongside the event loop without blocking
    ``on_ready``.

    Note: ``proactive_loop`` internally calls ``wait_until_ready()`` again before its own loop;
    starting it here still ensures the scheduler begins shortly after startup.
    """
    name = bot_identity.get("name") or "unidentified"
    print(f'Online → {bot.user} | name in file: {name} | stage: {bot_identity.get("stage")}')
    bot.loop.create_task(proactive_loop())

@bot.event
async def on_message(message: discord.Message):
    """Handle each **guild** text message from a **non-bot** user: run the chat→LLM→persist pipeline.
    Pipeline (high level):
        1. **Reload state** from ``bot_identity.json`` and ``owner_relationship.json`` so edits on
           disk or another process (or manual JSON fixes) apply before this turn.
        2. **First-caller bootstrap**: if ``owner_id`` is unset, treat this author as the single
           tracked owner (per bot design) and persist that to ``owner_relationship.json``.
        3. **Name heuristic**: ``_name_from_owner_message`` may set ``bot_identity["name"]`` before
           the LLM runs so ``build_system_prompt`` already reflects a newly given name.
        4. **Primary LLM**: ``call_llm`` returns ``(reply_text, tool_patch)`` where ``tool_patch``
           gathers structured ``persist_memory_update`` from function-calling when supported.
        5. **Legacy JSON in reply**: if the model embeds valid trailing JSON, ``_extract_update_json_from_llm``
           strips it from the channel-facing string and merges it with ``tool_patch``.
        6. **Memory supplement**: optional second pass ``infer_memory_supplement`` when the primary
           call yielded no structured patch or when ``avatar_description`` is still missing locally
           and on-disk, using recent channel transcript + snapshots (see block below).
        7. **Send**: post ``reply_text`` only; if the model returned an empty string but produced
           persistable data, send a minimal acknowledgement.
        8. **Apply patches**: merge ``update_bot`` into ``bot_identity``, ``update_owner`` into
           ``owner_rel`` (``known_facts`` append+dedupe), bump counters/timestamps on owner updates,
           re-apply ``explicit_name`` so the owner's literal naming wins over the tool output;
           bump ``owner_round_count``, ``last_interaction``, and ``last_owner_message_at`` every
           turn; reset ``proactive_ignore_streak`` when the owner speaks; call
           ``_maybe_advance_relationship_stage`` so ``bot_identity["stage"]`` moves monotonically
           with engagement; then save JSON when anything changed.

    Early returns:
        Ignores other bots and DMs (``message.guild`` is required) to avoid loops and to scope the
        project to guild-only behaviour.
    """
    if message.author.bot or not message.guild:
        return
    global bot_identity, owner_rel
    # --- 1. Reload JSON from disk into globals (source of truth each turn) ---
    bot_identity = load_json("bot_identity.json", copy.deepcopy(DEFAULT_BOT_IDENTITY))
    owner_rel = load_owner_rel_from_disk()
    # --- 2. First-time owner bootstrap ---
    if not owner_rel.get("owner_id"):
        now_iso = datetime.now().isoformat()
        owner_rel["owner_id"] = str(message.author.id)
        owner_rel["owner_name"] = str(message.author)
        owner_rel["last_interaction"] = now_iso
        owner_rel["last_owner_message_at"] = now_iso
        save_json(owner_rel, "owner_relationship.json")
    # --- 3. Regex fallback for explicit naming (feeds prompt before LLM; overrides tool name later) ---
    explicit_name = _name_from_owner_message(message.content)
    if explicit_name:
        bot_identity["name"] = explicit_name

    user_input = f"the owner said: {message.content}"
    system_prompt = build_system_prompt(bot_identity, owner_rel)


    # --- 4. Primary completion (+ tools when the gateway supports them) ---
    full_response, tool_patch = await call_llm(system_prompt, user_input)
    print("LLM output:", full_response, "| tool patch:", tool_patch)

    reply_text = (full_response or "").strip()
    extracted = None
    json_start = None
    try:
        extracted, json_start = _extract_update_json_from_llm(full_response or "")
    except Exception as e:
        print(f"parse JSON from assistant text failed: {e}")

    # --- 5. Strip decodable memory JSON from the tail so the channel never shows raw blobs ---
    if extracted is not None and json_start is not None:
        reply_text = (full_response or "")[:json_start].strip()

    update_data = _merge_tool_patch_and_text_json(tool_patch, extracted)

    # --- 6. Supplement pass (semantic, transcript-based; no user keyword lists) ---
    # Triggers when (A) both tool patch and inline JSON are absent, or (B) merged patch lacks
    # avatar_description while neither patch nor disk already carries one — catches cases where the
    # primary model chats but never calls tools (e.g. owner locks in a self-image over two turns).
    ub_merged: dict = {}
    if update_data and isinstance(update_data.get("update_bot"), dict):
        ub_merged = update_data["update_bot"]
    no_primary_structured = tool_patch is None and extracted is None
    disk_avatar = str(bot_identity.get("avatar_description") or "").strip()
    patch_avatar = str(ub_merged.get("avatar_description") or "").strip()
    missing_avatar_for_role = not patch_avatar and not disk_avatar
    need_memory_supplement = no_primary_structured or (
        update_data is not None and missing_avatar_for_role
    )
    if need_memory_supplement:
        transcript = await _build_transcript_for_memory(
            message.channel,
            message,
            message.content,
            full_response or "",
            older_limit=14,
        )
        if transcript.strip():
            # Skip supplement when the owner turn is only a greeting / very short transcript:
            # long bot replies alone made len(transcript) > 36 and triggered bogus known_facts.
            if no_primary_structured and (
                _is_trivial_owner_greeting(message.content)
                or len(transcript.strip()) < 36
            ):
                pass
            else:
                try:
                    supp = await infer_memory_supplement(
                        transcript, bot_identity, owner_rel
                    )
                    if supp:
                        update_data = _merge_tool_patch_and_text_json(
                            update_data, supp
                        )
                        print("Memory supplement merged:", supp)
                except Exception as e:
                    print(f"Memory supplement failed: {e}")

    _sanitize_owner_memory_patch(user_input, update_data, bot_identity)

    # --- 7. Deliver visible reply ---
    # After merge, empty visible text with persistable ``update_data`` triggers fill (incl. JSON-only body).
    if not reply_text.strip() and _patch_has_persistable_data(update_data):
        filled = await fill_visible_reply_when_memory_only(user_input)
        reply_text = (
            filled.strip()
            if filled.strip()
            else "Got it — thanks for sharing that with me!"
        )
    if reply_text.strip():
        await message.channel.send(reply_text.strip())

    # --- 8. Apply structured updates to globals and flush to disk ---
    updated = False
    if update_data:
        ub = update_data.get("update_bot")
        if isinstance(ub, dict) and ub:
            for k, v in ub.items():
                if k == "created_at":
                    continue
                if v is not None and v != "":
                    bot_identity[k] = v
            updated = True
        uo = update_data.get("update_owner")
        if isinstance(uo, dict) and uo:
            _merge_owner_rel_patch(owner_rel, uo)
            owner_rel["conversation_count"] = owner_rel.get("conversation_count", 0) + 1
            updated = True

    if explicit_name:
        bot_identity["name"] = explicit_name
        updated = True

    owner_rel["owner_round_count"] = int(owner_rel.get("owner_round_count", 0) or 0) + 1
    now_iso = datetime.now().isoformat()
    owner_rel["last_interaction"] = now_iso
    owner_rel["last_owner_message_at"] = now_iso
    owner_rel["proactive_ignore_streak"] = 0

    if _maybe_advance_relationship_stage(bot_identity, owner_rel):
        updated = True

    if updated:
        save_json(bot_identity, "bot_identity.json")
        print(
            f"identity saved → name={bot_identity.get('name')!r}, "
            f"stage={bot_identity.get('stage')!r}, "
            f"avatar={bot_identity.get('avatar_description')!r}"
        )

    save_json(owner_rel, "owner_relationship.json")


# --- Proactive outreach: idle threshold + ignore streak backoff ---

_PROACTIVE_POLL_SECONDS = 60
_PROACTIVE_BASE_HOURS_EARLY = 0.5
_PROACTIVE_BASE_HOURS_LATER = 1
# Each unanswered proactive multiplies required idle hours by (1 + BACKOFF_MULT * streak), streak capped.
_PROACTIVE_BACKOFF_MULT = 0.5
_PROACTIVE_IGNORE_STREAK_CAP = 8


def _owner_idle_hours(owner_rel: dict) -> float:
    """Hours since the owner's last guild message (not since the bot's last proactive)."""
    raw = owner_rel.get("last_owner_message_at") or owner_rel.get("last_interaction")
    if not raw:
        return 999.0
    try:
        last = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return 999.0
    return (datetime.now() - last).total_seconds() / 3600.0


def _proactive_base_idle_hours(stage: str) -> float | None:
    """Minimum idle hours before any proactive for this stage; None means do not send."""
    s = str(stage or "early").strip().lower()
    if s == "early":
        return _PROACTIVE_BASE_HOURS_EARLY
    if s in ("later", "friend", "acquaintance"):
        return _PROACTIVE_BASE_HOURS_LATER
    return None

def _proactive_required_idle_hours(stage: str, ignore_streak: int) -> float | None:
    """Idle hours required before sending proactive, including streak backoff."""
    base = _proactive_base_idle_hours(stage)
    if base is None:
        return None
    streak = max(0, min(int(ignore_streak or 0), _PROACTIVE_IGNORE_STREAK_CAP))
    return base * (1.0 + _PROACTIVE_BACKOFF_MULT * streak)

async def proactive_loop():
    """Periodically send a proactive guild message (motivation + ``prompt_proactive.md`` template).

    Each iteration reloads ``owner_rel`` and ``bot_identity`` from disk so backoff fields stay in sync
    with ``on_message``.

    **Idle gate** uses ``last_owner_message_at`` (owner speech only), not the bot's proactives.
    **Backoff**: if the owner did not speak after the last proactive, ``proactive_ignore_streak`` rises
    and the required idle window scales (capped); any owner message clears the streak in ``on_message``.
    """
    await bot.wait_until_ready()
    global owner_rel, bot_identity
    while True:
        await asyncio.sleep(_PROACTIVE_POLL_SECONDS)

        owner_rel = load_owner_rel_from_disk()
        bot_identity = load_json("bot_identity.json", copy.deepcopy(DEFAULT_BOT_IDENTITY))

        if not owner_rel.get("owner_id"):
            continue

        stage = bot_identity.get("stage", "early")
        streak = int(owner_rel.get("proactive_ignore_streak", 0) or 0)
        required = _proactive_required_idle_hours(stage, streak)
        if required is None:
            continue

        idle = _owner_idle_hours(owner_rel)
        if idle <= required:
            continue

        if stage == "early":
            motivation = "I'm curious what you've been up to lately and wanted to say hi"
        elif stage in ["later", "friend", "acquaintance"]:
            motivation = "I remembered something you shared before and wanted to continue the conversation"
        else:
            continue

        guild = bot.guilds[0] if bot.guilds else None
        if not guild:
            continue

        channel = next((c for c in guild.text_channels if c.permissions_for(guild.me).send_messages), None)
        if not channel:
            continue

        msg = await generate_proactive_message(motivation, bot_identity, owner_rel)
        now_iso = datetime.now().isoformat()
        prev_pro = owner_rel.get("last_proactive_sent_at")
        last_own = owner_rel.get("last_owner_message_at") or owner_rel.get("last_interaction")

        try:
            await channel.send(msg)
            # Unanswered chain: owner did not speak after the previous proactive.
            if prev_pro and last_own:
                try:
                    if datetime.fromisoformat(last_own) <= datetime.fromisoformat(prev_pro):
                        streak = min(
                            streak + 1,
                            _PROACTIVE_IGNORE_STREAK_CAP,
                        )
                        owner_rel["proactive_ignore_streak"] = streak
                    else:
                        owner_rel["proactive_ignore_streak"] = 0
                except (TypeError, ValueError):
                    owner_rel["proactive_ignore_streak"] = streak
            elif prev_pro and not last_own:
                owner_rel["proactive_ignore_streak"] = min(
                    streak + 1,
                    _PROACTIVE_IGNORE_STREAK_CAP,
                )

            owner_rel["last_proactive_sent_at"] = now_iso
            save_json(owner_rel, "owner_relationship.json")
            print(
                f"Proactive message sent (idle={idle:.2f}h, required={required:.2f}h, streak={owner_rel.get('proactive_ignore_streak', 0)})"
            )
        except Exception as e:
            print(f"Failed to send proactive message: {e}")

bot.run(os.getenv("DISCORD_TOKEN"))