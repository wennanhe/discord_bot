# Discord Relationship Bot

A Discord bot that builds a **relationship with one owner** through natural chat: it learns identity and stable facts over time, persists them to JSON files, and can **reach out proactively** with backoff when ignored.

---

## Video walkthrough

Three short screen recordings cover:

1. **Project architecture** — how the pieces fit together  
2. **Feature demo** — onboarding, memory, proactive messages  
3. **Code & files** — main modules, prompts, and `bot_identity.json` / `owner_relationship.json`

---

## Getting started

### Prerequisites

- **Python 3.10+** (version aligned with your `discord.py` install)  
- A **Discord application / bot token**  
- An **OpenRouter API key** (OpenAI-compatible client in code)

### Install

```bash
pip install -r requirements.txt
```

### Configuration

Create a **`.env`** file in the project root (do **not** commit it):

| Variable | Required | Purpose |
|----------|----------|---------|
| `DISCORD_TOKEN` | Yes | Discord bot token |
| `LLM_API_KEY` | Yes | OpenRouter API key |
| `LLM_MODEL` | No | Chat model (default: `openrouter/auto`) |
| `LLM_SUPPLEMENT_MODEL` | No | Optional model for the memory supplement pass |
| `LLM_REPLY_FILL_MODEL` | No | Optional model when filling an empty visible reply after a tool-only turn |

### Run the bot

```bash
python main.py
```

Invite the bot to a guild, grant **Message Content** intent if needed, and send messages in a channel the bot can read. The **first** non-bot user who is processed becomes the tracked owner (`owner_relationship.json`).

### Optional: fresh state from templates

To reset or bootstrap files without hand-editing:

- Copy `bot_identity_default.json` → `bot_identity.json`  
- Copy `owner_relationship_default.json` → `owner_relationship.json`

---

## Secrets and GitHub

**Do not put real tokens or API keys in the repository.** GitHub blocks and flags secret leaks; keeping credentials out of git is also basic hygiene. Use a **`.env`** file locally and list `.env` in **`.gitignore`**.

If you clone this repo, you must supply your own `DISCORD_TOKEN` and `LLM_API_KEY` (for reviewers, tokens can be shared through a separate, private channel—not in the repo).

---

## How the implementation meets `description.txt`

Brief mapping from product requirements to this codebase.

### Scenario: relationship from scratch

- **Cold start:** Defaults live in `DEFAULT_BOT_IDENTITY` and `DEFAULT_OWNER_REL` in `main.py` (mirrored by `bot_identity_default.json` / `owner_relationship_default.json`). The bot starts with generic self-text and no owner facts until someone talks.
- **Single tracked owner:** The first guild user who triggers `on_message` gets written into `owner_relationship.json` (`owner_id`, `owner_name`, timestamps).

### 1. Conversational onboarding

- **No forms:** There is no survey UI. Every turn is normal Discord chat; the system prompt is built from `prompt_dialogue.md` via `build_system_prompt()` in `llm_openrouter.py`.
- **Name, personality, avatar over time:** The model is instructed to call `persist_memory_update` with `update_bot` (e.g. `name`, `personality`, `avatar_description`, `tone`, `backstory`). A regex path `_name_from_owner_message()` in `main.py` can also set the name from clear phrasing before/after the LLM.
- **Pacing:** Prompts emphasize curiosity and natural rhythm, not interrogation; we do not hard-code question flows.

### 2. Persistent identity (files, not a DB)

- **`bot_identity.json`:** Bot-side fields (name, personality, `avatar_description`, tone, backstory, preferences, `stage`, etc.). Loaded each turn in `on_message` and saved when updates apply.
- **`owner_relationship.json`:** Owner id/name, `known_facts`, interaction counters, proactive timestamps, and `proactive_ignore_streak`.
- **Context for generation:** `build_system_prompt()` injects JSON snapshots of both into the dialogue prompt so behavior evolves instead of using one static prompt.

### 3. Proactive behavior

- **Started at connect:** `on_ready()` schedules `proactive_loop()` (`main.py`).
- **When:** The loop polls periodically, reloads JSON from disk, checks idle time since **`last_owner_message_at`** (owner speech only—not the bot’s own messages).
- **Stage-based frequency:** `_proactive_base_idle_hours()` uses `bot_identity["stage"]` (`early` vs `later` / `friend` / `acquaintance`) for a shorter vs longer minimum idle window (tunable via `_PROACTIVE_BASE_HOURS_*`).
- **Motivation copy:** Early vs later uses different `motivation` strings before filling `prompt_proactive.md` and calling `generate_proactive_message()`.
- **Backoff:** If a proactive was sent and the owner still has not spoken since then, `proactive_ignore_streak` increases (capped). Required idle time scales with streak (`_proactive_required_idle_hours`). Any owner message in `on_message` resets the streak to `0`.

### 4. Memory that feels natural

- **Not full transcripts:** We persist structured snippets, not the whole channel log. `known_facts` holds short natural sentences about the owner; the model is steered away from mechanical phrasing in `prompt_dialogue.md`.
- **Judgment:** Instructions tell the model what to store vs skip (stable preferences vs one-off small talk). Optional `infer_memory_supplement()` can propose patches from a short recent transcript when the main call missed tools/JSON.
- **Guards:** `_sanitize_owner_memory_patch()` drops owner patches on pure greetings and removes `known_facts` that mostly echo `bot_identity` text (e.g. avatar), reducing false owner memories.

### Implementation constraint: two persistent files

- Satisfied by **`bot_identity.json`** (bot) and **`owner_relationship.json`** (owner / relationship). Both are read before each reply and written after applicable updates.

### Failure handling (evaluation criteria)

- **LLM / tools:** `call_llm()` tries tool calling first; on gateway errors that look tool-related, it **retries once without tools** so inline JSON in the reply can still be merged (`llm_openrouter.py`).
- **Empty channel text but saved memory:** `fill_visible_reply_when_memory_only()` generates a short visible reply.
- **Memory supplement errors:** Logged and skipped; the turn still uses the main reply.
- **Proactive send errors:** Caught and printed; the loop continues.
- **Ignored bot:** Handled by proactive backoff (streak + longer idle), reset when the owner speaks again.

---

## Current limitations & future work

### Short-term memory (planned improvement)

Today the bot **does not** keep a rolling **chat transcript** as context. Each reply is driven mainly by:

- The **system prompt** (persona + JSON snapshots of identity and owner relationship)  
- A **single user message** for the current line (e.g. `the owner said: …`)

So the model does **not** see prior turns in the same channel inside the main chat completion. Long-term flavor comes from **structured memory** (`known_facts`, `update_bot`, etc.) and optional **memory supplement** logic that peeks at recent channel history for patching—not full multi-turn context in the primary LLM call.

**Next optimization:** add **short-term memory** (e.g. last *N* user/assistant pairs, or a running summary) and inject that into each request so follow-ups and references to the immediate conversation are grounded without stuffing the whole channel into the prompt.

---

## Repository layout (quick reference)

| Path | Role |
|------|------|
| `main.py` | Discord `on_message`, JSON load/save, proactive loop |
| `llm_openrouter.py` | OpenRouter client, `call_llm`, memory supplement, proactive template |
| `prompt_dialogue.md` | Main dialogue system prompt template |
| `prompt_proactive.md` | Proactive message template |
| `description.txt` | Original product requirements |
| `bot_identity.json` / `owner_relationship.json` | Runtime state (created/updated by the bot) |
