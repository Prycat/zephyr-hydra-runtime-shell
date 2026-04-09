# Obsidian Plugin Design — Hermes Knowledge Agent
**Date:** 2026-04-08
**Status:** Approved

## Overview

An Obsidian plugin that turns the local Hermes 3 + vLLM + TurboQuant stack into an in-Obsidian knowledge engine. The plugin handles server control, article generation, review workflow, and flashcard distillation. Obsidian's built-in graph view handles knowledge graph visualization automatically via `[[wiki links]]`.

## Architecture

```
Obsidian
  └── Hermes Agent sidebar (4 tabs)
       ├── Status  → GET http://localhost:8000/health
       ├── Generate → POST http://localhost:8000/v1/chat/completions
       ├── Review  → reads/writes frontmatter of vault notes
       └── Cards   → generates flashcard notes from articles

vault/
  ├── Wiki/          ← generated articles (Markdown + [[wiki links]])
  └── Flashcards/    ← Newton one-liner fact cards
```

The vLLM server (with TurboQuant) runs independently via `launch.bat`. The plugin talks to it over HTTP — no Python code in the plugin itself.

## UI — Sidebar Panel (4 Tabs)

### Status Tab
- 🟢/🔴 Server health indicator (polls `localhost:8000/health` every 10s)
- VRAM used / total (from `/health` or a custom endpoint)
- Model name, TurboQuant status
- "Open Server Window" button (triggers `launch.bat`)

### Generate Tab
- Topic text input field
- "Generate Article" button
- Progress indicator while Hermes writes
- On completion: opens the new note in the editor
- Article saved to `Wiki/<Topic>.md` with `status: draft` frontmatter

### Review Tab
- Three columns: **Draft** | **Review** | **Published**
- Each article shown as a card with title + created date
- "Promote →" button moves article to next status
- "Reject" button sets status back to draft
- Click article title to open it in the editor

### Cards Tab
- Lists all published articles
- "Generate Cards" button per article → Hermes distills into 3–5 one-liner flashcards
- Cards saved to `Flashcards/<Topic> - Cards.md`
- Preview of existing cards inline

## Data Model

### Article frontmatter
```yaml
---
status: draft          # draft | review | published
created: 2026-04-08
topic: Quantum Entanglement
model: NousResearch/Hermes-3-Llama-3.1-8B
---
```

### Article body pattern
```markdown
# Topic Name
[[Related Topic]] · [[Another Topic]]

Body text with [[wiki links]] throughout to build the graph...
```

### Flashcard format
```markdown
# Quantum Entanglement — Cards

1. Entangled particles share quantum state regardless of distance.
2. Measuring one particle instantly determines the state of its partner.
3. Einstein called this "spooky action at a distance."
```

## Workflow States
```
draft → review → published
         ↑
    (reject back)
```

## File Structure

```
hermes-agent-plugin/     ← plugin source (outside vault)
├── src/
│   ├── main.ts          ← Plugin class, registers sidebar view
│   ├── HermesView.ts    ← Root sidebar view, tab switcher
│   ├── StatusTab.ts     ← Server health + VRAM display
│   ├── GenerateTab.ts   ← Topic input + article generation
│   ├── ReviewTab.ts     ← Draft/Review/Published kanban
│   └── CardsTab.ts      ← Flashcard generation
├── manifest.json
├── package.json
└── esbuild.config.mjs
```

Built output (`main.js` + `manifest.json`) is copied to:
```
[vault]/.obsidian/plugins/hermes-agent/
```

## Tech Stack
- TypeScript
- Obsidian Plugin API (no React/Vue — native DOM API sufficient)
- `esbuild` for bundling
- `fetch` for HTTP calls to vLLM API
- No external npm dependencies beyond Obsidian types + esbuild

## Build & Install

```bash
# One-time setup
npm install

# Development (watch mode)
npm run dev

# Production build
npm run build

# Copy to vault (replace YOUR_VAULT_PATH)
cp dist/main.js dist/manifest.json "YOUR_VAULT_PATH/.obsidian/plugins/hermes-agent/"
```

Then in Obsidian: Settings → Community Plugins → enable "Hermes Agent".

## Out of Scope
- Syncing to remote vaults or cloud
- Multi-vault support
- Custom graph layouts (Obsidian's built-in graph handles this)
- Authentication for the vLLM server
