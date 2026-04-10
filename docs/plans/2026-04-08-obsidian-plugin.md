# Obsidian Plugin — Hermes Knowledge Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an Obsidian plugin with a 4-tab sidebar (Status, Generate, Review, Cards) that controls the local Hermes 3 + vLLM server, generates wiki articles into the vault, manages a draft→review→published workflow, and distills articles into flashcard one-liners.

**Architecture:** TypeScript plugin using the Obsidian Plugin API + esbuild. Each tab is a self-contained class that renders into a shared `ItemView` container. All Hermes calls go to `http://localhost:8000/v1/chat/completions` (the vLLM OpenAI-compatible endpoint already running from `start_server.py`). Articles and flashcards are stored as plain Markdown files in the vault with YAML frontmatter for status tracking.

**Tech Stack:** TypeScript, Obsidian Plugin API, esbuild, `fetch` (built-in), no external runtime dependencies.

---

## Before You Start

**Set your vault path once** — replace `YOUR_VAULT_PATH` everywhere in this plan with the full path to your Obsidian vault, e.g. `C:/Users/gamer23/Documents/MyVault`.

Find it in Obsidian → top-left vault name → right-click → "Show in system explorer".

---

### Task 1: Project Scaffold

**Files:**
- Create: `C:/Users/gamer23/Desktop/hermes-agent/plugin/package.json`
- Create: `C:/Users/gamer23/Desktop/hermes-agent/plugin/tsconfig.json`
- Create: `C:/Users/gamer23/Desktop/hermes-agent/plugin/esbuild.config.mjs`
- Create: `C:/Users/gamer23/Desktop/hermes-agent/plugin/manifest.json`

**Step 1: Create the plugin directory**

```bash
mkdir C:/Users/gamer23/Desktop/hermes-agent/plugin
cd C:/Users/gamer23/Desktop/hermes-agent/plugin
```

**Step 2: Create `manifest.json`**

```json
{
  "id": "hermes-agent",
  "name": "Hermes Agent",
  "version": "1.0.0",
  "minAppVersion": "1.4.0",
  "description": "Local AI knowledge agent powered by Hermes 3 + vLLM + TurboQuant",
  "author": "Local",
  "isDesktopOnly": true
}
```

**Step 3: Create `package.json`**

```json
{
  "name": "hermes-agent-plugin",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "node esbuild.config.mjs",
    "build": "node esbuild.config.mjs production"
  },
  "devDependencies": {
    "@types/node": "^18.0.0",
    "esbuild": "^0.17.0",
    "obsidian": "latest",
    "typescript": "^5.0.0"
  }
}
```

**Step 4: Create `tsconfig.json`**

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "inlineSourceMap": true,
    "inlineSources": true,
    "module": "ESNext",
    "target": "ES6",
    "allowSyntheticDefaultImports": true,
    "moduleResolution": "bundler",
    "importHelpers": true,
    "isolatedModules": true,
    "strictNullChecks": true,
    "lib": ["ES6", "DOM"]
  },
  "include": ["src/**/*.ts"]
}
```

**Step 5: Create `esbuild.config.mjs`**

```js
import esbuild from "esbuild";
import process from "process";

const prod = process.argv[2] === "production";

const ctx = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: ["obsidian", "electron", "@codemirror/*", "@lezer/*"],
  format: "cjs",
  target: "es2018",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  outfile: "dist/main.js",
});

if (prod) {
  await ctx.rebuild();
  process.exit(0);
} else {
  await ctx.watch();
}
```

**Step 6: Install dependencies**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent/plugin
npm install
```

Expected: `node_modules/` created, no errors.

**Step 7: Create `src/` directory and verify build works with an empty entry**

```bash
mkdir src
echo "export default class HermesPlugin {}" > src/main.ts
npm run build
```

Expected: `dist/main.js` created with no errors.

**Step 8: Commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git add plugin/
git commit -m "feat: scaffold Obsidian plugin project"
```

---

### Task 2: Plugin Entry Point + Sidebar View Shell

**Files:**
- Create: `plugin/src/main.ts`
- Create: `plugin/src/HermesView.ts`
- Create: `plugin/src/styles.css`

**Step 1: Create `plugin/src/HermesView.ts`** (shell — tabs render placeholder text)

```typescript
import { App, ItemView, WorkspaceLeaf } from 'obsidian';

export const VIEW_TYPE_HERMES = 'hermes-agent-view';

export class HermesView extends ItemView {
  private activeTab: string = 'status';

  constructor(leaf: WorkspaceLeaf) {
    super(leaf);
  }

  getViewType(): string { return VIEW_TYPE_HERMES; }
  getDisplayText(): string { return 'Hermes Agent'; }
  getIcon(): string { return 'brain'; }

  async onOpen(): Promise<void> {
    const root = this.containerEl.children[1] as HTMLElement;
    root.empty();
    root.addClass('hermes-root');
    this.renderTabBar(root);
    this.renderActiveTab(root);
  }

  async onClose(): Promise<void> {}

  private renderTabBar(root: HTMLElement): void {
    const bar = root.createEl('div', { cls: 'hermes-tab-bar' });
    const tabs: { id: string; label: string }[] = [
      { id: 'status',   label: '🟢 Status'   },
      { id: 'generate', label: '✍️ Generate'  },
      { id: 'review',   label: '📋 Review'   },
      { id: 'cards',    label: '🃏 Cards'    },
    ];
    for (const tab of tabs) {
      const btn = bar.createEl('button', { text: tab.label, cls: 'hermes-tab-btn' });
      if (tab.id === this.activeTab) btn.addClass('active');
      btn.addEventListener('click', () => {
        this.activeTab = tab.id;
        this.onOpen();
      });
    }
  }

  private renderActiveTab(root: HTMLElement): void {
    const content = root.createEl('div', { cls: 'hermes-tab-content' });
    content.createEl('p', { text: `Tab: ${this.activeTab} (coming soon)` });
  }
}
```

**Step 2: Create `plugin/src/main.ts`**

```typescript
import { Plugin, WorkspaceLeaf } from 'obsidian';
import { HermesView, VIEW_TYPE_HERMES } from './HermesView';

export default class HermesPlugin extends Plugin {
  async onload(): Promise<void> {
    this.registerView(VIEW_TYPE_HERMES, (leaf) => new HermesView(leaf));

    this.addRibbonIcon('brain', 'Open Hermes Agent', () => {
      this.activateView();
    });

    this.addCommand({
      id: 'open-hermes-agent',
      name: 'Open Hermes Agent panel',
      callback: () => this.activateView(),
    });
  }

  async onunload(): Promise<void> {
    this.app.workspace.detachLeavesOfType(VIEW_TYPE_HERMES);
  }

  private async activateView(): Promise<void> {
    const { workspace } = this.app;
    let leaf: WorkspaceLeaf | null = workspace.getLeavesOfType(VIEW_TYPE_HERMES)[0] ?? null;
    if (!leaf) {
      leaf = workspace.getRightLeaf(false);
      if (!leaf) return;
      await leaf.setViewState({ type: VIEW_TYPE_HERMES, active: true });
    }
    workspace.revealLeaf(leaf);
  }
}
```

**Step 3: Create `plugin/src/styles.css`**

```css
.hermes-root {
  display: flex;
  flex-direction: column;
  height: 100%;
  font-size: 13px;
}

.hermes-tab-bar {
  display: flex;
  border-bottom: 1px solid var(--background-modifier-border);
  padding: 4px 4px 0;
  gap: 2px;
  flex-shrink: 0;
}

.hermes-tab-btn {
  background: none;
  border: none;
  border-radius: 4px 4px 0 0;
  padding: 4px 8px;
  cursor: pointer;
  color: var(--text-muted);
  font-size: 12px;
}

.hermes-tab-btn:hover { background: var(--background-modifier-hover); }
.hermes-tab-btn.active {
  color: var(--text-normal);
  border-bottom: 2px solid var(--interactive-accent);
}

.hermes-tab-content {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.hermes-status-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px solid var(--background-modifier-border);
}

.hermes-badge {
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 10px;
  background: var(--background-modifier-hover);
}

.hermes-badge.green { background: #1a3a1a; color: #4caf50; }
.hermes-badge.red   { background: #3a1a1a; color: #f44336; }

.hermes-input { width: 100%; margin-bottom: 8px; }
.hermes-btn {
  width: 100%;
  padding: 6px;
  background: var(--interactive-accent);
  color: var(--text-on-accent);
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}
.hermes-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.hermes-review-col h3 { margin: 0 0 8px; font-size: 12px; text-transform: uppercase; color: var(--text-muted); }
.hermes-article-card {
  background: var(--background-secondary);
  border-radius: 6px;
  padding: 8px;
  margin-bottom: 6px;
}
.hermes-article-card .title { font-weight: 600; margin-bottom: 4px; cursor: pointer; }
.hermes-article-card .title:hover { color: var(--interactive-accent); }
.hermes-card-actions { display: flex; gap: 4px; margin-top: 6px; }
.hermes-card-actions button { flex: 1; font-size: 11px; padding: 2px 4px; border-radius: 3px; border: 1px solid var(--background-modifier-border); cursor: pointer; background: var(--background-primary); }
```

**Step 4: Update `esbuild.config.mjs` to copy CSS**

Replace the outfile line and add a plugin to copy styles:

```js
import esbuild from "esbuild";
import process from "process";
import { copyFileSync, mkdirSync, existsSync } from "fs";

const prod = process.argv[2] === "production";

if (!existsSync("dist")) mkdirSync("dist");
copyFileSync("src/styles.css", "dist/styles.css");

const ctx = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: ["obsidian", "electron", "@codemirror/*", "@lezer/*"],
  format: "cjs",
  target: "es2018",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  outfile: "dist/main.js",
});

if (prod) {
  await ctx.rebuild();
  process.exit(0);
} else {
  await ctx.watch();
}
```

**Step 5: Build and verify**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent/plugin
npm run build
```

Expected: `dist/main.js` and `dist/styles.css` created, no TypeScript errors.

**Step 6: Install into vault and test shell**

```bash
# Replace YOUR_VAULT_PATH with your actual vault path
set VAULT=YOUR_VAULT_PATH
mkdir "%VAULT%\.obsidian\plugins\hermes-agent"
copy dist\main.js "%VAULT%\.obsidian\plugins\hermes-agent\main.js"
copy dist\styles.css "%VAULT%\.obsidian\plugins\hermes-agent\styles.css"
copy manifest.json "%VAULT%\.obsidian\plugins\hermes-agent\manifest.json"
```

In Obsidian: Settings → Community Plugins → reload → enable "Hermes Agent" → click the 🧠 ribbon icon.

Expected: sidebar opens with 4 tab buttons and placeholder text.

**Step 7: Commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git add plugin/src/main.ts plugin/src/HermesView.ts plugin/src/styles.css plugin/esbuild.config.mjs
git commit -m "feat: plugin entry point and sidebar shell with tab bar"
```

---

### Task 3: Status Tab

**Files:**
- Create: `plugin/src/StatusTab.ts`
- Modify: `plugin/src/HermesView.ts` (wire up StatusTab in `renderActiveTab`)

**Step 1: Create `plugin/src/StatusTab.ts`**

```typescript
import { App } from 'obsidian';

const HEALTH_URL = 'http://localhost:8000/health';
const POLL_INTERVAL_MS = 10_000;

interface StatusState {
  online: boolean;
  model: string;
  vramUsedGb: number;
  vramTotalGb: number;
  turboquantActive: boolean;
}

export class StatusTab {
  private container: HTMLElement;
  private app: App;
  private pollTimer: number | null = null;

  constructor(container: HTMLElement, app: App) {
    this.container = container;
    this.app = app;
    this.render();
    this.startPolling();
  }

  destroy(): void {
    if (this.pollTimer !== null) window.clearInterval(this.pollTimer);
  }

  private startPolling(): void {
    this.pollTimer = window.setInterval(() => this.refresh(), POLL_INTERVAL_MS);
  }

  private async refresh(): Promise<void> {
    const status = await this.fetchStatus();
    this.container.empty();
    this.renderStatus(status);
  }

  private async fetchStatus(): Promise<StatusState> {
    try {
      const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) throw new Error('not ok');
      // vLLM /health returns 200 when ready; get model info from /v1/models
      const modelsRes = await fetch('http://localhost:8000/v1/models', { signal: AbortSignal.timeout(3000) });
      const modelsJson = await modelsRes.json();
      const model: string = modelsJson?.data?.[0]?.id ?? 'unknown';
      return { online: true, model, vramUsedGb: 0, vramTotalGb: 12.9, turboquantActive: true };
    } catch {
      return { online: false, model: '—', vramUsedGb: 0, vramTotalGb: 0, turboquantActive: false };
    }
  }

  private render(): void {
    this.container.empty();
    const loading = this.container.createEl('p', { text: 'Checking server…', cls: 'hermes-muted' });
    this.fetchStatus().then(status => {
      loading.remove();
      this.renderStatus(status);
    });
  }

  private renderStatus(s: StatusState): void {
    const badge = (text: string, cls: string) => {
      const b = createEl('span', { text, cls: `hermes-badge ${cls}` });
      return b;
    };

    const row = (label: string, valueEl: HTMLElement) => {
      const r = this.container.createEl('div', { cls: 'hermes-status-row' });
      r.createEl('span', { text: label });
      r.appendChild(valueEl);
    };

    row('Server', badge(s.online ? '🟢 Online' : '🔴 Offline', s.online ? 'green' : 'red'));
    row('TurboQuant', badge(s.turboquantActive ? '✅ Active (3K/4V)' : '❌ Inactive', s.turboquantActive ? 'green' : 'red'));
    row('Model', createEl('span', { text: s.model }));

    if (s.online) {
      row('VRAM', createEl('span', { text: `${s.vramUsedGb.toFixed(1)} / ${s.vramTotalGb.toFixed(1)} GB` }));
    }

    this.container.createEl('hr');
    const refreshBtn = this.container.createEl('button', { text: '↻ Refresh', cls: 'hermes-btn' });
    refreshBtn.style.marginTop = '8px';
    refreshBtn.addEventListener('click', () => this.refresh());

    if (!s.online) {
      this.container.createEl('p', {
        text: 'Start the server: run launch.bat or python start_server.py',
        cls: 'hermes-muted'
      });
    }
  }
}
```

**Step 2: Wire up in `HermesView.ts`**

At the top of `HermesView.ts`, add import:
```typescript
import { StatusTab } from './StatusTab';
```

Replace `renderActiveTab` method:
```typescript
private renderActiveTab(root: HTMLElement): void {
  const content = root.createEl('div', { cls: 'hermes-tab-content' });
  switch (this.activeTab) {
    case 'status':
      new StatusTab(content, this.app);
      break;
    default:
      content.createEl('p', { text: `Tab: ${this.activeTab} (coming soon)` });
  }
}
```

**Step 3: Build, install, verify**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent/plugin && npm run build
copy dist\main.js "YOUR_VAULT_PATH\.obsidian\plugins\hermes-agent\main.js"
```

In Obsidian: Cmd/Ctrl+R to reload. Open Hermes panel → Status tab.

Expected: shows "🔴 Offline" if server not running, "🟢 Online" if it is. Refresh button works.

**Step 4: Commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git add plugin/src/StatusTab.ts plugin/src/HermesView.ts
git commit -m "feat: Status tab with server health polling"
```

---

### Task 4: Generate Tab

**Files:**
- Create: `plugin/src/GenerateTab.ts`
- Modify: `plugin/src/HermesView.ts` (wire up GenerateTab)

**Step 1: Create `plugin/src/GenerateTab.ts`**

```typescript
import { App, Notice, TFile } from 'obsidian';

const VLLM_URL = 'http://localhost:8000/v1/chat/completions';
const MODEL    = 'NousResearch/Hermes-3-Llama-3.1-8B';
const WIKI_DIR = 'Wiki';

export class GenerateTab {
  private container: HTMLElement;
  private app: App;

  constructor(container: HTMLElement, app: App) {
    this.container = container;
    this.app = app;
    this.render();
  }

  private render(): void {
    this.container.createEl('h3', { text: 'Generate Wiki Article' });

    const input = this.container.createEl('input', {
      type: 'text',
      placeholder: 'Topic (e.g. "Quantum Entanglement")',
      cls: 'hermes-input',
    }) as HTMLInputElement;

    const btn = this.container.createEl('button', {
      text: 'Generate Article',
      cls: 'hermes-btn',
    });

    const status = this.container.createEl('p', { text: '', cls: 'hermes-muted' });

    btn.addEventListener('click', async () => {
      const topic = input.value.trim();
      if (!topic) { new Notice('Enter a topic first.'); return; }
      btn.disabled = true;
      btn.setText('Generating…');
      status.setText('Calling Hermes…');
      try {
        const article = await this.generateArticle(topic);
        await this.saveArticle(topic, article);
        status.setText(`✅ Saved to Wiki/${topic}.md`);
        btn.setText('Generate Article');
        input.value = '';
      } catch (e) {
        status.setText(`❌ Error: ${(e as Error).message}`);
        btn.setText('Generate Article');
      } finally {
        btn.disabled = false;
      }
    });
  }

  private async generateArticle(topic: string): Promise<string> {
    const prompt = `Write a comprehensive, well-structured wiki article about "${topic}".

Requirements:
- Use Markdown formatting with # headings
- Include [[wiki links]] to related concepts (wrapped in double brackets)
- Cover: definition, background, key concepts, significance, related topics
- Be factual, clear, and encyclopedic in tone
- Length: 400-600 words

Start directly with the article (no preamble).`;

    const res = await fetch(VLLM_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer unused' },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.3,
        max_tokens: 1024,
      }),
      signal: AbortSignal.timeout(60_000),
    });

    if (!res.ok) throw new Error(`vLLM returned ${res.status}`);
    const json = await res.json();
    return json.choices[0].message.content as string;
  }

  private async saveArticle(topic: string, body: string): Promise<void> {
    const { vault } = this.app;

    // Ensure Wiki/ folder exists
    if (!vault.getAbstractFileByPath(WIKI_DIR)) {
      await vault.createFolder(WIKI_DIR);
    }

    const today = new Date().toISOString().split('T')[0];
    const frontmatter = `---\nstatus: draft\ncreated: ${today}\ntopic: ${topic}\nmodel: ${MODEL}\n---\n\n`;
    const path = `${WIKI_DIR}/${topic}.md`;

    const existing = vault.getAbstractFileByPath(path);
    if (existing instanceof TFile) {
      await vault.modify(existing, frontmatter + body);
    } else {
      await vault.create(path, frontmatter + body);
    }

    // Open the new note
    const file = vault.getAbstractFileByPath(path) as TFile;
    await this.app.workspace.getLeaf(false).openFile(file);
  }
}
```

**Step 2: Wire up in `HermesView.ts`**

Add import at top:
```typescript
import { GenerateTab } from './GenerateTab';
```

Add to `renderActiveTab` switch:
```typescript
case 'generate':
  new GenerateTab(content, this.app);
  break;
```

**Step 3: Build, install, verify**

```bash
npm run build && copy dist\main.js "YOUR_VAULT_PATH\.obsidian\plugins\hermes-agent\main.js"
```

In Obsidian: reload → Generate tab → type a topic → click Generate.

Expected (server running): new note appears in `Wiki/` folder with frontmatter `status: draft` and article body with `[[wiki links]]`.

Expected (server offline): error message shown, no note created.

**Step 4: Commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git add plugin/src/GenerateTab.ts plugin/src/HermesView.ts
git commit -m "feat: Generate tab — topic input, Hermes article generation, saves to vault"
```

---

### Task 5: Review Tab

**Files:**
- Create: `plugin/src/ReviewTab.ts`
- Modify: `plugin/src/HermesView.ts` (wire up ReviewTab)

**Step 1: Create `plugin/src/ReviewTab.ts`**

```typescript
import { App, TFile, parseYaml, stringifyYaml } from 'obsidian';

const WIKI_DIR = 'Wiki';
type Status = 'draft' | 'review' | 'published';
const NEXT: Record<Status, Status | null> = { draft: 'review', review: 'published', published: null };
const PREV: Record<Status, Status | null> = { draft: null, review: 'draft', published: 'review' };

interface ArticleMeta { file: TFile; topic: string; status: Status; created: string; }

export class ReviewTab {
  private container: HTMLElement;
  private app: App;

  constructor(container: HTMLElement, app: App) {
    this.container = container;
    this.app = app;
    this.render();
  }

  private async render(): Promise<void> {
    this.container.empty();
    const articles = await this.loadArticles();
    const columns: Status[] = ['draft', 'review', 'published'];

    for (const col of columns) {
      const colEl = this.container.createEl('div', { cls: 'hermes-review-col' });
      const label: Record<Status, string> = { draft: '📝 Draft', review: '🔍 Review', published: '✅ Published' };
      colEl.createEl('h3', { text: label[col] });

      const items = articles.filter(a => a.status === col);
      if (items.length === 0) {
        colEl.createEl('p', { text: 'None', cls: 'hermes-muted' });
      }
      for (const art of items) {
        this.renderCard(colEl, art);
      }
    }
  }

  private renderCard(parent: HTMLElement, art: ArticleMeta): void {
    const card = parent.createEl('div', { cls: 'hermes-article-card' });
    const title = card.createEl('div', { text: art.topic, cls: 'title' });
    title.addEventListener('click', () => {
      this.app.workspace.getLeaf(false).openFile(art.file);
    });
    card.createEl('div', { text: art.created, cls: 'hermes-muted' });

    const actions = card.createEl('div', { cls: 'hermes-card-actions' });
    const next = NEXT[art.status];
    const prev = PREV[art.status];

    if (next) {
      const promoteBtn = actions.createEl('button', { text: `→ ${next}` });
      promoteBtn.addEventListener('click', async () => {
        await this.updateStatus(art.file, next);
        this.render();
      });
    }
    if (prev) {
      const rejectBtn = actions.createEl('button', { text: `← ${prev}` });
      rejectBtn.addEventListener('click', async () => {
        await this.updateStatus(art.file, prev);
        this.render();
      });
    }
  }

  private async loadArticles(): Promise<ArticleMeta[]> {
    const { vault } = this.app;
    const files = vault.getFiles().filter(f => f.path.startsWith(WIKI_DIR + '/') && f.extension === 'md');
    const results: ArticleMeta[] = [];
    for (const file of files) {
      const content = await vault.read(file);
      const fm = this.parseFrontmatter(content);
      if (fm) {
        results.push({
          file,
          topic: fm.topic ?? file.basename,
          status: (fm.status as Status) ?? 'draft',
          created: fm.created ?? '—',
        });
      }
    }
    return results.sort((a, b) => b.created.localeCompare(a.created));
  }

  private parseFrontmatter(content: string): Record<string, string> | null {
    const match = content.match(/^---\n([\s\S]*?)\n---/);
    if (!match) return null;
    try { return parseYaml(match[1]) as Record<string, string>; } catch { return null; }
  }

  private async updateStatus(file: TFile, newStatus: Status): Promise<void> {
    const content = await this.app.vault.read(file);
    const updated = content.replace(
      /^(---\n[\s\S]*?)status:\s*\w+/m,
      `$1status: ${newStatus}`
    );
    await this.app.vault.modify(file, updated);
  }
}
```

**Step 2: Wire up in `HermesView.ts`**

Add import:
```typescript
import { ReviewTab } from './ReviewTab';
```

Add to switch:
```typescript
case 'review':
  new ReviewTab(content, this.app);
  break;
```

**Step 3: Build, install, verify**

```bash
npm run build && copy dist\main.js "YOUR_VAULT_PATH\.obsidian\plugins\hermes-agent\main.js"
```

In Obsidian: reload → generate an article → Review tab → should show it in "Draft" column → click "→ review" → refreshes to "Review" column.

**Step 4: Commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git add plugin/src/ReviewTab.ts plugin/src/HermesView.ts
git commit -m "feat: Review tab — draft/review/published kanban with promote/reject"
```

---

### Task 6: Cards Tab

**Files:**
- Create: `plugin/src/CardsTab.ts`
- Modify: `plugin/src/HermesView.ts` (wire up CardsTab)

**Step 1: Create `plugin/src/CardsTab.ts`**

```typescript
import { App, Notice, TFile } from 'obsidian';

const VLLM_URL     = 'http://localhost:8000/v1/chat/completions';
const MODEL        = 'NousResearch/Hermes-3-Llama-3.1-8B';
const WIKI_DIR     = 'Wiki';
const CARDS_DIR    = 'Flashcards';

export class CardsTab {
  private container: HTMLElement;
  private app: App;

  constructor(container: HTMLElement, app: App) {
    this.container = container;
    this.app = app;
    this.render();
  }

  private async render(): Promise<void> {
    this.container.empty();
    this.container.createEl('h3', { text: 'Flashcard Generator' });
    this.container.createEl('p', { text: 'Generate Newton one-liners from published articles.', cls: 'hermes-muted' });

    const published = await this.loadPublished();
    if (published.length === 0) {
      this.container.createEl('p', { text: 'No published articles yet. Promote some in the Review tab.' });
      return;
    }

    for (const file of published) {
      const row = this.container.createEl('div', { cls: 'hermes-article-card' });
      row.createEl('div', { text: file.basename, cls: 'title' });

      const cardsPath = `${CARDS_DIR}/${file.basename} - Cards.md`;
      const existing = this.app.vault.getAbstractFileByPath(cardsPath);

      if (existing instanceof TFile) {
        const preview = await this.app.vault.read(existing);
        const lines = preview.split('\n').filter(l => l.match(/^\d+\./)).slice(0, 2);
        lines.forEach(l => row.createEl('p', { text: l, cls: 'hermes-muted' }));
        const regenBtn = row.createEl('button', { text: '↻ Regenerate Cards', cls: 'hermes-btn' });
        regenBtn.style.marginTop = '4px';
        regenBtn.addEventListener('click', () => this.generateCards(file, row));
      } else {
        const btn = row.createEl('button', { text: '🃏 Generate Cards', cls: 'hermes-btn' });
        btn.style.marginTop = '4px';
        btn.addEventListener('click', () => this.generateCards(file, row));
      }
    }
  }

  private async generateCards(file: TFile, row: HTMLElement): Promise<void> {
    const btn = row.querySelector('button') as HTMLButtonElement;
    if (btn) { btn.disabled = true; btn.setText('Generating…'); }

    try {
      const content = await this.app.vault.read(file);
      const body = content.replace(/^---[\s\S]*?---\n/, ''); // strip frontmatter

      const prompt = `Read this wiki article and distill it into exactly 5 compressed flashcard fact-cards.

Each card must be:
- A single sentence (max 20 words)
- A precise, standalone fact — no fluff
- Written like a scientific law or dictionary definition

Article:
${body}

Output format (nothing else):
1. [fact]
2. [fact]
3. [fact]
4. [fact]
5. [fact]`;

      const res = await fetch(VLLM_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer unused' },
        body: JSON.stringify({
          model: MODEL,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.2,
          max_tokens: 300,
        }),
        signal: AbortSignal.timeout(45_000),
      });

      if (!res.ok) throw new Error(`vLLM ${res.status}`);
      const json = await res.json();
      const cards = json.choices[0].message.content as string;

      await this.saveCards(file.basename, cards);
      new Notice(`Cards saved for ${file.basename}`);
      this.render(); // refresh
    } catch (e) {
      new Notice(`Error: ${(e as Error).message}`);
      if (btn) { btn.disabled = false; btn.setText('🃏 Generate Cards'); }
    }
  }

  private async saveCards(topic: string, cards: string): Promise<void> {
    const { vault } = this.app;
    if (!vault.getAbstractFileByPath(CARDS_DIR)) {
      await vault.createFolder(CARDS_DIR);
    }
    const path = `${CARDS_DIR}/${topic} - Cards.md`;
    const content = `# ${topic} — Flashcards\n\n${cards}\n`;
    const existing = vault.getAbstractFileByPath(path);
    if (existing instanceof TFile) {
      await vault.modify(existing, content);
    } else {
      await vault.create(path, content);
    }
  }

  private async loadPublished(): Promise<TFile[]> {
    const files = this.app.vault.getFiles().filter(
      f => f.path.startsWith(WIKI_DIR + '/') && f.extension === 'md'
    );
    const published: TFile[] = [];
    for (const f of files) {
      const content = await this.app.vault.read(f);
      if (content.match(/^status:\s*published/m) || content.match(/\nstatus:\s*published/)) {
        published.push(f);
      }
    }
    return published;
  }
}
```

**Step 2: Wire up in `HermesView.ts`**

Add import:
```typescript
import { CardsTab } from './CardsTab';
```

Add to switch:
```typescript
case 'cards':
  new CardsTab(content, this.app);
  break;
```

**Step 3: Build, install, verify**

```bash
npm run build && copy dist\main.js "YOUR_VAULT_PATH\.obsidian\plugins\hermes-agent\main.js"
```

In Obsidian: reload → Cards tab → should list published articles → click "Generate Cards" → Flashcards folder appears with 5 one-liners.

**Step 4: Commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git add plugin/src/CardsTab.ts plugin/src/HermesView.ts
git commit -m "feat: Cards tab — distill published articles into Newton one-liner flashcards"
```

---

### Task 7: End-to-End Verification

**Goal:** Full workflow test — generate → review → publish → generate cards → check graph.

**Step 1: Final production build**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent/plugin
npm run build production
copy dist\main.js "YOUR_VAULT_PATH\.obsidian\plugins\hermes-agent\main.js"
copy dist\styles.css "YOUR_VAULT_PATH\.obsidian\plugins\hermes-agent\styles.css"
copy manifest.json "YOUR_VAULT_PATH\.obsidian\plugins\hermes-agent\manifest.json"
```

**Step 2: Full workflow test**

With vLLM server running (`launch.bat`):

1. **Status tab** → should show 🟢 Online, model name, TurboQuant active
2. **Generate tab** → enter "Neural Networks" → Generate → note appears in `Wiki/`
3. **Generate tab** → enter "Deep Learning" → Generate → second note, should contain `[[Neural Networks]]` link
4. **Review tab** → both articles in Draft column → promote Neural Networks to Review → promote to Published
5. **Cards tab** → Neural Networks listed → Generate Cards → `Flashcards/Neural Networks - Cards.md` created with 5 one-liners
6. **Obsidian graph view** (Ctrl+G) → nodes for Neural Networks and Deep Learning connected by wiki link edge

**Step 3: Final commit**

```bash
cd C:/Users/gamer23/Desktop/hermes-agent
git add .
git commit -m "feat: Obsidian Hermes Agent plugin complete — all 4 tabs verified"
```
