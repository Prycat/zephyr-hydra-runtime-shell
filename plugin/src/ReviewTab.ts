import { App, TFile, parseYaml } from 'obsidian';

const WIKI_DIR = 'Wiki';
type Status = 'draft' | 'review' | 'published';
const NEXT: Record<Status, Status | null> = { draft: 'review', review: 'published', published: null };
const PREV: Record<Status, Status | null> = { draft: null, review: 'draft', published: 'review' };

interface Article { file: TFile; topic: string; status: Status; created: string; }

export class ReviewTab {
  private container: HTMLElement;
  private app: App;

  constructor(container: HTMLElement, app: App) {
    this.container = container;
    this.app = app;
    this.render().catch(err => {
      this.container.createEl('p', { text: `Error loading review: ${err}`, cls: 'hermes-muted' });
    });
  }

  private async render(): Promise<void> {
    this.container.empty();
    const articles = await this.loadArticles();
    const cols: { status: Status; label: string }[] = [
      { status: 'draft',     label: '📝 Draft'     },
      { status: 'review',    label: '🔍 Review'    },
      { status: 'published', label: '✅ Published'  },
    ];
    for (const col of cols) {
      const colEl = this.container.createEl('div', { cls: 'hermes-review-col' });
      colEl.createEl('h3', { text: col.label });
      const items = articles.filter(a => a.status === col.status);
      if (items.length === 0) {
        colEl.createEl('p', { text: 'None', cls: 'hermes-muted' });
      }
      for (const art of items) this.renderCard(colEl, art);
    }
  }

  private renderCard(parent: HTMLElement, art: Article): void {
    const card = parent.createEl('div', { cls: 'hermes-article-card' });
    const title = card.createEl('div', { text: art.topic, cls: 'title' });
    title.addEventListener('click', () => this.app.workspace.getLeaf(false).openFile(art.file));
    card.createEl('div', { text: art.created, cls: 'hermes-muted' });

    const actions = card.createEl('div', { cls: 'hermes-card-actions' });
    const next = NEXT[art.status];
    const prev = PREV[art.status];
    if (next) {
      const btn = actions.createEl('button', { text: `→ ${next}` });
      btn.addEventListener('click', async () => { await this.setStatus(art.file, next); this.render(); });
    }
    if (prev) {
      const btn = actions.createEl('button', { text: `← ${prev}` });
      btn.addEventListener('click', async () => { await this.setStatus(art.file, prev); this.render(); });
    }
  }

  private async loadArticles(): Promise<Article[]> {
    const files = this.app.vault.getFiles().filter(
      f => f.path.startsWith(WIKI_DIR + '/') && f.extension === 'md'
    );
    const results: Article[] = [];
    for (const file of files) {
      const content = await this.app.vault.read(file);
      const fm = this.parseFm(content);
      if (fm) results.push({
        file,
        topic: (fm.topic as string) ?? file.basename,
        status: ((fm.status as Status) ?? 'draft'),
        created: (fm.created as string) ?? '—',
      });
    }
    return results.sort((a, b) => b.created.localeCompare(a.created));
  }

  private parseFm(content: string): Record<string, unknown> | null {
    const m = content.match(/^---\n([\s\S]*?)\n---/);
    if (!m) return null;
    try { return parseYaml(m[1]) as Record<string, unknown>; } catch { return null; }
  }

  private async setStatus(file: TFile, s: Status): Promise<void> {
    const content = await this.app.vault.read(file);
    const updated = content.replace(/^(status:\s*)\w+/m, `$1${s}`);
    await this.app.vault.modify(file, updated);
  }
}
