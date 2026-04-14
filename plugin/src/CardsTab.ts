import { App, Notice, TFile } from 'obsidian';

const OLLAMA_URL = 'http://localhost:11434/v1/chat/completions';
const MODEL      = 'hermes3:8b';
const WIKI_DIR  = 'Wiki';
const CARDS_DIR = 'Flashcards';

function sanitizeName(name: string): string {
  return name.replace(/[:\\/*?"<>|]/g, '-').trim();
}

export class CardsTab {
  private container: HTMLElement;
  private app: App;

  constructor(container: HTMLElement, app: App) {
    this.container = container;
    this.app = app;
    this.render().catch(err => {
      this.container.createEl('p', { text: `Error loading cards: ${err}`, cls: 'hermes-muted' });
    });
  }

  private async render(): Promise<void> {
    this.container.empty();
    this.container.createEl('h3', { text: 'Flashcard Generator' });
    this.container.createEl('p', { text: 'Distill published articles into fact-card one-liners.', cls: 'hermes-muted' });

    const published = await this.getPublished();
    if (published.length === 0) {
      this.container.createEl('p', { text: 'No published articles yet — promote some in the Review tab.' });
      return;
    }
    for (const file of published) this.renderRow(file);
  }

  private renderRow(file: TFile): void {
    const card = this.container.createEl('div', { cls: 'hermes-article-card' });
    card.createEl('div', { text: file.basename, cls: 'title' });
    const cardsPath = `${CARDS_DIR}/${sanitizeName(file.basename)} - Cards.md`;
    const existing = this.app.vault.getAbstractFileByPath(cardsPath);

    if (existing instanceof TFile) {
      card.createEl('p', { text: '✅ Cards exist', cls: 'hermes-muted' });
      const btn = card.createEl('button', { text: '↻ Regenerate', cls: 'hermes-btn' });
      btn.addEventListener('click', () => this.generate(file, card));
    } else {
      const btn = card.createEl('button', { text: '🃏 Generate Cards', cls: 'hermes-btn' });
      btn.addEventListener('click', () => this.generate(file, card));
    }
  }

  private async generate(file: TFile, card: HTMLElement): Promise<void> {
    const btn = card.querySelector('button') as HTMLButtonElement;
    const originalLabel = btn ? btn.textContent ?? '🃏 Generate Cards' : '🃏 Generate Cards';
    if (btn) { btn.disabled = true; btn.textContent = 'Generating…'; }
    try {
      const content = await this.app.vault.read(file);
      const body = content.replace(/^---[\s\S]*?---\n/, '');
      const res = await fetch(OLLAMA_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ollama' },
        body: JSON.stringify({
          model: MODEL,
          messages: [{
            role: 'user',
            content: `Distill this article into exactly 5 precise flashcard facts. Each must be a single sentence under 20 words. Output only:\n1. [fact]\n2. [fact]\n3. [fact]\n4. [fact]\n5. [fact]\n\nArticle:\n${body}`
          }],
          temperature: 0.2,
          max_tokens: 300,
        }),
        signal: AbortSignal.timeout(45_000),
      });
      if (!res.ok) throw new Error(`Ollama ${res.status}`);
      const json = await res.json();
      const cards = json.choices[0].message.content as string;
      await this.saveCards(file.basename, cards);
      new Notice(`Cards saved: ${file.basename}`);
      this.render().catch(err => {
        this.container.createEl('p', { text: `Error loading cards: ${err}`, cls: 'hermes-muted' });
      });
    } catch (e) {
      new Notice(`Error: ${(e as Error).message}`);
      if (btn) { btn.disabled = false; btn.textContent = originalLabel; }
    }
  }

  private async saveCards(topic: string, cards: string): Promise<void> {
    const { vault } = this.app;
    if (!vault.getAbstractFileByPath(CARDS_DIR)) await vault.createFolder(CARDS_DIR);
    const path = `${CARDS_DIR}/${sanitizeName(topic)} - Cards.md`;
    const content = `# ${topic} — Flashcards\n\n${cards}\n`;
    const existing = vault.getAbstractFileByPath(path);
    if (existing instanceof TFile) {
      await vault.modify(existing, content);
    } else {
      await vault.create(path, content);
    }
  }

  private async getPublished(): Promise<TFile[]> {
    const files = this.app.vault.getFiles().filter(
      f => f.path.startsWith(WIKI_DIR + '/') && f.extension === 'md'
    );
    const out: TFile[] = [];
    for (const f of files) {
      const c = await this.app.vault.read(f);
      if (/status:\s*published/.test(c)) out.push(f);
    }
    return out;
  }
}
