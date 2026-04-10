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
      cls: 'hermes-input',
    }) as HTMLInputElement;
    input.placeholder = 'Topic (e.g. "Quantum Entanglement")';
    input.type = 'text';

    const btn = this.container.createEl('button', {
      text: 'Generate Article',
      cls: 'hermes-btn',
    });

    const status = this.container.createEl('p', { cls: 'hermes-muted' });

    btn.addEventListener('click', async () => {
      const topic = input.value.trim();
      if (!topic) { new Notice('Enter a topic first.'); return; }
      btn.disabled = true;
      btn.textContent = 'Generating…';
      status.textContent = 'Calling Hermes…';
      try {
        const article = await this.generateArticle(topic);
        await this.saveArticle(topic, article);
        status.textContent = `✅ Saved to Wiki/${topic}.md`;
        input.value = '';
      } catch (e) {
        status.textContent = `❌ ${(e as Error).message}`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Article';
      }
    });
  }

  private async generateArticle(topic: string): Promise<string> {
    const res = await fetch(VLLM_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer unused' },
      body: JSON.stringify({
        model: MODEL,
        messages: [{
          role: 'user',
          content: `Write a comprehensive wiki article about "${topic}". Use Markdown with # headings. Include [[wiki links]] to related concepts. 400-600 words. Start directly with the article.`
        }],
        temperature: 0.3,
        max_tokens: 1024,
      }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!res.ok) throw new Error(`vLLM returned ${res.status}`);
    const json = await res.json();
    return (json.choices[0].message.content as string);
  }

  private async saveArticle(topic: string, body: string): Promise<void> {
    const { vault } = this.app;
    if (!vault.getAbstractFileByPath(WIKI_DIR)) await vault.createFolder(WIKI_DIR);
    const today = new Date().toISOString().split('T')[0];
    const fm = `---\nstatus: draft\ncreated: ${today}\ntopic: ${topic}\n---\n\n`;
    const path = `${WIKI_DIR}/${topic}.md`;
    const existing = vault.getAbstractFileByPath(path);
    if (existing instanceof TFile) {
      await vault.modify(existing, fm + body);
    } else {
      await vault.create(path, fm + body);
    }
    const file = vault.getAbstractFileByPath(path) as TFile;
    await this.app.workspace.getLeaf(false).openFile(file);
  }
}
