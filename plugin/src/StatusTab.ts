import { App } from 'obsidian';

const TAGS_URL   = 'http://localhost:11434/api/tags';
const MODELS_URL = 'http://localhost:11434/v1/models';
const POLL_MS    = 10_000;

export class StatusTab {
  private container: HTMLElement;
  private timer: number | null = null;

  constructor(container: HTMLElement, _app: App) {
    this.container = container;
    this.refresh();
    this.timer = window.setInterval(() => this.refresh(), POLL_MS);
  }

  destroy(): void {
    if (this.timer !== null) window.clearInterval(this.timer);
  }

  private async refresh(): Promise<void> {
    this.container.empty();
    try {
      const [tagsRes, modelsRes] = await Promise.all([
        fetch(TAGS_URL,   { signal: AbortSignal.timeout(3000) }),
        fetch(MODELS_URL, { signal: AbortSignal.timeout(3000) }),
      ]);
      if (!tagsRes.ok) throw new Error(`Ollama returned ${tagsRes.status}`);
      const modelsJson = await modelsRes.json();
      const model: string = modelsJson?.data?.[0]?.id ?? 'unknown';
      this.renderOnline(model);
    } catch {
      this.renderOffline();
    }
  }

  private row(label: string, value: string, badgeCls = ''): void {
    const r = this.container.createEl('div', { cls: 'hermes-status-row' });
    r.createEl('span', { text: label });
    const v = r.createEl('span', { text: value });
    if (badgeCls) v.addClass('hermes-badge', badgeCls);
  }

  private renderOnline(model: string): void {
    this.row('Ollama', '🟢 Running', 'green');
    this.row('Model',  model);
    this.container.createEl('hr');
    const btn = this.container.createEl('button', { text: '↻ Refresh', cls: 'hermes-btn' });
    btn.addEventListener('click', () => this.refresh());
  }

  private renderOffline(): void {
    this.row('Ollama', '🔴 Offline', 'red');
    this.container.createEl('p', {
      text: 'Start Ollama from your system tray, then click Refresh.',
      cls: 'hermes-muted'
    });
    this.container.createEl('hr');
    const btn = this.container.createEl('button', { text: '↻ Refresh', cls: 'hermes-btn' });
    btn.addEventListener('click', () => this.refresh());
  }
}
