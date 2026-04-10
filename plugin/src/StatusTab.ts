import { App } from 'obsidian';

const HEALTH_URL  = 'http://localhost:8000/health';
const MODELS_URL  = 'http://localhost:8000/v1/models';
const POLL_MS     = 10_000;

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
      const [healthRes, modelsRes] = await Promise.all([
        fetch(HEALTH_URL, { signal: AbortSignal.timeout(3000) }),
        fetch(MODELS_URL, { signal: AbortSignal.timeout(3000) }),
      ]);
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
    this.row('Server',     '🟢 Online',  'green');
    this.row('TurboQuant', '✅ Active',   'green');
    this.row('Model',      model);
    this.container.createEl('hr');
    const btn = this.container.createEl('button', { text: '↻ Refresh', cls: 'hermes-btn' });
    btn.addEventListener('click', () => this.refresh());
  }

  private renderOffline(): void {
    this.row('Server', '🔴 Offline', 'red');
    this.container.createEl('p', {
      text: 'Start with: python start_server.py (or double-click launch.bat)',
      cls: 'hermes-muted'
    });
    this.container.createEl('hr');
    const btn = this.container.createEl('button', { text: '↻ Refresh', cls: 'hermes-btn' });
    btn.addEventListener('click', () => this.refresh());
  }
}
