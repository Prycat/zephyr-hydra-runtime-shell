import { App, ItemView, WorkspaceLeaf } from 'obsidian';
import { StatusTab } from './StatusTab';
import { GenerateTab } from './GenerateTab';
import { ReviewTab } from './ReviewTab';
import { CardsTab } from './CardsTab';

export const VIEW_TYPE_HERMES = 'hermes-agent-view';

export class HermesView extends ItemView {
  private activeTab: string = 'status';
  private contentEl2: HTMLElement | null = null;

  constructor(leaf: WorkspaceLeaf) {
    super(leaf);
  }

  getViewType(): string { return VIEW_TYPE_HERMES; }
  getDisplayText(): string { return 'Hermes Agent'; }
  getIcon(): string { return 'brain'; }

  async onOpen(): Promise<void> {
    this.contentEl.empty();
    this.contentEl.addClass('hermes-root');

    // Tab bar — rendered once, persists across tab switches
    this.renderTabBar(this.contentEl);

    // Content area — replaced on every tab switch
    this.contentEl2 = this.contentEl.createEl('div', { cls: 'hermes-tab-content' });
    this.renderActiveTab();
  }

  async onClose(): Promise<void> {
    this.contentEl2 = null;
  }

  private renderTabBar(root: HTMLElement): void {
    const bar = root.createEl('div', { cls: 'hermes-tab-bar' });
    const tabs = [
      { id: 'status',   label: '🟢 Status'   },
      { id: 'generate', label: '✍️ Generate'  },
      { id: 'review',   label: '📋 Review'   },
      { id: 'cards',    label: '🃏 Cards'    },
    ];
    for (const tab of tabs) {
      const btn = bar.createEl('button', { text: tab.label, cls: 'hermes-tab-btn' });
      if (tab.id === this.activeTab) btn.addClass('active');
      btn.addEventListener('click', () => {
        this.switchTab(tab.id, bar);
      });
    }
  }

  private switchTab(tabId: string, bar: HTMLElement): void {
    this.activeTab = tabId;

    // Update active button styling
    bar.querySelectorAll('.hermes-tab-btn').forEach((btn, i) => {
      const tabs = ['status', 'generate', 'review', 'cards'];
      btn.classList.toggle('active', tabs[i] === tabId);
    });

    // Only replace content area, not the tab bar
    if (this.contentEl2) {
      this.contentEl2.empty();
      this.renderActiveTab();
    }
  }

  private renderActiveTab(): void {
    if (!this.contentEl2) return;
    switch (this.activeTab) {
      case 'status':   new StatusTab(this.contentEl2, this.app);   break;
      case 'generate': new GenerateTab(this.contentEl2, this.app); break;
      case 'review':   new ReviewTab(this.contentEl2, this.app);   break;
      case 'cards':    new CardsTab(this.contentEl2, this.app);    break;
    }
  }
}
