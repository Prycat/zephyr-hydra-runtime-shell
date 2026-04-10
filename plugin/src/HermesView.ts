import { ItemView, WorkspaceLeaf } from 'obsidian';

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
        this.activeTab = tab.id;
        this.onOpen();
      });
    }
  }

  private renderActiveTab(root: HTMLElement): void {
    const content = root.createEl('div', { cls: 'hermes-tab-content' });
    content.createEl('p', { text: `${this.activeTab} tab — coming soon`, cls: 'hermes-muted' });
  }
}
