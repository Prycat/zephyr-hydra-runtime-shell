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
