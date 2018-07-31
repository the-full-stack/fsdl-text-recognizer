import {JupyterLab, JupyterLabPlugin} from '@jupyterlab/application';
import {ICommandPalette} from '@jupyterlab/apputils';
import {Widget} from '@phosphor/widgets';

import '../style/index.css';

/**
 * Initialization data for the jupyterlab_wandb extension.
 */
const extension: JupyterLabPlugin<void> = {
  id: 'jupyterlab_wandb',
  autoStart: true,
  requires: [ICommandPalette],
  activate: (app: JupyterLab, palette: ICommandPalette) => {
    // Create a single widget
    let widget: Widget = new Widget();
    widget.id = 'wandb';
    widget.title.label = 'W&B Tutorial';
    widget.title.closable = true;
    // Add an application command
    const command: string = 'wandb:open';
    app.commands.addCommand(command, {
      label: 'W&B',
      execute: () => {
        if (!widget.isAttached) {
          // Attach the widget to the main work area if it's not there
          app.shell.addToMainArea(widget);
        }
        // Activate the widget
        app.shell.activateById(widget.id);
      },
    });

    // Add the command to the palette.
    palette.addItem({command, category: 'Tutorial'});
  },
};

export default extension;
