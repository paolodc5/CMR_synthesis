import torch
import os


class EarlyStopping:
    def __init__(self, patience=10, delta=0.005, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


def save_checkpoint(exp_dir, epoch, generator, discriminator, opt_g, opt_d, history):
    """
    Saves the complete training state to be able to resume it.
    Always overwrites the same file 'last_checkpoint.pth' to save space.
    """
    checkpoint_path = os.path.join(exp_dir, "last_checkpoint.pth")
    
    state = {
        'epoch': epoch,
        'gen_state_dict': generator.state_dict(),
        'disc_state_dict': discriminator.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(),
        'opt_d_state_dict': opt_d.state_dict(),
        'history': history
    }
    
    torch.save(state, checkpoint_path)
