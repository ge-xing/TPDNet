import numpy as np
from utils.data_utils_video import get_train_val_dataset
import torch 
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
import random 
import os 
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(42)
from utils.misc import cal_Jaccard, fscore
import os
from losses.hinge import lovasz_hinge

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

logdir = f"./logs/tpdm"
env = "pytorch"
model_save_path = os.path.join(logdir, "model")
max_epoch = 15
batch_size = 6
val_every = 1
num_gpus = 1
device = "cuda:0"
image_size = 512

class MirrorTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        from networks.tpdnet.model import SegFormerAdapterExtraDepthMono
        self.model = SegFormerAdapterExtraDepthMono(C_out=6)
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=3e-5, eps=1e-8)
        self.scheduler_type = "poly"
        
    def training_step(self, batch):
        
        image, label = self.get_input(batch)

        pred = self.model(image)

        loss_lovasz = lovasz_hinge(pred, label)

        self.log("loss_lovasz", loss_lovasz, step=self.global_step)
        return loss_lovasz

    def get_input(self, batch):
        image = batch["image"]
        label = batch["mask"]
        return image, label

    def cal_metric(self, pred, gt):
        if pred.sum() > 0 and gt.sum() > 0:
            d = cal_Jaccard(pred, gt)
            f1 = fscore(pred, gt)
            acc = (pred == gt).mean()

            return d, f1, acc
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0, 1.0, 1.0
        
        else:
            return 0.0, 0.0, 0.0
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.model(image)
        
        output = output > 0
        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dice, f1, acc = self.cal_metric(output, target)
        
        return dice, f1, acc
    
    def validation_end(self, val_outputs):
        dices, f1, acc = val_outputs
        dices = dices.mean()
        f1 = f1.mean()
        acc = acc.mean()
        print(f"dices is {dices}, f1 is {f1}, acc is {acc}")
    
        self.log("dices", dices, step=self.epoch)
        self.log("f1", f1, step=self.epoch)
        self.log("acc", acc, step=self.epoch)

        if dices > self.best_mean_dice:
            self.best_mean_dice = dices
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{dices:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{dices:.4f}.pt"), 
                                        delete_symbol="final_model")

if __name__ == "__main__":

    trainer = MirrorTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17752,
                            training_script=__file__)
    
    train_ds, val_ds = get_train_val_dataset(image_size=image_size)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

   