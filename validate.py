import numpy as np
from utils.data_utils_video import get_train_val_dataset
import torch 
from light_training.trainer import Trainer, dummy_context
from utils.misc import cal_Jaccard
import os
import time 
from PIL import Image
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

env = "pytorch"
max_epoch = 15
batch_size = 8
val_every = 1
num_gpus = 1
device = "cuda:1"
image_size = 512

class MirrorTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        from networks.tpdnet.model import SegFormerAdapterExtraDepthMono
        self.model = SegFormerAdapterExtraDepthMono(C_out=6)
        self.load_state_dict("./logs/tpdm/model/best_model_0.6690.pt", strict=True)
        self.save_dir = "./results/tqdm"
        os.makedirs(self.save_dir, exist_ok=True)
        self.threshold = 0.7

        self.index = 0 

        self.times = []
    def get_input(self, batch):
        image = batch["image"]
        label = batch["mask"]

        video_name = batch["video_name"]
        case_name = batch["case_name"]
        size = batch["size"]
        print(video_name, case_name, size)
        return image, label, video_name[0], case_name[0], [size[0].item(), size[1].item()]
    

    def cal_metric(self, pred, gt):
        if pred.sum() > 0 and gt.sum() > 0:
            d = cal_Jaccard(pred, gt)  
            return d 
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return 1.0
        
        else:
            return 0.0
    
    def validation_step(self, batch):
        save_dir = self.save_dir
        self.index += 1

        image, label, video_name, case_name, size = self.get_input(batch)
        
        s = time.time()
        with torch.autocast("cuda", enabled=True) if 'cuda' in device else dummy_context():
            output = self.model(image)
        
        e = time.time()
        print(f"time is {e - s}")

        self.times.append(e - s)
        output = torch.sigmoid(output)
        output = output > self.threshold
        output_m = output.cpu().numpy()
        target = label.cpu().numpy()
        
        iou = self.cal_metric(output_m, target)

        print(f"iou: {iou}")

        to_pil = transforms.ToPILImage()

        output = output[0, 0].to(torch.float32)
        output = np.array(
                    transforms.Resize(size)(to_pil(output)))
        save_name = f"{case_name}.png"
        os.makedirs(os.path.join(save_dir, video_name), exist_ok=True)
        Image.fromarray(output).save(os.path.join(save_dir, video_name, save_name))

        return 1
    

if __name__ == "__main__":

    trainer = MirrorTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir="",
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17752,
                            training_script=__file__)
    
    train_ds, val_ds = get_train_val_dataset(image_size=image_size)

    mean_m, _ = trainer.validation_single_gpu(val_ds)
    print(mean_m)

    t = trainer.times
    t = sum(t) / len(t)
    print(f"final time is {t}")
   