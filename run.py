import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import os
import yaml
import wandb
import argparse
from datetime import datetime
from torchvision import datasets
from torchvision import transforms
from timm.scheduler import CosineLRScheduler

from utils.utils import dict2namespace, seed_everything
from utils.augment import AllSky, PhysicsAwareAugmentation
from scripts.train import train_regression
from scripts.test import test
#from utils.loss import CustomCrossEntropyLoss
from dataset.dataset import CloudDataset
from archs.__init__ import make_model

#os.environ['WANDB_API_KEY'] = 
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='Train or Inference')    
    parser.add_argument('--config', type=str, default='cfg/baseline.yml', help='Path to config file')
    parser.add_argument('--device', type=int, default=0, help="GPU device")
    parser.add_argument('--name', type=str, default='test-model')
    args = parser.parse_args()

    SEED=42
    seed_everything(SEED=SEED)
    torch.backends.cudnn.deterministic = True

    GPU        = args.device
    MODEL_NAME = args.name
    CONFIG     = args.config
    MODE       = args.mode

    MODEL_NAME += "_" + datetime.now().strftime("%d%H%M")

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{GPU}')
        torch.cuda.set_device(device)
        print(f"Using NVIDIA GPU (GPU {GPU})")
    else:
        device = torch.device("cpu")
        print("No NVIDIA GPU detected. Using CPU.")

    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)

    cfg = dict2namespace(config)

    ###### WANDB ######

    USE_WANDB   = cfg.wandb.use

    print (20*"****")
    print (MODEL_NAME, device, USE_WANDB, CONFIG)
    print (20*"****")

    ###### TRAINING DATASET ######

    train_dir = cfg.train.train_dir
    augmentations_cfg = cfg.augmentations

    if cfg.train.physics_degradations:
        transform_train = PhysicsAwareAugmentation(cfg=augmentations_cfg)
    else:
        transform_train = AllSky()

    train_dataset = CloudDataset(folder_path=train_dir, txt_path="./datos/train.txt", transform=transform_train)
    
    # if cfg.train.refine_flag == True: #no lo usamos en este proyecto

    #     refine_dir = cfg.train.refine_dir
    #     refine_dataset = datasets.ImageFolder(refine_dir, transform_train)
        
    #     # Fix the proportion of refine data in each batch
    #     n_old = len(train_dataset)
    #     n_refine = len(refine_dataset)

    #     target_percentage_refine = cfg.train.percentage_refine
    #     target_percentage_old = 1 - target_percentage_refine

    #     weight_old = target_percentage_old / n_old
    #     weight_refine = target_percentage_refine / n_refine

    #     # Create a list of weights for each sample in both datasets
    #     weights = [weight_old] * n_old + [weight_refine] * n_refine
    #     weights = torch.DoubleTensor(weights) # store as DoubleTensor

    #     sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    #     train_dataset_combined = ConcatDataset([train_dataset, refine_dataset])
    #     train_dataloader = DataLoader(train_dataset_combined, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)

    # else:

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)

    ###### VALIDATION DATASET DAY ######

    validation_dir = cfg.validation.val_dir
    transform_val = transforms.Compose([transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    val_dataset = CloudDataset(folder_path=validation_dir, txt_path="./datos/val.txt", transform=transform_val)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.validation.batch_size, shuffle=True, num_workers=cfg.validation.num_workers, pin_memory=True, drop_last=False)

    ###### TEST DATASET DAY ######

    if cfg.test.test_set == True:
        test_dir = cfg.test.test_dir
    elif cfg.test.test_set == False:
       test_dir = cfg.validation.val_dir

    test_dataset = CloudDataset(folder_path=test_dir, txt_path="./datos/test.txt", transform=transform_val)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.test.batch_size, shuffle=True, num_workers=cfg.test.num_workers, pin_memory=True, drop_last=False)


    ###### MODEL, OPTIMIZER, SCHEDULER & CRITERION ######

    model_config = cfg.model.models[cfg.model.pick]
    model = make_model(pick=cfg.model.pick, cfg=model_config, device=device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    if cfg.transformers.warmup == True:
        scheduler = CosineLRScheduler(optimizer, t_initial=cfg.train.epochs, lr_min=cfg.transformers.lr_min, warmup_lr_init=cfg.transformers.warmup_lr_init, warmup_t=cfg.transformers.warmup_t, cycle_limit=cfg.transformers.cycle_limit)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.T_max ,eta_min=cfg.optim.eta_min)

    if cfg.train.loss == 'MSE':
        criterion = torch.nn.MSELoss()
   # elif cfg.train.loss == 'CustomLoss':
   #    criterion = CustomCrossEntropyLoss()
    elif cfg.train.loss == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()

    torch.cuda.empty_cache()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    if MODE == 'train':

        if cfg.train.refine_flag:
            
            print(f"""
            Experiment {MODEL_NAME}
            ---------------------------------
            Train Dataset {len(train_dataset_combined)}, Batch size Train {cfg.train.batch_size}
            Validation Dataset {len(val_dataset)}, Batch size Validation {cfg.validation.batch_size}
            Scheduler: Max LR {cfg.optim.lr} --> Min LR {cfg.optim.eta_min}
            ---------------------------------
            """)

        else:

            print(f"""
            Experiment {MODEL_NAME}
            ---------------------------------
            Train Dataset {len(train_dataset)}, Batch size Train {cfg.train.batch_size}
            Validation Dataset {len(val_dataset)}, Batch size Validation {cfg.validation.batch_size}
            Scheduler: Max LR {cfg.optim.lr} --> Min LR {cfg.optim.eta_min}
            ---------------------------------
            """)
        
        
        print (20*"****")

    elif MODE == 'inference':

        ruta_modelo = os.path.join("results", args.name, f"{args.name}.pt")
        if os.path.exists(ruta_modelo):
            print(f"Loading saved model weights from: {ruta_modelo}")
            model.load_state_dict(torch.load(ruta_modelo, map_location=device))
        else:
            print(f"WARNING: Model file not found at: {ruta_modelo}")


        print(f"""
            Testing {args.name}
            ---------------------------------
            Test Dataset {len(test_dataset)}, Batch size Test {cfg.test.batch_size}
            ---------------------------------
            """)
        
        print (20*"****")


###### TRAIN & INFERENCE ######

    if MODE == 'train':

        if USE_WANDB:
            ruta_wandb_id = './checkpoints/wandb_id.txt'
            
            if os.path.exists(ruta_wandb_id):
                # Si existe el archivo, leemos el ID y reanudamos
                with open(ruta_wandb_id, 'r') as f:
                    run_id = f.read().strip()
                
                wandb.init(project=cfg.wandb.project, id=run_id, resume="must", config=cfg)
                print(f" Resuming W&B session with ID: {run_id}")
            else:
                # Si no existe, creamos una nueva ejecución y guardamos su ID en Drive
                run = wandb.init(project=cfg.wandb.project, name=MODEL_NAME, config=cfg)
                
                with open(ruta_wandb_id, 'w') as f:
                    f.write(run.id)
                print(f"New W&B session started. ID saved: {run.id}")
                
            wandb.save(f"{CONFIG}")

        train_regression(model, optimizer, scheduler, train_dataloader, val_dataloader, criterion, device, use_wandb=USE_WANDB, epochs=cfg.train.epochs, verbose=cfg.train.log_freq, 
            modelname=MODEL_NAME, use_amp=cfg.optim.amsgrad, out_path=f"./results/{MODEL_NAME}/", patience=cfg.train.patience, accum_steps=cfg.optim.accum_steps, freeze_steps=cfg.transformers.freeze_steps, 
            freeze=cfg.transformers.freeze, warmup=cfg.transformers.warmup, refine_flag=cfg.train.refine_flag)

        if USE_WANDB:
            wandb.finish()

        print (" TO BE CONTINUED ")
        print (20*"****")

    elif MODE == 'inference':

        test(model, test_dataloader, modelname=args.name, device=device, mode_test=True, save_errors=cfg.test.save_errors, 
            save_predictions=cfg.test.save_predictions)

        print('Test finished, you can see results in results_test.txt')




