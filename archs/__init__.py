from xml.parsers.expat import model
import torch
import timm
import torch.nn as nn
from utils.utils import count_params, load_weights, dict2namespace, interpolate_pos_embed
from fvcore.nn import FlopCountAnalysis
import os


def select_model(pick, model_cfg):

    model_cfg = dict2namespace(model_cfg)

    if pick == 0:

        model_name = f"vgg{model_cfg.type}"
        if model_cfg.bn:
            model_name += "_bn"

        model = timm.create_model(model_name, pretrained=False, num_classes= model_cfg.num_classes, in_chans= model_cfg.in_chans, output_stride= model_cfg.output_stride, 
                                  global_pool= model_cfg.global_pool, drop_rate= model_cfg.drop_rate)

        return model
    
    elif pick == 1:

        model_name = f"resnet{model_cfg.type}"

        model = timm.create_model(model_name, pretrained=False, num_classes= model_cfg.num_classes, in_chans= model_cfg.in_chans, output_stride= model_cfg.output_stride, 
                                  global_pool= model_cfg.global_pool, drop_rate= model_cfg.drop_rate, drop_path_rate= model_cfg.drop_path_rate)

        return model
    
    elif pick == 2:

        model = timm.create_model('inception_v3', pretrained=False, num_classes= model_cfg.num_classes, in_chans= model_cfg.in_chans, 
                                  global_pool= model_cfg.global_pool, drop_rate= model_cfg.drop_rate)
        
        return model
    
    elif pick == 3:

        model_name = f"densenet{model_cfg.type}"

        model = timm.create_model(model_name, pretrained=False, num_classes= model_cfg.num_classes, in_chans= model_cfg.in_chans, 
                                  global_pool= model_cfg.global_pool, drop_rate= model_cfg.drop_rate)
        
        return model
        
    elif pick == 4:

        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes= model_cfg.num_classes, in_chans= model_cfg.in_chans, 
                                  global_pool= model_cfg.global_pool, drop_rate= model_cfg.drop_rate, drop_path_rate= model_cfg.drop_path_rate)
        
        return model
        
    elif pick == 5:

        model = timm.create_model('regnety_040', pretrained=False, num_classes= model_cfg.num_classes, in_chans= model_cfg.in_chans, output_stride= model_cfg.output_stride,
                                  global_pool= model_cfg.global_pool, drop_rate= model_cfg.drop_rate, drop_path_rate= model_cfg.drop_path_rate)
        
        return model
    
    
    elif pick == 6:  

        model_name = f"vit_{model_cfg.type}_patch16_224"

        model = timm.create_model(model_name, pretrained=False, num_classes=model_cfg.num_classes, in_chans=model_cfg.in_chans, drop_rate=model_cfg.drop_rate,
                                drop_path_rate=model_cfg.drop_path_rate, global_pool=model_cfg.global_pool, img_size=model_cfg.img_size)
        print("Embed dim:", model.embed_dim)
        return model
    
    elif pick == 7:  

        model_name = f"swin_{model_cfg.type}_patch4_window7_224"

        model = timm.create_model(model_name, pretrained=False, num_classes=model_cfg.num_classes, in_chans=model_cfg.in_chans, drop_rate=model_cfg.drop_rate,
                                drop_path_rate=model_cfg.drop_path_rate, global_pool=model_cfg.global_pool, img_size=model_cfg.img_size)

        return model
    
    elif pick == 8:

        model_name = f"convnextv2_{model_cfg.type}"

        model = timm.create_model(model_name, pretrained=True, num_classes= model_cfg.num_classes, in_chans= model_cfg.in_chans, output_stride= model_cfg.output_stride,
                                  global_pool= model_cfg.global_pool, drop_rate= model_cfg.drop_rate, drop_path_rate= model_cfg.drop_path_rate)
        
        return model


def make_model(pick, cfg, device):
    model = select_model(pick, cfg)
    torch.cuda.empty_cache()
    model = model.to(device)
    model.eval()
    
    if pick in [0, 1, 2, 3, 4, 5]:
        with torch.no_grad():
            input_fake = torch.rand(1, 3, 224, 224).to(device)
            flops = FlopCountAnalysis(model, input_fake).total()
            nparams_hot = count_params(model.eval())
            print(f"Trainable Parameters {nparams_hot / 1e6} M")
            print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    
    # Standard checkpoint load
    if cfg.get("ckpt"):
        try:
            weights = torch.load(cfg["ckpt"], map_location='cpu')
            model = load_weights(model, weights)
            print("Successfully loaded weights from", cfg["ckpt"])
        except Exception as e:
            print("Failed to load pretrained weights", cfg["ckpt"], "->", e)
    
    # Transformer checkpoint load with pos_embed interpolation (needed if image size differs)
    elif cfg.get('ckpt_transformers'):
        try:
            checkpoint = torch.load(cfg["ckpt_transformers"], map_location="cpu")

            # Conditional options to extract state_dict from various checkpoint formats
            if 'model_state_dict' in checkpoint:
                
                state_dict = checkpoint['model_state_dict']
                print("Loaded checkpoint with model_state_dict")
            elif 'state_dict' in checkpoint:
                
                state_dict = checkpoint['state_dict']
                print("Loaded checkpoint with state_dict")
            elif 'model' in checkpoint:
                
                state_dict = checkpoint['model']
                print("Loaded checkpoint with model")
            else:
                
                state_dict = checkpoint
                print("Loaded checkpoint as direct state_dict")
            
            # Verify and interpolate pos_embed if necessary
            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']
                
                print(f"Checkpoint pos_embed shape: {pos_embed_checkpoint.shape}")
                print(f"Model pos_embed shape: {model.pos_embed.shape}")
                
                if pos_embed_checkpoint.shape != model.pos_embed.shape:
                    
                    pos_embed_checkpoint = state_dict.pop('pos_embed')
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    
                    # Interpolate pos_embed
                    interpolated_pos_embed = interpolate_pos_embed(model, pos_embed_checkpoint)
                    model.pos_embed.data.copy_(interpolated_pos_embed)
                    
                    print(f"Interpolated pos_embed from {pos_embed_checkpoint.shape} to {model.pos_embed.shape}")
                else:
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    print("pos_embed dimensions match, no interpolation needed")
            else:
                # No pos_embed found in checkpoint
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print("No pos_embed found in checkpoint")
            
            # Missing and unexpected keys report
            if missing:
                print(f"Missing keys ({len(missing)}):", missing[:5], "..." if len(missing) > 5 else "")
            if unexpected:
                print(f"Unexpected keys ({len(unexpected)}):", unexpected[:5], "..." if len(unexpected) > 5 else "")
            
            print("Successfully loaded and interpolated weights")
            
        except Exception as e:
            print(f"Failed to load pretrained weights from {cfg['ckpt_transformers']}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    return model