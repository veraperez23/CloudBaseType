import torch
import torch.nn as nn
import os
from tqdm import tqdm
import wandb
import gc
from scripts.val import validate 
from utils.utils import freeze_backbone_layers, unfreezing_scheduler

#CARPETA CHECKPOINTS
#ruta_carpeta = '/content/drive/MyDrive/TFG/checkpoints'
ruta_carpeta='./checkpoints'
os.makedirs(ruta_carpeta, exist_ok=True)


def train_regression(model, optimizer, scheduler, train_dataloader, val_dataloader, criterion, device, use_wandb=True, epochs=1000, 
            verbose = 20, modelname= 'model', use_amp= True, out_path='results/', patience=100, accum_steps=1, freeze_steps=5, freeze=False, warmup=False, refine_flag=False):

    steps = 0
    # En regresión buscamos la tasa de aciertos más alta
    best_val_accuracy = 0.0 
    early_stopping_counter = 0

    # Creamos la carpeta de resultados si no existe
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    scaler = torch.amp.GradScaler(enabled=use_amp) #entrena a la red más rápido

    # BUCLE PRINCIPAL DE ÉPOCAS. Una época es cuando la red ha visto todas las fotos una vez

    start_epoch=0
    ruta_ultimo_checkpoint = os.path.join(ruta_carpeta, "ultimo_checkpoint.pt")

    if os.path.exists(ruta_ultimo_checkpoint):
        print(f"Encontrado checkpoint previo. Reanudando...")
        checkpoint_guardado = torch.load(ruta_ultimo_checkpoint)
        
        # Restauramos los pesos
        model.load_state_dict(checkpoint_guardado['estado_modelo'])
        optimizer.load_state_dict(checkpoint_guardado['estado_optimizador'])
        start_epoch = checkpoint_guardado['epoca'] + 1
    else:
        print("No hay guardados previos. Empezando desde cero...")
    
    for epoch in range(start_epoch, epochs):

        if refine_flag:
            unfreezing_scheduler(model, epoch, epochs)

        if freeze == True and refine_flag == False:
            if epoch < freeze_steps:
                freeze_backbone_layers(model, freeze=True)
            else:
                freeze_backbone_layers(model, freeze=False)

        model.train() # Ponemos el modelo en modo entrenamiento
        running_loss = 0.0 # Guardará la pérdida total
        running_correct = 0
        total_samples = 0

        # BUCLE DE BATCHES (Lotes de imágenes)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = batch[0].to(device, non_blocking=True) #imágenes
            labels = batch[1].to(device, non_blocking=True) #clases

            try:
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    #predicción
                    logits = model(inputs)
                    
                    #Ajustamos las dimensiones de [batch_size, 1] a [batch_size]
                    logits = logits.squeeze(dim=1) 
                    
                    # Cálculos de error
                    loss = criterion(logits, labels)
                    loss = loss / accum_steps #accum_steps es para que el ordenador trate imágenes de menos en menos, por ejemplo grupos de 4.
                    #el error medio lo va a dividir entre este número de steps para asemejar a que has metido todas las fotos de golpe

                if torch.isnan(loss).any():
                    print("Warning: NaN loss encountered. Skipping batch.")
                    continue

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (steps + 1) % accum_steps == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                steps += 1

                pred_labels = torch.argmax(logits, dim=1)
                correct = (pred_labels == labels).sum().item()
                accuracy = correct / labels.size(0)

                #Weights & Biases
                if use_wandb:
                    wandb.log({"Loss": loss.item()})
                    wandb.log({"Accuracy": accuracy}) 

                running_loss += loss.item() * labels.size(0)
                running_correct += correct
                total_samples += labels.size(0)

            except Exception as e:
                print(f"Error during training step: {e}")
                continue
                
        # ACTUALIZAMOS LEARNING RATE
        if warmup == False:
            scheduler.step()
        else:
            scheduler.step(epoch)

        # ESTADÍSTICAS AL FINAL DE LA ÉPOCA
        epoch_loss = running_loss / total_samples if total_samples > 0 else float('nan')
        epoch_accuracy = running_correct / total_samples if total_samples > 0 else 0.0

        print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Train Accuracy = {epoch_accuracy:.4f}")

        # FASE DE VALIDACIÓN
        if epoch % verbose == 0 or epoch == epochs - 1:

            val_accuracy = validate(model, val_dataloader, device, use_wandb, mode_test=False, save_errors=False)

            if refine_flag:
                torch.save(model.state_dict(), os.path.join(out_path, f"{modelname}_{epoch}.pt"))

            else:

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    early_stopping_counter = 0 
                    print(f"New best validation accuracy: {best_val_accuracy:.4f}. Saving model.")
                    torch.save(model.state_dict(), os.path.join(out_path, f"{modelname}.pt"))
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= patience:
                        print("Early stopping triggered.")
                        break


        print(f"Época {epoch} terminada.")
        torch.save({
            'epoca': epoch,
            'estado_modelo': model.state_dict(),
            'estado_optimizador': optimizer.state_dict(),
            'precision':epoch_accuracy # Opcional: guardar el loss actual
        }, ruta_ultimo_checkpoint)
        
        gc.collect()
