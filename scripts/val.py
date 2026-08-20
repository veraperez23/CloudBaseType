import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Si quiero separar entre día y noche tengo que poner val_dataloader_day y val_dataloader_night
def validate(model, val_dataloader, device, use_wandb=False, mode_test=False, save_errors=False, cm_file="confusion_matrix.png", save_confusion_matrix= True):

    model.eval() #modelo en modo examen (congela el aprendizaje)

    # Función para evaluar cualquier dataloader (día o noche) y no repetir el código dos veces
    def evaluar_conjunto(dataloader, nombre_conjunto):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_labels = []
        all_preds = []
        all_expected_values = []
        misclassified = []

        # Le decimos a PyTorch que no calcule derivadas para ahorrar memoria
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
                nombres = batch[2] if len(batch) > 2 else [f"img_{i}" for i in range(labels.size(0))]

                logits = model(inputs)
                pred_labels = torch.argmax(logits, dim=1)

                loss_batch = F.cross_entropy(logits, labels, reduction='sum').item()
                total_loss += loss_batch

                # Calculamos el error sumado de todo este batch
                softmax_output = F.softmax(logits, dim=1) #traduce la salida del modelo a un porcentaje
                class_values = torch.arange(softmax_output.shape[1]).to(softmax_output.device).float()
                expected_value = torch.sum(softmax_output*class_values, dim=1)

                #Guardamos los datos para las métricas finales
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(pred_labels.cpu().tolist())
                all_expected_values.extend(expected_value.cpu().tolist())

                total_correct += (pred_labels == labels).sum().item()
                total_samples += labels.size(0)

                # Guardamos las predicciones fallidas si nos lo piden
                if save_errors:
                    for pred, true, path in zip(pred_labels.cpu(), labels.cpu(), nombres):
                        if pred.item() != true.item():
                            misclassified.append([path, true.item(), pred.item()])


        # Calculamos los promedios finales
        if total_samples > 0:
            loss_media = total_loss / total_samples
            accuracy = total_correct / total_samples
            
            diffs = np.array(all_expected_values) - np.array(all_labels)
            mae = np.mean(abs(diffs))
            rmse = np.sqrt(np.mean(diffs**2))
            std_diff = np.std(diffs)
        else:
            loss_media, accuracy, mae, rmse, std_diff = 0.0, 0.0, 0.0, 0.0, 0.0     


        # Guardamos los gráficos si usamos Weights & Biases
        if use_wandb:
            import wandb
            wandb.log({f"Val Accuracy {nombre_conjunto}": accuracy})
            wandb.log({f"Val Loss {nombre_conjunto}": loss_media})
            wandb.log({f"Val RMSE {nombre_conjunto}": rmse})

        return loss_media, accuracy, all_labels, all_preds, mae, rmse, std_diff, misclassified


    #Evalúo todos los datos juntos. Luego si eso puedo separar día/noche
    loss_global, accuracy_global, all_labels, all_preds, mae, rmse, std_diff, misclassified = evaluar_conjunto(val_dataloader, "Validación")    
    
    # Devolvemos resultados dependiendo de lo que haya pedido el bucle
    if mode_test:
        # Si es el examen final, devolvemos un diccionario con todos los detalles
        print(f"\n[Examen Final]")
        print(f"Accuracy (Aciertos): {accuracy_global*100:.2f}%")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Desviación Estándar: {std_diff:.4f}")

        if save_errors and len(misclassified) > 0:
            df = pd.DataFrame(misclassified, columns=["image_path", "true_label", "pred_label"])
            # NOTA: Cambié error_file por un nombre directo ya que la variable no estaba definida arriba
            df.to_csv("misclassified.csv", index=False)
            print(f"Se guardaron {len(misclassified)} errores en misclassified.csv")

        #matriz de confusión
        if save_confusion_matrix:
            cm = confusion_matrix(all_labels, all_preds).T
            n_classes = cm.shape[0]
            cm_sums = cm.sum(axis=0, keepdims=True)
            cm_percent = np.divide(cm.astype(np.float32), cm_sums, out=np.zeros_like(cm, dtype=float), where=cm_sums!=0) * 100

            annot_labels = np.array([[f"{val:.0f}%" for val in row] for row in cm_percent])

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_percent, annot=annot_labels, fmt="",
                    cmap=sns.color_palette("RdBu_r", 12, as_cmap=True), alpha=0.8, annot_kws={"size": 15}, vmin=0, vmax=100,
                    xticklabels=[f"{j}" for j in range(n_classes)],
                    yticklabels=[f"{i}" for i in range(n_classes)],
                    cbar_kws={'format': '%.0f%%'})
            plt.xticks(fontsize=12)             
            plt.yticks(fontsize=12, rotation=0)
            
            plt.ylabel("Clase Predicha", fontsize=14)
            plt.xlabel("Clase Real", fontsize=14)
            plt.title("Matriz de Confusión", fontsize=16)
            
            plt.savefig(cm_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Matriz de confusión guardada como: {cm_file}")
            print("-----------------------------\n")

        return {
            "loss": loss_global,
            "accuracy": accuracy_global,
            "rmse": rmse,
            "mae": mae,
            "std": std_diff,
            "labels": all_labels,  
            "preds": all_preds     
        }
    else:
        return accuracy_global


    #SI HAGO LA SEPARACIÓN EN DÍA Y NOCHE
    #mse_day, rmse_day = evaluar_conjunto(val_dataloader_day, "Día")
    #mse_night, rmse_night = evaluar_conjunto(val_dataloader_night, "Noche")

    # 3. Calculamos el error medio global
    #rmse_global = (rmse_day + rmse_night) / 2.0

    # 4. Devolvemos resultados dependiendo de lo que haya pedido el bucle
    # if mode_test:
    #     # Si es el examen final, devolvemos un diccionario con todos los detalles
    #     print(f"\n[Examen Final] RMSE Día: {rmse_day:.2f}m | RMSE Noche: {rmse_night:.2f}m")
    #     return {
    #         "rmse_global": rmse_global,
    #         "rmse_day": rmse_day,
    #         "rmse_night": rmse_night,
    #         "mse_day": mse_day,
    #         "mse_night": mse_night
    #     }
    # else:
    #     # Si estamos en medio del entrenamiento, solo devolvemos el error global
    #     # para que el 'train.py' sepa si tiene que guardar el modelo (Early Stopping)
    #     return rmse_global
