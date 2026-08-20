from xml.parsers.expat import model
import torch
from fvcore.nn import FlopCountAnalysis
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

def validate(model, val_dataloader_day, val_dataloader_night, device, use_wandb, mode_test=False, save_errors=False, 
             error_file="misclassified.csv", cm_file="confusion_matrix.png", save_predictions=False, 
             save_confusion_matrix=False, save_predictions_file="predictions.pth"):

    model.eval()

    total_correct = 0
    plus_minus_1_correct = 0
    total_samples = 0
    misclassified = []

    # DAY 
    all_labels_day, all_preds_day = [], []    
    all_pre_logits_day, all_logits_day = [], []
    all_paths_day = []  
    all_expected_values_day = []

    with torch.no_grad():
        for batch in val_dataloader_day:
            inputs, labels, paths = batch[0].to(device), batch[1].to(device), batch[2]
            features = model.forward_features(inputs)
            pre_logits = model.forward_head(features, pre_logits=True)
            logits = model.forward_head(features)
            pred_labels = torch.argmax(logits, dim=1)

            softmax_output = F.softmax(logits, dim=1)
            class_values = torch.arange(softmax_output.shape[1]).to(softmax_output.device).float()
            expected_value = torch.sum(softmax_output*class_values, dim=1)

            all_logits_day.append(logits.cpu())
            all_pre_logits_day.append(pre_logits.cpu())
            all_labels_day.extend(labels.cpu().tolist())
            all_preds_day.extend(pred_labels.cpu().tolist())
            all_paths_day.extend(paths)  
            all_expected_values_day.extend(expected_value.cpu().tolist())

            total_correct += (pred_labels == labels).sum().item()
            plus_minus_1_correct += sum(abs(p.item() - t.item()) <= 1 for p, t in zip(pred_labels, labels))
            total_samples += labels.size(0)

            if save_errors:
                for pred, true, path in zip(pred_labels.cpu(), labels.cpu(), paths):
                    if pred.item() != true.item():
                        misclassified.append([path, true.item(), pred.item()])

    all_pre_logits_day = torch.cat(all_pre_logits_day, dim=0)
    all_logits_day = torch.cat(all_logits_day, dim=0)

    data_day = {
        "prelogits": all_pre_logits_day,
        "logits": all_logits_day,
        "y_true": all_labels_day,
        "y_pred": all_preds_day,
        "paths": all_paths_day 
    }

    day_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    day_accuracy_plus_minus_1 = plus_minus_1_correct / total_samples if total_samples > 0 else 0.0

    if use_wandb:
        import wandb
        wandb.log({"Val Accuracy Day": day_accuracy})
        wandb.log({"Val Accuracy Day (±1)": day_accuracy_plus_minus_1})

    # NIGHT 
    total_correct = 0
    total_samples = 0
    plus_minus_1_correct = 0

    all_labels_night, all_preds_night = [], []    
    all_pre_logits_night, all_logits_night = [], []
    all_paths_night = []  
    all_expected_values_night = []

    with torch.no_grad():
        for batch in val_dataloader_night:
            inputs, labels, paths = batch[0].to(device), batch[1].to(device), batch[2]
            features = model.forward_features(inputs)
            pre_logits = model.forward_head(features, pre_logits=True)
            logits = model.forward_head(features)
            pred_labels = torch.argmax(logits, dim=1)

            softmax_output = F.softmax(logits, dim=1)
            class_values = torch.arange(softmax_output.shape[1]).to(softmax_output.device).float()
            expected_value = torch.sum(softmax_output*class_values, dim=1)

            all_logits_night.append(logits.cpu())
            all_pre_logits_night.append(pre_logits.cpu())
            all_labels_night.extend(labels.cpu().tolist())
            all_preds_night.extend(pred_labels.cpu().tolist())
            all_paths_night.extend(paths)  
            all_expected_values_night.extend(expected_value.cpu().tolist())

            total_correct += (pred_labels == labels).sum().item()
            plus_minus_1_correct += sum(abs(p.item() - t.item()) <= 1 for p, t in zip(pred_labels, labels))
            total_samples += labels.size(0)

            if save_errors:
                for pred, true, path in zip(pred_labels.cpu(), labels.cpu(), paths):
                    if pred.item() != true.item():
                        misclassified.append([path, true.item(), pred.item()])

    accuracy_night = total_correct / total_samples if total_samples > 0 else 0.0
    accuracy_night_plus_minus_1 = plus_minus_1_correct / total_samples if total_samples > 0 else 0.0

    all_pre_logits_night = torch.cat(all_pre_logits_night, dim=0)
    all_logits_night = torch.cat(all_logits_night, dim=0)

    data_night = {
        "prelogits": all_pre_logits_night,
        "logits": all_logits_night,
        "y_true": all_labels_night,
        "y_pred": all_preds_night,
        "paths": all_paths_night  
    }

    if use_wandb:
        import wandb
        wandb.log({"Val Accuracy Night": accuracy_night})
        wandb.log({"Val Accuracy Night (±1)": accuracy_night_plus_minus_1})

    if save_errors and len(misclassified) > 0:
        df = pd.DataFrame(misclassified, columns=["image_path", "true_label", "pred_label"])
        df.to_csv(error_file, index=False)
        print(f"Se guardaron {len(misclassified)} errores en {error_file}")

    if mode_test:
        if save_predictions:
            torch.save(data_day, f"{save_predictions_file.replace('.pth', '_day.pth')}")
            torch.save(data_night, f"{save_predictions_file.replace('.pth', '_night.pth')}")
        
        # Calculate FLOPs and number of parameters
        B, C, H, W = inputs.shape
        input_fake = torch.rand(1, C, H, W).to(device)
        flops = FlopCountAnalysis(model, input_fake).total() / 1e9
        num_parameters = sum(p.numel() for p in model.parameters()) / 1e6

        all_labels = all_labels_day + all_labels_night
        all_preds = all_preds_day + all_preds_night

        # Confusion Matrix
        if save_confusion_matrix:
            cm = confusion_matrix(all_labels, all_preds).T
            n_classes = cm.shape[0]
            cm_sums = cm.sum(axis=0, keepdims=True)
            cm_percent = np.divide(cm.astype(np.float32), cm_sums, where=cm_sums!=0) * 100

            annot_labels = np.array([
                [f"{val:.0f}%" for val in row] 
                for row in cm_percent
            ])

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_percent, annot=annot_labels, fmt="",
                    cmap=sns.color_palette("RdBu_r", 12, as_cmap=True), alpha=0.8, annot_kws={"size": 15}, vmin=0, vmax=100,
                    xticklabels=[f"{j}" for j in range(n_classes)],
                    yticklabels=[f"{i}" for i in range(n_classes)],
                    cbar_kws={'format': '%.0f%%'})
            plt.xticks(fontsize=14)             
            plt.yticks(fontsize=14, rotation=0)
            plt.ylabel("Predicted label (oktas)", fontsize=14)
            plt.xlabel("Ground truth label (oktas)", fontsize=14)
            # plt.title("Test Dataset", fontsize=16)
            # plt.title("Validation Dataset", fontsize=16)
            plt.savefig(cm_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Confusion matrix saved at: {cm_file}")

        # Difference statistics

        all_expected_values = all_expected_values_day + all_expected_values_night

        diffs = np.array(all_expected_values) - np.array(all_labels)
        mae = np.mean(abs(diffs))
        rmse = np.sqrt(np.mean(diffs**2))
        mean_diff = np.mean(diffs)
        median_diff = np.median(diffs)
        std_diff = np.std(diffs)

        return {
            "accuracy": (day_accuracy + accuracy_night) / 2.0,
            "day_accuracy": day_accuracy,
            "night_accuracy": accuracy_night,
            "accuracy_plus_minus_1": (day_accuracy_plus_minus_1 + accuracy_night_plus_minus_1) / 2.0,
            "day_accuracy_plus_minus_1": day_accuracy_plus_minus_1,
            "night_accuracy_plus_minus_1": accuracy_night_plus_minus_1,
            "flops": flops,
            "params": num_parameters,
            "diff_mean": mean_diff,
            "diff_median": median_diff,
            "diff_std": std_diff,
            "mae": mae,
            "rmse": rmse
        }
    else:
        return (day_accuracy + accuracy_night) / 2.0






