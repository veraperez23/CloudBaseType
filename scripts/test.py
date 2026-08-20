import os
import json
from scripts.val import validate

def test(model, test_dataloader, modelname, device, mode_test=False, save_errors=False, save_predictions=False):
    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # Obtenemos los resultados incluyendo la lista de detalles
    test_results = validate(model, test_dataloader, device, use_wandb=False, mode_test=mode_test, save_errors=save_errors)
    
    # Guardamos el resumen en el diccionario
    results[modelname] = {
        "loss": test_results["loss"],
        "accuracy": test_results["accuracy"],
        "rmse": test_results["rmse"],
        "mae": test_results["mae"],
        "std": test_results["std"]
    }

    labels = test_results["labels"]
    preds = test_results["preds"]

    with open(json_dir, "w") as f:
        json.dump(results, f, indent=4)

    # --- ARCHIVO 1: Resumen General (results_test.txt) ---
    fmt_resumen = "{0:25} | {1:15} | {2:15} | {3:15} | {4:15} | {5:15}\n"
    s_resumen = fmt_resumen.format("Modelo", "Loss", "Accuracy", "RMSE", "MAE", "STD")
    s_resumen += "-" * 63 + "\n"
    for k, v in results.items():
        s_resumen += fmt_resumen.format(k, f"{v['loss']:.6f}", f"{v['accuracy']:.6f}", f"{v['rmse']:.6f}", f"{v['mae']:.6f}", f"{v['std']:.6f}")
    
    with open(os.path.join(os.getcwd(), 'results_test.txt'), "w") as f:
        f.write(s_resumen)

    # --- ARCHIVO 2: Predicciones Individuales (predicciones_detalladas.txt) ---
    fmt_detallado = "{0:40} | {1:15} | {2:15} \n"
    s_detallado = fmt_detallado.format("Nombre de Imagen", "Real (m)", "Predicho (m)")
    s_detallado += "-" * 90 + "\n"
    
    for i, (real, pred) in enumerate(zip(labels, preds)):
        
        
        s_detallado += fmt_detallado.format(
            f"Imagen_{i}", 
            f"{real}", 
            f"{pred}"
        )

    with open(os.path.join(os.getcwd(), 'predicciones_detalladas.txt'), "w") as f:
        f.write(s_detallado)

    print(f"¡Test completado!")
    print(f"-> Resumen en: results_test.txt")
    print(f"-> Lista completa en: predicciones_detalladas.txt")