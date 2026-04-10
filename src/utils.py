import yaml
from sklearn.metrics import classification_report, roc_auc_score

def load_config(config_path="config/model_params.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def full_evaluation(model_name, y_true, y_pred):
    print(f"\n--- {model_name} Evaluation Report ---")
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Heart Disease']))
    
    auc = roc_auc_score(y_true, y_pred)
    print(f"ROC_AUC Score: {auc:.4f}")