import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np

class CLIPSVMDiscriminator:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on:", self.device)
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.svm = SVC(kernel="linear", C=1.0, probability=True)
        self.svm_trained = False

    def run_clip(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.size != (512, 512):
                img = img.resize((512, 512))
                img.save(image_path)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.vision_model(inputs.pixel_values)
            image_features = outputs.last_hidden_state[:, 0, :]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.squeeze().cpu().numpy()

    def train_svm(self, X_train, y_train):
        self.svm.fit(X_train, y_train)
        self.svm_trained = True
        train_accuracy = self.svm.score(X_train, y_train)
        print(f"Training accuracy for discriminator: {train_accuracy:.4f}")
        return self.svm

    def predict_from_embeddings(self, embeddings):
        preds = self.svm.predict(embeddings)
        probs = self.svm.predict_proba(embeddings)[:, 1]
        return preds, probs

    def evaluate(self, X_test, y_test):
        model = self.svm
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        ap_per_class = []
        for class_label in np.unique(y_test):
            y_test_binary = (y_test == class_label).astype(int)
            ap = average_precision_score(y_test_binary, y_pred_proba)
            ap_per_class.append(ap)
        map_score = np.mean(ap_per_class)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"mAP: {map_score:.4f}")
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "map": map_score
        }