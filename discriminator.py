import torch
from transformers import CLIPProcessor, CLIPModel

from sklearn.svm import SVC
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class CLIPSVMDiscriminator:
    """
    A discriminator that leverages the CLIP model for feature extraction and an SVM classifier
    to label an image as real (human–created art) or fake (AI–generated art, even after adding noise).
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        # Set device to GPU if available.
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running on:", "cuda" if torch.cuda.is_available() else "cpu")

        # Load the CLIP model and processor.
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Initialize the SVM classifier.
        # Here we use a linear kernel and probability estimates.
        self.svm = SVC(kernel="linear", C=1.0, probability=True)
        self.svm_trained = False

    def run_clip(self, image_path, is_path=True):
      if is_path:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if img.size != (512, 512):
                print(f"Resizing {image_path} to 512x512")
                img = img.resize((512, 512))
                img.save(image_path)   
      else:
        img=image_path    
      # inputs = processor(images=img, return_tensors="pt", padding=True, truncation=True)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
      processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
      model.to(device)
      inputs = processor(images=img, return_tensors="pt")
      inputs = inputs.to(device)
      with torch.no_grad():
        # image_features = model.get_image_features(**inputs)
        outputs = model.vision_model(inputs.pixel_values)
        image_features = outputs.last_hidden_state[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.squeeze().cpu().numpy()

    def train_svm(self, X_train, y_train):
        """
        Train the internal SVM classifier using the provided features and labels.

        Parameters:
            X_train (np.ndarray): Training feature vectors.
            y_train (np.ndarray): Corresponding labels.

        Returns:
            The trained SVM classifier.
        """
        self.svm.fit(X_train, y_train)
        self.svm_trained = True
        train_accuracy = self.svm.score(X_train, y_train)
        print(f"Training accuracy for discriminator: {train_accuracy:.4f}")
        return self.svm


    def predict_from_embeddings(self, embeddings):
        """
        Given a numpy array of embeddings (shape: [B, 768]), use the trained SVM to predict labels
        and output probabilities that the embedding is from a "real" image.
        """
        preds = self.svm.predict(embeddings)
        probs = self.svm.predict_proba(embeddings)[:, 1]
        return preds, probs

    def evaluate(self, X_test, y_test):
        model=self.svm
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)

        # Calculate mean average precision (mAP)
        ap_per_class = []
        for class_label in np.unique(y_test):
            y_test_binary = (y_test == class_label).astype(int)
            ap = average_precision_score(y_test_binary, y_pred_proba)
            ap_per_class.append(ap)
        map_score = np.mean(ap_per_class)

        # Print results
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


def evaluate_generator_with_discriminator(generator, fixed_discriminator, ai_dataloader, device):
    """
    Evaluate the generator's outputs by passing the generated (perturbed) embeddings to
    the fixed SVM discriminator. This function computes the average SVM probability
    that the perturbed embeddings are classified as "real."

    This version only evaluates on AI-generated samples.
    """
    generator.eval()
    all_probs = []
    all_preds = []
    original_embeddings = []
    perturbed_embeddings = []

    with torch.no_grad():
        for batch in ai_dataloader:
            x = batch["features"]
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)


            original_embeddings.append(x.cpu().numpy())

            # Generate perturbed embeddings
            x_noisy, _ = generator(x)

            # Keep perturbed embeddings for visualization
            perturbed_embeddings.append(x_noisy.cpu().numpy())


            x_noisy_np = x_noisy.cpu().numpy()

            preds, probs = fixed_discriminator.predict_from_embeddings(x_noisy_np)

            all_preds.extend(preds)
            all_probs.extend(probs)

    if len(all_probs) > 0:
        avg_prob = np.mean(all_probs)
        fool_rate = np.mean(np.array(all_preds) == 1)  # Rate at which embeddings are classified as "real"

        print(f"Evaluation - Average SVM 'real' probability: {avg_prob:.4f}")
        print(f"Evaluation - Fool rate (classified as real): {fool_rate:.4f}")

        # Concatenate all embedding arrays
        original_embeddings = np.vstack(original_embeddings) if original_embeddings else np.array([])
        perturbed_embeddings = np.vstack(perturbed_embeddings) if perturbed_embeddings else np.array([])

        return avg_prob, fool_rate, original_embeddings, perturbed_embeddings
    else:
        print("No AI-generated samples to evaluate")
        return 0, 0, None, None