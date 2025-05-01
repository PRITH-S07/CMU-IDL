import os
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from models.dataset import ArtEmbeddingDataset, AITestDataset, string_to_np
from models.discriminator import CLIPSVMDiscriminator
from models.generator import NoiseGenerator
from models.training import (
    train_generator_with_selective_noise,
    evaluate_generator_with_discriminator,
)
from models.visualization import visualize_embeddings


def main():
    ROOT = "results"
    os.makedirs(ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    CSV_PATH = (
        "/content/drive/MyDrive/IDL Image Generation/images_hidden_state_embedding.csv"
    )
    df = pd.read_csv(CSV_PATH)
    X = np.stack(df["Features"].apply(string_to_np).to_numpy()).astype(np.float32)
    y = df["Label"].to_numpy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    discriminator = CLIPSVMDiscriminator(device=device)
    generator = NoiseGenerator().to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Visualize original embeddings
    visualize_embeddings(
        X,
        y,
        method="tsne",
        title="Original Embeddings (t-SNE)",
        save_path=os.path.join(ROOT, "original_tsne.png"),
    )

    # Train discriminator
    discriminator.train_svm(X_train, y_train)

    # Dataloaders
    full_dataset = ArtEmbeddingDataset(CSV_PATH)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

    ai_test_indices = np.where(y_test == 0)[0]
    X_test_ai = X_test[ai_test_indices]
    y_test_ai = y_test[ai_test_indices]
    ai_test_dataset = AITestDataset(X_test_ai, y_test_ai)
    ai_test_loader = DataLoader(ai_test_dataset, batch_size=32, shuffle=False)

    # Train generator
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}")
        train_generator_with_selective_noise(
            generator, discriminator, full_loader, optimizer, device
        )

        avg_prob, fool_rate, orig_embeds, pert_embeds = (
            evaluate_generator_with_discriminator(
                generator, discriminator, ai_test_loader, device
            )
        )

        if epoch == 2 or epoch == 4:
            visualize_embeddings(
                embeddings=X,
                labels=y,
                method="tsne",
                title=f"Embeddings After Epoch {epoch+1}",
                save_path=os.path.join(ROOT, f"embeddings_epoch_{epoch+1}.png"),
                perturbed_embeddings=pert_embeds,
            )

    # Final evaluation
    print("\nFinal Evaluation:")
    discriminator.evaluate(X_test, y_test)

    # Save models
    torch.save(generator.state_dict(), os.path.join(ROOT, "noise_generator.pth"))
    joblib.dump(discriminator.svm, os.path.join(ROOT, "clip_svm_discriminator.joblib"))


if __name__ == "__main__":
    main()
