import torch
import torch.nn.functional as F
import numpy as np

def train_generator_with_selective_noise(generator, fixed_discriminator, dataloader, optimizer, device, lambda_reg=0.1):
    generator.train()
    total_loss = 0.0
    total_fool_rate = 0.0
    batch_count = 0

    for batch in dataloader:
        x = batch["features"]
        labels = batch["label"]

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long)

        x = x.to(device)
        labels = labels.to(device)

        ai_indices = (labels == 0).nonzero(as_tuple=True)[0]
        if len(ai_indices) == 0:
            continue

        x_ai = x[ai_indices]
        x_noisy, noise = generator(x_ai)
        x_noisy_np = x_noisy.detach().cpu().numpy()

        _, probs = fixed_discriminator.predict_from_embeddings(x_noisy_np)
        probs_tensor = torch.tensor(probs, device=device)

        adversarial_loss = torch.mean(1 - probs_tensor)
        noise_magnitude = torch.mean(torch.norm(noise, dim=1))
        loss = adversarial_loss + lambda_reg * noise_magnitude

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        fool_rate = (probs > 0.5).mean()
        total_fool_rate += fool_rate
        batch_count += 1

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_fool_rate = total_fool_rate / batch_count if batch_count > 0 else 0
    print(f"Average Generator Loss: {avg_loss:.4f}, Fool Rate: {avg_fool_rate:.4f}")
    return avg_loss, avg_fool_rate

def evaluate_generator_with_discriminator(generator, fixed_discriminator, ai_dataloader, device):
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
            x_noisy, _ = generator(x)
            perturbed_embeddings.append(x_noisy.cpu().numpy())

            x_noisy_np = x_noisy.cpu().numpy()
            preds, probs = fixed_discriminator.predict_from_embeddings(x_noisy_np)

            all_preds.extend(preds)
            all_probs.extend(probs)

    if len(all_probs) > 0:
        avg_prob = np.mean(all_probs)
        fool_rate = np.mean(np.array(all_preds) == 1)
        print(f"Evaluation - Average SVM 'real' probability: {avg_prob:.4f}")
        print(f"Evaluation - Fool rate (classified as real): {fool_rate:.4f}")
        original_embeddings = np.vstack(original_embeddings) if original_embeddings else np.array([])
        perturbed_embeddings = np.vstack(perturbed_embeddings) if perturbed_embeddings else np.array([])
        return avg_prob, fool_rate, original_embeddings, perturbed_embeddings
    else:
        print("No AI-generated samples to evaluate")
        return 0, 0, None, None