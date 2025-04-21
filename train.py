from dataloader import ArtImageDataset, TupleDataset, train_test_ai_vs_all, collate_with_embeddings
from discriminator import CLIPSVMDiscriminator
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.svm import SVC
from generator import NoiseGenerator
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

VIS_DIR = "/ocean/projects/cis250019p/kanand/IDL Image Generation/embedding_plots"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = CLIPSVMDiscriminator(device=device)
    generator = NoiseGenerator().to(device)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    csv_meta = "/ocean/projects/cis250019p/kanand/IDL Image Generation/images_hidden_state_embedding.csv"
    dataset  = ArtImageDataset(csv_meta, ai_only=False)  # we need *all* rows
    X_train, X_test, y_train, y_test = train_test_ai_vs_all(dataset)

    print(len(X_train), "AI images in train set")
    print(len(X_test),  "images in test set (AI + Real)")
    print("first X tuple shapes:", X_train[0][0].shape, X_train[0][1].shape)

    if X_train and X_test:
        train_ds = TupleDataset(X_train, y_train)        # AI‑only   (label==0)
        test_ds = TupleDataset(X_test, y_test)           # AI + Real
        train_loader = DataLoader(train_ds, batch_size  = 64, shuffle= True, collate_fn =collate_with_embeddings)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn = collate_with_embeddings)
        print("DataLoaders created successfully")
    else:
        print("Error: Unable to create DataLoaders due to lack of valid samples")

    train_emb = np.stack([emb for emb, img in X_train], axis=0)    
    train_lbl = np.array(y_train, dtype=np.int64) 
    
    train_emb_ai = [emb for emb, _ in X_train]     
    train_lbl_ai = [lbl for lbl in y_train]         
    test_emb_all = [emb for emb, _ in X_test]       
    test_lbl_all = [lbl for lbl in y_test]

    svm_X = np.stack(train_emb_ai + test_emb_all, axis=0).astype(np.float32)
    svm_y = np.array(train_lbl_ai + test_lbl_all, dtype=np.int64)

    discriminator.train_svm(svm_X, svm_y)
    print("discriminator trained - layers now frozen for training of generator")

    orig_test_emb = np.stack([emb for emb, _ in X_test], axis=0)    # (N_test, 768)
    orig_test_lbl = np.array(y_test, dtype=np.int64) 
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        generator.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        pert_list = []
        all_probs = []
        all_preds = []
        for batch in train_bar:
            
            imgs = batch["image"].to(device)
            # print(imgs.shape)

            perturbed, noise = generator(imgs)
            mse = F.mse_loss(perturbed, imgs)
            reg= noise.pow(2).mean()

            pil_batch = [ to_pil_image(img.cpu()) for img in perturbed ]
            emb_batch = np.stack([ discriminator.run_clip(pil, is_path=False) for pil in pil_batch ], axis=0)
            pert_list.append(emb_batch)
            preds, probs = discriminator.predict_from_embeddings(emb_batch)  # np.ndarray shape (B,)
            all_probs.extend(probs)
            if len(probs) > 0:
                adv_loss = -torch.tensor(probs.mean(), device=device)
            else:
                adv_loss = torch.tensor(0.0, device=device)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

        # evaluate how well the perturbed AI images fool the SVM 
        generator.eval()
        all_probs = []
       
        avg_prob  = np.mean(all_probs) if all_probs else 0.0
        fool_rate = np.mean(all_preds) if all_preds else 0.0

        print(f"Epoch {epoch+1}: Avg SVM ‘real’ prob on AI samples = {avg_prob:.4f}")
        print(f"Epoch {epoch+1}: Fool rate (AI→Real) = {fool_rate:.4%}")
        pert_emb = np.concatenate(pert_list, axis=0)  # shape (N_test, 768)
        tsne_path = os.path.join(VIS_DIR, f"tsne_epoch_{epoch+1}.png")
        visualize_embeddings(
                embeddings        = orig_test_emb,
                labels            = orig_test_lbl,
                method            = 'tsne',
                title             = f"t-SNE after Epoch {epoch+1}",
                save_path         = tsne_path,
                perturbed_embeddings = pert_emb
            )
        pca_path = os.path.join(VIS_DIR, f"pca_epoch_{epoch+1}.png")
        visualize_embeddings(
                embeddings        = orig_test_emb,
                labels            = orig_test_lbl,
                method            = 'pca',
                title             = f"PCA after Epoch {epoch+1}",
                save_path         = pca_path,
                perturbed_embeddings = pert_emb
            )
        print(f"Saved epoch {epoch+1} embeddings to {tsne_path} and {pca_path}")

        loss = mse + 0.1 * reg + 0.1 * adv_loss
        gen_optimizer.zero_grad()
        loss.backward()
        gen_optimizer.step()

##final testing
    ai_idxs       = np.where(orig_test_lbl == 0)[0]
    orig_ai_embs  = orig_test_emb[ai_idxs]            
    noised_embs = []
    for idx in tqdm(ai_idxs, desc="Noising AI test samples"):
        # X_test[idx] was (embedding, image); we only need the image
        _, img_np = X_test[idx]
        img_tensor = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).to(device)  # (1,3,H,W)

        with torch.no_grad():
            perturbed, _ = generator(img_tensor)

        pil = to_pil_image(perturbed[0].cpu())
        emb = discriminator.run_clip(pil)             # (768,)
        noised_embs.append(emb)

    noised_embs = np.stack(noised_embs, axis=0)       # (N_ai_test, 768)

    # 3) Build evaluation set: originals label=0, noised label=1
    X_eval = np.vstack([orig_ai_embs, noised_embs])   # (2*N_ai_test, 768)
    y_eval = np.concatenate([
        np.zeros(len(orig_ai_embs), dtype=np.int64),  # originals
        np.ones (len(noised_embs),  dtype=np.int64)   # noised
    ])

    # 4) Run the frozen SVM on this “AI vs. Noised‑AI” task
    print("\n=== SVM on Original‑vs‑Noised AI Embeddings ===")
    metrics = discriminator.evaluate(X_eval, y_eval)

if __name__=='__main__':
    main()