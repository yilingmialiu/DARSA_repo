"""
Empirical analysis version of WDGRL with clustering (DARSA).
Logs training time, GPU memory usage, forward/backward pass time,
total parameters, standard accuracy, and balanced accuracy on the target domain.
"""

import sys
sys.path.insert(0, '../../')
sys.path.append('/datacommons/carlsonlab/yl407/packages')

import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

# Local imports
from data.mnist_mnistm_data import load_mnist_LS, load_mnistm_LS, MNISTM
from models.model_mnist_mnistm import Net, Classifier
from utils.helper import set_requires_grad, seed_torch, sntg_loss_func, centroid_loss_func
from geomloss import SamplesLoss

from sklearn.metrics import balanced_accuracy_score, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------- #
#                          Utility / Helper functions                       #
# ------------------------------------------------------------------------- #

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

seed_new = 1234
seed_torch(seed=seed_new)

def train_with_analysis(
    batch_size: int,
    lr: float,
    K: int,
    lamb_clf: float,
    lamb_wd: float,
    lamb_centroid: float,
    lamb_sntg: float,
    epochs: int,
    LAMBDA: float,
    momentum: float,
    alpha: float,
    source_num: int,
    target_num: int,
    eval_target_num: int
):
    """
    Train WDGRL with clustering on MNIST -> MNIST-M, 
    measuring and logging time, memory, and performance.
    """

    # ------------------ Prepare/Load Models ------------------ #
    model_feature_source = Net().to(device)
    model_feature_target = Net().to(device)
    model_clf = Classifier().to(device)

    path_to_model = 'trained_models/'
    model_feature_source.load_state_dict(torch.load(path_to_model + f'source_feature_rd_128_SGD_alpha_{alpha}_v4.pt'))
    model_feature_target.load_state_dict(torch.load(path_to_model + f'source_feature_rd_128_SGD_alpha_{alpha}_v4.pt'))
    model_clf.load_state_dict(torch.load(path_to_model + f'source_clf_rd_128_SGD_alpha_{alpha}_v4.pt'))

    # Count total trainable parameters
    params_feature_source = count_parameters(model_feature_source)
    params_feature_target = count_parameters(model_feature_target)
    params_clf = count_parameters(model_clf)
    total_params = params_feature_source + params_feature_target + params_clf
    print(f"[INFO] Model Feature (Source) params: {params_feature_source}")
    print(f"[INFO] Model Feature (Target) params: {params_feature_target}")
    print(f"[INFO] Classifier params: {params_clf}")
    print(f"[INFO] Total trainable params: {total_params}")

    # ------------------ Prepare Data ------------------ #
    half_batch = batch_size // 2

    source_dataset = load_mnist_LS(source_num=source_num, alpha=alpha, train_flag=True)
    source_loader = DataLoader(
        source_dataset,
        batch_size=half_batch,
        drop_last=True,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    target_dataset_label = load_mnistm_LS(target_num=target_num, alpha=alpha, MNISTM=MNISTM, train_flag=True)
    target_loader = DataLoader(
        target_dataset_label,
        batch_size=half_batch,
        drop_last=True,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    eval_dataset = load_mnistm_LS(target_num=eval_target_num, alpha=alpha, MNISTM=MNISTM, train_flag=False)
    eval_dataloader_all = DataLoader(
        eval_dataset,
        batch_size=len(eval_dataset),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    # ------------------ Optimizers and Losses ------------------ #
    feature_source_optim = torch.optim.SGD(model_feature_source.parameters(), lr=lr, momentum=momentum)
    feature_target_optim = torch.optim.SGD(model_feature_target.parameters(), lr=lr, momentum=momentum)
    clf_optim = torch.optim.SGD(model_clf.parameters(), lr=lr, momentum=momentum)

    clf_criterion_weighted = nn.CrossEntropyLoss(reduction='none')
    clf_criterion_unweighted = nn.CrossEntropyLoss()
    w1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.05)

    # ------------------ Clustering Weights ------------------ #
    w_src = torch.ones(K, 1, requires_grad=False).div(K).to(device)
    w_tgt = torch.ones(K, 1, requires_grad=False).div(K).to(device)
    w_imp = torch.ones(K, 1, requires_grad=False).to(device)

    # ------------------ Logging Structures ------------------ #
    best_accu_t = 0.0

    # For peak memory usage
    peak_mem = 0.0

    # For timing
    total_train_start = time.time()

    # Per-epoch logs
    epoch_times = []
    epoch_forward_backward_times = []
    accuracy_log = []
    balanced_accuracy_log = []

    # ========================= Training Loop ========================= #
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch [{epoch}/{epochs}] (alpha={alpha}) ===")

        epoch_start_time = time.time()

        source_batch_iter = iter(source_loader)
        target_batch_iter = iter(target_loader)
        len_dataloader = min(len(source_loader), len(target_loader))

        total_unweight_clf_loss = 0
        total_clf_loss = 0
        total_centroid_loss = 0
        total_sntg_loss = 0
        total_w1_loss = 0
        total_w1_original = 0

        # --------------------- Mini-batch Loop --------------------- #
        for _ in range(len_dataloader):
            # Track forward/backward pass start
            pass_start = time.time()

            # Fetch data
            data_source = next(source_batch_iter)
            source_x, source_y = data_source
            data_target = next(target_batch_iter)
            target_x, _ = data_target

            source_x, source_y = source_x.to(device), source_y.to(device, dtype=torch.int64)
            target_x = target_x.to(device)

            # Enable gradient
            set_requires_grad(model_feature_source, True)
            set_requires_grad(model_feature_target, True)
            set_requires_grad(model_clf, True)

            model_feature_source.train()
            model_feature_target.train()
            model_clf.train()

            # 1) Feature Extraction
            source_feature = model_feature_source(source_x)
            target_feature = model_feature_target(target_x)
            target_feature_2 = model_feature_target(target_x)

            # 2) Classification Loss
            source_preds = model_clf(source_feature)
            clf_loss_each = clf_criterion_weighted(source_preds, source_y)
            clf_loss_unweight_batch = clf_criterion_unweighted(source_preds, source_y)

            # For predicted clusters
            target_preds = model_clf(target_feature_2).argmax(dim=1)

            # 3) Clustering Info
            cluster_s = F.one_hot(source_y, num_classes=K).float()
            cluster_t = F.one_hot(target_preds, num_classes=K).float()

            # Weighted classification loss
            weighted_clf_err = cluster_s * clf_loss_each.unsqueeze(1)
            expected_clf_err = torch.mean(weighted_clf_err, dim=0)  # shape: [K]
            clf_loss_value = torch.sum(expected_clf_err.view(K, 1) * w_imp)

            # 4) Wasserstein Distance (cluster-based)
            wasserstein_distance = 0.0
            for cluster_id in range(K):
                cond_src = (source_y == cluster_id)
                cond_tgt = (target_preds == cluster_id)
                if cond_src.sum() > 0 and cond_tgt.sum() > 0:
                    wasserstein_distance += w_tgt[cluster_id] * w1_loss(
                        source_feature[cond_src],
                        target_feature[cond_tgt]
                    )

            # Reference W1 distance (not cluster-based)
            wasserstein_distance_all = w1_loss(source_feature, target_feature)

            # 5) Clustering & SNTG Loss
            centroid_loss_val = centroid_loss_func(K, device, source_y, target_preds, source_feature, target_feature)
            source_sntg_loss = sntg_loss_func(cluster=cluster_s, feature=source_feature, LAMBDA=LAMBDA)
            target_sntg_loss = sntg_loss_func(cluster=cluster_t, feature=target_feature, LAMBDA=LAMBDA)
            sntg_loss_val = source_sntg_loss + target_sntg_loss

            # 6) Combined Loss
            loss = (lamb_clf * clf_loss_value
                    + lamb_wd * wasserstein_distance
                    + lamb_centroid * centroid_loss_val
                    + lamb_sntg * sntg_loss_val)

            # 7) Update cluster weights (w_src, w_tgt, w_imp)
            with torch.no_grad():
                w_src_batch = cluster_s.mean(dim=0)
                w_tgt_batch = cluster_t.mean(dim=0)
                w_src = w_src * 0.0 + w_src_batch.view(K, 1) * 1.0
                w_tgt = w_tgt * 0.0 + w_tgt_batch.view(K, 1) * 1.0
                w_imp = w_tgt / w_src

                # Avoid zeros and infinities
                for i in range(K):
                    if w_src[i] == 0:
                        w_imp[i] = 2.0
                    elif torch.isinf(w_imp[i]):
                        w_imp[i] = 1.0

            # 8) Backprop
            feature_source_optim.zero_grad()
            feature_target_optim.zero_grad()
            clf_optim.zero_grad()
            loss.backward()
            feature_source_optim.step()
            feature_target_optim.step()
            clf_optim.step()

            # 9) Log iteration-level stats
            total_unweight_clf_loss += clf_loss_unweight_batch.item()
            total_clf_loss += clf_loss_value.item()
            total_centroid_loss += centroid_loss_val.item()
            total_sntg_loss += sntg_loss_val.item()
            total_w1_loss += wasserstein_distance.item()
            total_w1_original += wasserstein_distance_all.item()

            # Check GPU memory
            used_mem = torch.cuda.memory_allocated(device=device)
            peak_mem = max(peak_mem, used_mem)

            # End forward/backward timing
            pass_end = time.time()
            pass_time = pass_end - pass_start
            epoch_forward_backward_times.append(pass_time)

        # --------------------- End of Epoch --------------------- #
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        mean_clf_loss_val = total_clf_loss / len_dataloader
        mean_unweighted_clf_loss_val = total_unweight_clf_loss / len_dataloader
        mean_centroid_loss_val = total_centroid_loss / len_dataloader
        mean_sntg_loss_val = total_sntg_loss / len_dataloader
        mean_w1_loss_val = total_w1_loss / len_dataloader
        mean_w1_original_val = total_w1_original / len_dataloader

        print(
            f"[Epoch {epoch:03d}] "
            f"W1 Loss (cluster) = {mean_w1_loss_val:.4f}, "
            f"Source Clf (weighted) = {mean_clf_loss_val:.4f}, "
            f"Source Clf (unweighted) = {mean_unweighted_clf_loss_val:.4f}, "
            f"Centroid Loss = {mean_centroid_loss_val:.4f}, "
            f"SNTG = {mean_sntg_loss_val:.4f}, "
            f"Epoch Time = {epoch_time:.2f}s"
        )

        # --------------------- Evaluate on Target --------------------- #
        set_requires_grad(model_feature_source, False)
        set_requires_grad(model_feature_target, False)
        set_requires_grad(model_clf, False)

        model_feature_source.eval()
        model_feature_target.eval()
        model_clf.eval()

        # Evaluate with standard and balanced accuracy
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_eval, y_eval in eval_dataloader_all:
                x_eval, y_eval = x_eval.to(device), y_eval.to(device)
                h_t = model_feature_target(x_eval)
                y_pred = model_clf(h_t)
                preds = y_pred.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_eval.cpu().numpy())

        # Flatten arrays
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute standard accuracy
        standard_acc = accuracy_score(all_labels, all_preds)
        # Compute balanced accuracy
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)

        accuracy_log.append(standard_acc)
        balanced_accuracy_log.append(balanced_acc)

        # Print and track best accuracies
        print(f"  Standard Accuracy on target data: {standard_acc:.4f}")
        print(f"  Balanced Accuracy on target data: {balanced_acc:.4f}")

        if balanced_acc > best_accu_t:
            best_accu_t = balanced_acc
            best_standard_acc = standard_acc
        elif balanced_acc == best_accu_t:
            # If balanced accuracy is the same, check standard accuracy
            if standard_acc > best_standard_acc:
                best_standard_acc = standard_acc

        print(f"  Best Balanced Accuracy so far: {best_accu_t:.4f}")
        print(f"  Best Standard Accuracy so far: {best_standard_acc:.4f}")

        # Re-enable grad for next epoch
        set_requires_grad(model_feature_source, True)
        set_requires_grad(model_feature_target, True)
        set_requires_grad(model_clf, True)

    # --------------------- End of Training --------------------- #
    total_train_end = time.time()
    total_training_time = total_train_end - total_train_start

    # --------------------- Final Report --------------------- #
    print("\n======== Training Complete ========")
    print(f"Total epochs:                {epochs}")
    print(f"Total training time:         {total_training_time:.2f}s (~{total_training_time/60:.2f} min)")
    print(f"Peak GPU memory usage:       {peak_mem / (1024 ** 2):.2f} MB")  # Convert bytes to MB
    print(f"Best Standard Accuracy:      {best_standard_acc:.4f}")
    print(f"Best Balanced Accuracy:      {best_accu_t:.4f}")

    # Optionally, compute average time stats
    avg_epoch_time = np.mean(epoch_times)
    avg_pass_time = np.mean(epoch_forward_backward_times)

    print(f"Average time per epoch:          {avg_epoch_time:.2f}s")
    print(f"Average forward/backward time:   {avg_pass_time:.4f}s (per mini-batch)")

    # --------------------- Save Logs --------------------- #
    # Create a DataFrame to store logs
    log_df = pd.DataFrame({
        "Epoch": range(1, epochs + 1),
        "Epoch_Time_s": epoch_times,
        "Forward_Backward_Time_s": epoch_forward_backward_times[:epochs],  # Ensure matching length
        "Standard_Accuracy": accuracy_log,
        "Balanced_Accuracy": balanced_accuracy_log
    })

    # Save to CSV
    log_df.to_csv("training_logs.csv", index=False)
    print("Training logs saved to 'training_logs.csv'.")

    # Return logs if desired
    return {
        "epoch_times": epoch_times,
        "forward_backward_times": epoch_forward_backward_times,
        "peak_memory_MB": peak_mem / (1024 ** 2),
        "standard_accuracy_history": accuracy_log,
        "balanced_accuracy_history": balanced_accuracy_log,
        "best_standard_accuracy": best_standard_acc,
        "best_balanced_accuracy": best_accu_t,
        "total_params": total_params
    }


def main():
    """
    Main entry: run WDGRL with clustering (DARSA) + analysis logging.
    """
    # Hyperparameters
    K = 10
    batch_size = 1024
    lr = 0.01 
    lamb_clf = 0.8
    lamb_wd = 0.4
    lamb_centroid = 0.9
    lamb_sntg = 0.9
    LAMBDA = 30
    momentum = 0.5
    alpha = 8
    source_num = 600
    target_num = 100
    eval_target_num = 10
    epochs = 3000  # Adjust as needed

    logs = train_with_analysis(
        batch_size=batch_size,
        lr=lr,
        K=K,
        lamb_clf=lamb_clf,
        lamb_wd=lamb_wd,
        lamb_centroid=lamb_centroid,
        lamb_sntg=lamb_sntg,
        epochs=epochs,
        LAMBDA=LAMBDA,
        momentum=momentum,
        alpha=alpha,
        source_num=source_num,
        target_num=target_num,
        eval_target_num=eval_target_num
    )

    # Example: print summary from logs
    print("\n[Final Log Summary]")
    for key, value in logs.items():
        if isinstance(value, list):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
