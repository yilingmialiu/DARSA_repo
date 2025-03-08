"""
Implements WDGRL with clustering and integrated analysis.
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""

###library loading###
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
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

seed_new = 1234
seed_torch(seed=seed_new)

def train(batch_size, lr, K, lamb_clf, lamb_wd, lamb_centroid, lamb_sntg, epochs, LAMBDA, momentum, alpha, source_num, target_num, eval_target_num, pseudo_label_threshold):

    best_accu_t = 0.0
    best_standard_acc = 0.0
    best_accu_t_epoch = 0
    best_standard_acc_epoch = 0
    peak_mem = 0.0
    total_train_start = time.time()
    epoch_times = []
    epoch_forward_backward_times = []
    accuracy_log = []
    balanced_accuracy_log = []
    weight_update = 1

    ##initialize models
    model_feature_source = Net().to(device)
    model_feature_target = Net().to(device)
    model_clf = Classifier().to(device)
    path_to_model = 'trained_models/'

    model_feature_source.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_alpha_'+str(alpha)+'_v3.pt'))
    model_feature_target.load_state_dict(torch.load(path_to_model+'source_feature_rd_128_SGD_alpha_'+str(alpha)+'_v3.pt'))
    model_clf.load_state_dict(torch.load(path_to_model+'source_clf_rd_128_SGD_alpha_'+str(alpha)+'_v3.pt'))

    params_feature_source = count_parameters(model_feature_source)
    params_feature_target = count_parameters(model_feature_target)
    params_clf = count_parameters(model_clf)
    total_params = params_feature_source + params_feature_target + params_clf
    print(f"[INFO] Model Feature (Source) params: {params_feature_source}")
    print(f"[INFO] Model Feature (Target) params: {params_feature_target}")
    print(f"[INFO] Classifier params: {params_clf}")
    print(f"[INFO] Total trainable params: {total_params}")

    half_batch = batch_size // 2
    ##########source data########################
    source_dataset = load_mnist_LS(source_num=source_num,alpha=alpha,train_flag=True)
    source_loader = DataLoader(source_dataset, batch_size=half_batch, drop_last=True, shuffle=True, num_workers=0, pin_memory=True)

    ##########target data########################
    target_dataset_label = load_mnistm_LS(target_num=target_num,alpha=alpha,MNISTM=MNISTM,train_flag=True)
    
    target_loader = DataLoader(target_dataset_label, batch_size=half_batch, drop_last=True, shuffle=True, num_workers=0, pin_memory=True)

    ##########evaluation data########################
    eval_dataset = load_mnistm_LS(target_num=eval_target_num,alpha=alpha,MNISTM=MNISTM,train_flag=False)

    eval_dataloader_all = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=True,
                            drop_last=True, num_workers=0, pin_memory=True)

    # Initial evaluation (optional)
    model_feature_source = model_feature_source.eval()
    model_feature_target = model_feature_target.eval()
    model_clf = model_clf.eval()
    
    all_preds_initial = []
    all_labels_initial = []
    with torch.no_grad():
        for x_eval, y_eval in eval_dataloader_all:
            x_eval, y_eval = x_eval.to(device), y_eval.to(device)
            h_t = model_feature_target(x_eval)
            y_pred = model_clf(h_t)
            preds = y_pred.argmax(dim=1)
            all_preds_initial.append(preds.cpu().numpy())
            all_labels_initial.append(y_eval.cpu().numpy())
    all_preds_initial = np.concatenate(all_preds_initial, axis=0)
    all_labels_initial = np.concatenate(all_labels_initial, axis=0)
    initial_accuracy = accuracy_score(all_labels_initial, all_preds_initial)
    initial_balanced_accuracy = balanced_accuracy_score(all_labels_initial, all_preds_initial)
    print(f'Initial accuracy on target eval data: {initial_accuracy:.4f}')
    print(f'Initial balanced accuracy on target eval data: {initial_balanced_accuracy:.4f}')

    ##initialize model optimizers
    feature_source_optim = torch.optim.SGD(model_feature_source.parameters(), lr=lr,momentum=momentum)
    feature_target_optim = torch.optim.SGD(model_feature_target.parameters(), lr=lr,momentum=momentum)
    clf_optim = torch.optim.SGD(model_clf.parameters(), lr=lr,momentum=momentum)

    scheduler_source = StepLR(feature_source_optim, step_size=5, gamma=0.9, verbose=True)
    scheduler_target = StepLR(feature_target_optim, step_size=5, gamma=0.9, verbose=True)
    scheduler_clf = StepLR(clf_optim, step_size=5, gamma=0.9, verbose=True)

    clf_criterion = nn.CrossEntropyLoss(reduction='none')
    clf_criterion_unweight = nn.CrossEntropyLoss()
    w1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=.05)
    soft_f = nn.Softmax(dim=1)

    ##initialize clustering weights
    w_src = torch.ones(K,1,requires_grad=False).divide(K).to(device)
    w_tgt = torch.ones(K,1,requires_grad=False).divide(K).to(device)
    w_imp = torch.ones(K,1,requires_grad=False).to(device)

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        forward_backward_times_epoch = []
        print(f"\n=== Epoch [{epoch}/{epochs}] (alpha={alpha}) ===")
        
        source_batch_iterator = iter(source_loader)
        target_batch_iterator = iter(target_loader)
        len_dataloader = min(len(source_loader), len(target_loader))

        total_unweight_clf_loss = 0
        total_clf_loss = 0
        total_centroid_loss = 0
        total_sntg_loss = 0
        total_w1_loss = 0
        total_w1_original = 0

        for i in range(len_dataloader):
            pass_start = time.time()

            data_source = next(source_batch_iterator)
            source_x, source_y = data_source
            data_target = next(target_batch_iterator)
            target_x, _ = data_target
            source_x, source_y = source_x.to(device), source_y.to(device, dtype=torch.int64)
            target_x = target_x.to(device)

            set_requires_grad(model_feature_source, requires_grad=True)
            set_requires_grad(model_feature_target, requires_grad=True)
            set_requires_grad(model_clf, requires_grad=True)

            model_feature_source = model_feature_source.train()
            model_feature_target = model_feature_target.train()
            model_clf = model_clf.train()

            ##extract latent features
            source_feature = model_feature_source(source_x)
            target_feature = model_feature_target(target_x)
            target_feature_2 = model_feature_target(target_x)

            ##unweighted classification loss
            source_preds = model_clf(source_feature)
            clf_loss_unweight = clf_criterion(source_preds, source_y)
            report_clf_loss_unweight = clf_criterion_unweight(source_preds, source_y)

            # --- Pseudo-label thresholding ---
            target_preds_probs = soft_f(model_clf(target_feature_2))
            max_probs, target_preds_raw = torch.max(target_preds_probs, dim=1)
            confident_mask = max_probs > pseudo_label_threshold
            
            target_preds = target_preds_raw[confident_mask]
            target_feature_masked = target_feature[confident_mask] # Apply mask to features as well

            ##get clustering information
            cluster_s = F.one_hot(source_y, num_classes=K).float()

            # Only create cluster_t if there are confident predictions
            cluster_t = F.one_hot(target_preds.to(torch.int64).to(device), num_classes=K).float() if len(target_preds) > 0 else torch.zeros((0, K)).to(device)

            ##weighted classification loss
            weighted_clf_err = cluster_s * clf_loss_unweight.reshape(-1,1)
            expected_clf_err = torch.mean(weighted_clf_err, dim=0)
            clf_loss = torch.sum(expected_clf_err.reshape(K,1) * w_imp)

            ##weighted domain invariant loss
            wasserstein_distance = 0
            if len(target_preds) > 0:
                for cluster_id in range(K):
                    source_mask = source_y == cluster_id
                    target_mask = target_preds == cluster_id
                    if torch.sum(target_mask) > 0 and torch.sum(source_mask) > 0:
                        wasserstein_distance += w_tgt[cluster_id] * w1_loss(
                            source_feature[source_mask,],
                            target_feature_masked[target_mask,]
                        )

            wasserstein_distance_all = w1_loss(source_feature,target_feature)
            ##clustering loss
            #L_orthogonal
            source_sntg_loss = sntg_loss_func(cluster=cluster_s,feature=source_feature,LAMBDA=LAMBDA)
            target_sntg_loss = torch.tensor(0.0).to(device) # Initialize to 0
            if len(cluster_t) > 0:
                target_sntg_loss = sntg_loss_func(cluster=cluster_t, feature=target_feature_masked, LAMBDA=LAMBDA)

            ##calculate centroids
            centroid_loss = torch.tensor(0.0).to(device)
            if len(target_preds) > 0:
                centroid_loss = centroid_loss_func(K, device, source_y, target_preds.to(torch.int64).to(device), source_feature, target_feature_masked)

            sntg_loss = source_sntg_loss + target_sntg_loss
            loss = lamb_clf*clf_loss + lamb_wd * wasserstein_distance + lamb_centroid*centroid_loss + lamb_sntg*sntg_loss

            #update weights
            with torch.no_grad():
                w_src_batch = cluster_s.mean(dim=0)
                w_tgt_batch = cluster_t.mean(dim=0)
                w_src = w_src * (1 - weight_update) + w_src_batch.reshape(K,1) * weight_update
                w_tgt = w_tgt * (1 - weight_update) + w_tgt_batch.reshape(K,1) * weight_update
                w_imp = w_tgt/w_src
                for k in range(w_imp.shape[0]):
                    if w_src[k] == 0:
                        w_imp[k] = 2.0
                    elif torch.isinf(w_imp[k]):
                        w_imp[k] = 1.0

            #backprop feature extraction+classifier
            feature_source_optim.zero_grad()
            feature_target_optim.zero_grad()
            clf_optim.zero_grad()
            loss.backward()
            feature_source_optim.step()
            feature_target_optim.step()
            clf_optim.step()

            pass_end = time.time()
            forward_backward_times_epoch.append(pass_end - pass_start)

            total_w1_loss += wasserstein_distance.item() if isinstance(wasserstein_distance, torch.Tensor) else 0
            total_unweight_clf_loss += report_clf_loss_unweight.item()
            total_clf_loss += clf_loss.item()
            total_centroid_loss += centroid_loss.item() if isinstance(centroid_loss, torch.Tensor) else 0
            total_sntg_loss += sntg_loss.item()
            total_w1_original += wasserstein_distance_all.item()

            used_mem = torch.cuda.memory_allocated(device=device)
            peak_mem = max(peak_mem, used_mem)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        epoch_forward_backward_times.extend(forward_backward_times_epoch)

        mean_clf_loss = total_clf_loss/(len_dataloader)
        mean_unweighted_clf_loss = total_unweight_clf_loss/(len_dataloader)
        mean_centroid_loss = total_centroid_loss/(len_dataloader)
        mean_sntg_loss = total_sntg_loss/(len_dataloader)
        mean_w1_loss = total_w1_loss/(len_dataloader)
        mean_w1_original = total_w1_original/(len_dataloader)

        tqdm.write(f'[Epoch {epoch:03d}] W1 Loss (cluster) = {mean_w1_loss:.4f}, Source Clf (weighted) = {mean_clf_loss:.4f}, Unweighted Source Clf = {mean_unweighted_clf_loss:.4f}, Centroid Loss = {mean_centroid_loss:.4f}, SNTG = {mean_sntg_loss:.4f}, Epoch Time = {epoch_time:.2f}s')

        ##evaluate models on target domain (Evaluation Set)##
        set_requires_grad(model_feature_source, requires_grad=False)
        set_requires_grad(model_feature_target, requires_grad=False)
        set_requires_grad(model_clf, requires_grad=False)

        model_feature_source = model_feature_source.eval()
        model_feature_target = model_feature_target.eval()
        model_clf = model_clf.eval()

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

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        standard_acc = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        accuracy_log.append(standard_acc)
        balanced_accuracy_log.append(balanced_acc)

        print(f"  Standard Accuracy on target data: {standard_acc:.4f}")
        print(f"  Balanced Accuracy on target data: {balanced_acc:.4f}")

        if balanced_acc > best_accu_t:
            best_accu_t = balanced_acc
            best_accu_t_epoch = epoch
        
        if standard_acc > best_standard_acc:
            best_standard_acc = standard_acc
            best_standard_acc_epoch = epoch
            
        print(f"  Best Balanced Accuracy so far: {best_accu_t:.4f} (achieved at epoch {best_accu_t_epoch})")
        print(f"  Best Standard Accuracy so far: {best_standard_acc:.4f} (achieved at epoch {best_standard_acc_epoch})")

        scheduler_source.step()
        scheduler_target.step()
        scheduler_clf.step()

    total_train_end = time.time()
    total_training_time = total_train_end - total_train_start

    print("\n======== Training Complete ========")
    print(f"Total epochs:                {epochs}")
    print(f"Total training time:         {total_training_time:.2f}s (~{total_training_time/60:.2f} min)")
    print(f"Peak GPU memory usage:       {peak_mem / (1024 ** 2):.2f} MB")
    print(f"Best Standard Accuracy:      {best_standard_acc:.4f} (epoch {best_standard_acc_epoch})")
    print(f"Best Balanced Accuracy:      {best_accu_t:.4f} (epoch {best_accu_t_epoch})")

    avg_epoch_time = np.mean(epoch_times)
    avg_pass_time = np.mean(epoch_forward_backward_times)

    print(f"Average time per epoch:          {avg_epoch_time:.2f}s")
    print(f"Average forward/backward time:   {avg_pass_time:.4f}s (per mini-batch)")

    log_df = pd.DataFrame({
        "Epoch": range(1, epochs + 1),
        "Epoch_Time_s": epoch_times,
        "Forward_Backward_Time_s": epoch_forward_backward_times[:epochs * len_dataloader],
        "Standard_Accuracy": accuracy_log,
        "Balanced_Accuracy": balanced_accuracy_log
    })
    log_df.to_csv("training_logs.csv", index=False)
    print("Training logs saved to 'training_logs.csv'.")

if __name__ == '__main__':
    # --- Set specific hyperparameters ---
    K = 10
    batch_size = 1024
    lr = 0.01  # Example: Set a specific learning rate
    lamb_clf = 1.0  # Example: Set a specific lambda_clf
    lamb_wd = 0.4  # Example: Set a specific lambda_wd
    lamb_centroid = 0.9
    lamb_sntg = 0.9
    LAMBDA = 30
    momentum = 0.5
    alpha = 8
    source_num = 600
    target_num = 100
    eval_target_num = 10
    pseudo_label_threshold = 0.1 #0.1 #0.1  # --- Set the pseudo-label threshold ---

    print("--- Starting Run with Specific Hyperparameters and Thresholding ---")
    print("lr", lr)
    print("momentum", momentum)
    print("K", K)
    print("batch_size", batch_size)
    print("lambda_clf", lamb_clf)
    print("lambda_wd", lamb_wd)
    print("lambda_centroid", lamb_centroid)
    print("lambda_sntg", lamb_sntg)
    print("LAMBDA", LAMBDA)
    print("alpha", alpha)
    print("pseudo_label_threshold", pseudo_label_threshold)

    train(batch_size=batch_size, lr=lr,
          K=K, lamb_clf=lamb_clf, lamb_wd=lamb_wd,
          lamb_centroid=lamb_centroid, lamb_sntg=lamb_sntg,
          epochs=3000,
          LAMBDA=LAMBDA, momentum=momentum,
          alpha=alpha,
          source_num=source_num, target_num=target_num,
          eval_target_num=eval_target_num,
          pseudo_label_threshold=pseudo_label_threshold) # Pass the threshold to the train function
    print("--- Finished Run ---")