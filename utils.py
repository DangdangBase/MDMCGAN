import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_UCIHAR_dataloader(algorithm, non_iid, batch_size, **kwargs):
    uci_har_folder = "UCI_HAR/filtered_data"
    non_iid_str = "non_iid" if non_iid else "iid"
    arr = ["acc", "gyro"]

    if non_iid:
        remove_labels_num = kwargs["remove_labels_num"]
        filter_ratio = kwargs["filter_ratio"]

        data = np.load(
            f"{uci_har_folder}/{non_iid_str}_{remove_labels_num}_{filter_ratio}.npz"
        )
    else:
        data = np.load(f"{uci_har_folder}/{non_iid_str}.npz")

    labels = torch.from_numpy(data["Y_train"])
    features = torch.from_numpy(data["X_train"]).chunk(2, dim=3)

    dataloader = []
    for idx, modal in enumerate(arr):
        if non_iid:
            cur_features = torch.cat(
                [features[idx], torch.from_numpy(data[f"{modal}_remain"])], dim=0
            )
            cur_labels = torch.cat(
                [labels, torch.from_numpy(data[f"y_remain_{modal}"])]
            )
        else:
            cur_features = features[idx]
            cur_labels = labels

        if algorithm == "mdmcgan":
            modals = torch.tensor([idx] * len(cur_features))
            cur_dataset = TensorDataset(cur_features, cur_labels, modals)
        else:
            cur_dataset = TensorDataset(cur_features, cur_labels)

        train_dataloader = DataLoader(
            dataset=cur_dataset, batch_size=batch_size, shuffle=True
        )

        dataloader.append(train_dataloader)

    return dataloader
