from tqdm import tqdm

from moment.data.anomaly_detection_datasets import get_anomaly_detection_datasets

anomaly_detection_datasets = get_anomaly_detection_datasets(collection="TSB-UAD-Public")
KDD_datasets = [i for i in anomaly_detection_datasets if "KDD" in i]
ID_subset = [i for i in range(1, 251)]
ID_subset = [i for i in ID_subset if i not in [32, 140]]
ucr_anomaly_archive_subset = []
for dataset_path in tqdm(KDD_datasets):
    if int(dataset_path.split("/")[-1].split("_")[0]) in ID_subset:
        ucr_anomaly_archive_subset.append(dataset_path)
