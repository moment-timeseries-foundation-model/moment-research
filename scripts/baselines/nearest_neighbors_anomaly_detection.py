import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from moment.common import PATHS
from moment.data.dataloader import get_timeseries_dataloader
from moment.models.anomaly_nearest_neighbors import AnomalyNearestNeighbors
from moment.utils.anomaly_detection_metrics import get_anomaly_detection_metrics
from moment.utils.config import Config
from moment.utils.utils import control_randomness, parse_config

DATASETS_PREFIX = "/TimeseriesDatasets/anomaly_detection/TSB-UAD-Public/KDD21/"

SMALL_DATASETS = [
    "140_UCR_Anomaly_InternalBleeding4_1000_4675_5033.out",
    "176_UCR_Anomaly_insectEPG4_1300_6508_6558.out",
    "180_UCR_Anomaly_ltstdbs30791ES_20000_52600_52800.out",
    "193_UCR_Anomaly_s20101m_10000_35774_35874.out",
]
DATASETS = [
    "109_UCR_Anomaly_1sddb40_35000_52000_52620.out",
    "112_UCR_Anomaly_BIDMC1_2500_5400_5600.out",
    "115_UCR_Anomaly_CIMIS44AirTemperature3_4000_6520_6544.out",
    "117_UCR_Anomaly_CIMIS44AirTemperature5_4000_4852_4900.out",
    "120_UCR_Anomaly_ECG2_15000_16000_16100.out",
    "121_UCR_Anomaly_ECG3_15000_16000_16100.out",
    "130_UCR_Anomaly_GP711MarkerLFM5z4_4000_6527_6645.out",
    "131_UCR_Anomaly_GP711MarkerLFM5z5_5000_8612_8716.out",
    "140_UCR_Anomaly_InternalBleeding4_1000_4675_5033.out",
    "141_UCR_Anomaly_InternalBleeding5_4000_6200_6370.out",
    "149_UCR_Anomaly_Lab2Cmac011215EPG5_7000_17390_17520.out",
    "176_UCR_Anomaly_insectEPG4_1300_6508_6558.out",
    "179_UCR_Anomaly_ltstdbs30791AS_23000_52600_52800.out",
    "180_UCR_Anomaly_ltstdbs30791ES_20000_52600_52800.out",
    "181_UCR_Anomaly_park3m_60000_72150_72495.out",
    "182_UCR_Anomaly_qtdbSel1005V_4000_12400_12800.out",
    "183_UCR_Anomaly_qtdbSel100MLII_4000_13400_13800.out",
    "186_UCR_Anomaly_resperation1_100000_110260_110412.out",
    "192_UCR_Anomaly_s20101mML2_12000_35774_35874.out",
    "193_UCR_Anomaly_s20101m_10000_35774_35874.out",
    "194_UCR_Anomaly_sddb49_20000_67950_68200.out",
    "195_UCR_Anomaly_sel840mECG1_17000_51370_51740.out",
    "150_UCR_Anomaly_Lab2Cmac011215EPG6_7000_12190_12420.out",
    "151_UCR_Anomaly_MesoplodonDensirostris_10000_19280_19440.out",
    "152_UCR_Anomaly_PowerDemand1_9000_18485_18821.out",
    "157_UCR_Anomaly_TkeepFirstMARS_3500_5365_5380.out",
    "159_UCR_Anomaly_TkeepSecondMARS_3500_9330_9340.out",
    "162_UCR_Anomaly_WalkingAceleration5_2700_5920_5979.out",
    "163_UCR_Anomaly_apneaecg2_10000_20950_21100.out",
    "166_UCR_Anomaly_apneaecg_10000_12240_12308.out",
    "167_UCR_Anomaly_gait1_20000_38500_38800.out",
    "170_UCR_Anomaly_gaitHunt1_18500_33070_33180.out",
    "174_UCR_Anomaly_insectEPG2_3700_8000_8025.out",
    "196_UCR_Anomaly_sel840mECG2_20000_49370_49740.out",
    "198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out",
    "199_UCR_Anomaly_tiltAPB3_40000_114000_114370.out",
    "205_UCR_Anomaly_CHARISfive_9812_28995_29085.out",
    "207_UCR_Anomaly_CHARISten_3165_26929_26989.out",
    "209_UCR_Anomaly_Fantasia_19000_26970_27270.out",
    "212_UCR_Anomaly_Italianpowerdemand_8913_29480_29504.out",
    "223_UCR_Anomaly_mit14046longtermecg_74123_131200_131700.out",
    "226_UCR_Anomaly_mit14046longtermecg_96123_123000_123300.out",
    "242_UCR_Anomaly_tilt12744mtable_100000_104630_104890.out",
    "244_UCR_Anomaly_tilt12754table_100013_104630_104890.out",
    "248_UCR_Anomaly_weallwalk_2000_4702_4707.out",
    "250_UCR_Anomaly_weallwalk_2951_7290_7296.out",
]
SMALL_DATASETS = [DATASETS_PREFIX + d for d in SMALL_DATASETS]
DATASETS = [DATASETS_PREFIX + d for d in DATASETS]


def control_experiment_arguments(args):
    if args.dataset_names in SMALL_DATASETS:
        args.train_ratio = 0.4
        args.val_ratio = 0.3
        args.test_ratio = 0.3
    return args


def get_dataloaders(args):
    args.batch_size = 1024
    # Train datalaoder
    args.data_split = "train"
    args.data_stride_len = 1
    train_dataloader = get_timeseries_dataloader(args=args)
    # Val datalaoder
    args.data_split = "val"
    val_dataloader = get_timeseries_dataloader(args=args)
    # Test datalaoder
    args.data_split = "test"
    args.data_stride_len = args.seq_len
    args.shuffle = False
    test_dataloader = get_timeseries_dataloader(args=args)
    return train_dataloader, val_dataloader, test_dataloader


def anomaly_detection():
    DEFAULT_CONFIG_PATH = "../../configs/default.yaml"
    config = Config(
        config_file_path="../../configs/anomaly_detection/nearest_neighbors_train.yaml",
        default_config_file_path=DEFAULT_CONFIG_PATH,
    ).parse()
    control_randomness(config["random_seed"])
    args = parse_config(config)

    pbar = tqdm(DATASETS, total=len(DATASETS))
    for id, dataset_name in enumerate(pbar):
        pbar.set_description(f"Processing {dataset_name.split('/')[-1]}")

        args.dataset_names = dataset_name
        args = control_experiment_arguments(args)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args)

        model = AnomalyNearestNeighbors(args)
        model.fit(train_dataloader=train_dataloader)

        trues, labels = [], []
        for batch_x in tqdm(test_dataloader):
            labels.append(batch_x.labels)
            trues.append(batch_x.timeseries.float().numpy())

        trues = np.concatenate(trues, axis=0)
        outputs = model.reconstruct(x_enc=trues)

        trues = trues.squeeze().flatten()
        preds = outputs.reconstruction.squeeze().flatten()
        labels = np.concatenate(labels, axis=0).squeeze().flatten()

        anomaly_scores = (trues - preds) ** 2

        metrics = get_anomaly_detection_metrics(
            anomaly_scores=anomaly_scores, labels=labels
        )

        results_path = os.path.join(
            PATHS.RESULTS_DIR,
            "supervised_anomaly_detection",
            args.model_name,
            args.finetuning_mode,
        )

        if not os.path.exists(results_path):
            os.makedirs(results_path)

        results_df = pd.DataFrame(
            data=[
                "AnomalyNearestNeighbors",
                id,
                metrics.adjbestf1,
                metrics.raucroc,
                metrics.raucpr,
                metrics.vusroc,
                metrics.vuspr,
            ],
            index=[
                "Model name",
                "ID",
                "Adj. Best F1",
                "rAUCROC",
                "rAUCPR",
                "VUSROC",
                "VUSPR",
            ],
        )

        metadata = args.dataset_names.split("/")[-1].split("_")
        data_id, data_name = metadata[0], metadata[3]

        results_df.to_csv(
            os.path.join(results_path, f"results_{data_id}_{data_name}.csv")
        )


if __name__ == "__main__":
    anomaly_detection()
