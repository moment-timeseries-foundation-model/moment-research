import numpy as np
import torch
import wandb
from torch import nn, optim
from tqdm import tqdm, trange

from moment.common import PATHS
from moment.data.dataloader import get_timeseries_dataloader
from moment.models.dghl import DGHL
from moment.utils.config import Config
from moment.utils.utils import parse_config


def train(args, model, train_dataloader):
    n_train_epochs = args.max_epoch

    # Training loop
    tr_loss = 0

    optimizer = optim.AdamW(
        model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay
    )

    logger = wandb.init(project="Time-series Foundation Model", dir=PATHS.WANDB_DIR)

    for epoch in trange(n_train_epochs):
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            timeseries = batch.timeseries.float().to(args.device)
            input_mask = batch.input_mask.long().to(args.device).unsqueeze(1)

            # Training step
            model.train()
            loss = model.training_step(Y=timeseries, mask=input_mask)

            if not np.isnan(float(loss)):
                loss.backward()

            logger.log({"step_loss_train": loss.item()})

            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

            optimizer.step()
            optimizer.zero_grad()

            tr_loss += loss.detach().cpu().numpy()

    logger.finish()

    return model


def main():
    DEFAULT_CONFIG_PATH = "../../configs/default.yaml"
    GPU_ID = 1

    config = Config(
        config_file_path="../../configs/anomaly_detection/dghl_train.yaml",
        default_config_file_path=DEFAULT_CONFIG_PATH,
    ).parse()
    config["device"] = GPU_ID if torch.cuda.is_available() else "cpu"
    args = parse_config(config)

    model = DGHL(args).to(args.device)

    args.batch_size = args.train_batch_size
    train_dataloader = get_timeseries_dataloader(args=args)

    model = train(args, model, train_dataloader)

    with open("/home/scratch/mgoswami/dghl.pth", "wb") as f:
        torch.save(model.state_dict(), f)


if __name__ == "__main__":
    main()
