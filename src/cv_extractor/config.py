"""CV extractor and training config."""

BALL_VISION_CONFIG = {
    "input_shape": (224, 224, 3),
    "reserve_dim": 128,
    "attention_channels": 64,
    "pretrained": True,
}

CV_XBIN_TRAIN_CONFIG = {
    "dataset_dir": "./data_isaac",
    "output_dir": "./models/cv_reserve",
    "updates": 1000,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "x_loss_weight": 1.0,
    "distance_loss_weight": 1.0,
    "eval_every": 5,
    "seed": 0,
}
