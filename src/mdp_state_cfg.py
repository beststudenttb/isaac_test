"""MDP state representation config."""

MODEL_CFG = {
    "input_shape": (224, 224, 3),
    "z_dim": 32,
    "action_dim": 3,
    "belief_dim": 128,
    "feature_dim": 256,
    "hidden_dim": 128,
    "pretrained": True,
    "freeze_backbone": True,
}

TRAIN_CFG = {
    "dataset_dir": "./data_mdp",
    "output_dir": "./models/vision/mdp_state",
    "batch_size": 16,
    "seq_len": 16,
    "updates": 2000,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "dyn_w": 1.0,
    "reward_w": 1.0,
    "value_w": 0.1,
    "done_w": 0.2,
    "probe_w": 0.2,
    "var_w": 0.1,
    "save_every": 100,
    "seed": 0,
}
