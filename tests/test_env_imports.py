def test_environment_symbols_import():
    from tdrp.envs import EnvConfig, RewardConfig, TruckMultiDroneCleanEnv, UncertaintyConfig

    cfg = EnvConfig()
    cfg.reward = RewardConfig()
    cfg.uncertainty = UncertaintyConfig(enabled=False)

    assert TruckMultiDroneCleanEnv is not None
    assert cfg.ndrones >= 1
