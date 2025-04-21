from loguru import logger

def test_rllib_trainer():
    try:
        import ray
        from backend.agents.rllib_trainer import train_with_rllib
        ray.init(ignore_reinit_error=True)
        train_with_rllib()
        ray.shutdown()
        logger.info("RLlib trainer ran successfully.")
    except ImportError as e:
        logger.warning(f"Ray or wandb not installed: {e}")
    except Exception as e:
        logger.error(f"RLlib trainer test failed: {e}")

if __name__ == "__main__":
    test_rllib_trainer() 