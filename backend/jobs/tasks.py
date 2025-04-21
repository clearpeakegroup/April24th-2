from backend.core import finrl_wrapper, registry
from backend.db import crud
from backend.services.gradio_hooks import redis_cli

@celery_app.task(bind=True)
def train_agent(self, config_id):
    """Train a FinRL+LNN hybrid agent and stream progress to Redis."""
    cfg = crud.get_config(config_id)
    env = finrl_wrapper.create_env(cfg.tickers, cfg.start, cfg.end)
    agent = registry.get_agent(cfg.agent_name)
    model = agent.get_model(env, learning_rate=cfg.lr, batch_size=cfg.batch)
    for epoch in range(cfg.epochs):
        model.learn(total_timesteps=cfg.timesteps_per_epoch)
        progress = int(100 * (epoch + 1) / cfg.epochs)
        redis_cli.set(f"gradio:train:{self.request.id}", progress)
    path = f"models/{config_id}.zip"
    model.save(path)
    crud.save_agent_path(cfg.id, path) 