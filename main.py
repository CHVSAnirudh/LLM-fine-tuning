
from training.base_trainer import BaseTrainer

trainer, model_name = BaseTrainer(config_path="config/config.yaml").get_trainer()

trainer.train()
trainer.model.save_pretrained(model_name)