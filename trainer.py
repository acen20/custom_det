from detectron2.engine import DefaultTrainer
import logging
from detectron2.utils.events import EventStorage
import json, os
from model_configs import get_custom_config
from eval import CustomEvalHook
from early_stopper import early_stopping
from utils import register_data


### Modified TrainerHook ###

class CustomTrainer(DefaultTrainer):

	def __init__(self,cfg):
		super(CustomTrainer,self).__init__(cfg)
		self.patience_iter = 0
		
	def train(self, start_iter: int, max_iter: int):

		logger = logging.getLogger(__name__)
		logger.info("Starting training from iteration {}".format(start_iter))

		self.iter = self.start_iter = start_iter
		self.max_iter = max_iter

		with EventStorage(start_iter) as self.storage:
			try:
				self.before_train()
				for self.iter in range(start_iter, max_iter):
					self.before_step()
					self.run_step()
					signal = self.after_step()
					if signal:
						break
				# self.iter == max_iter can be used by `after_train` to
				# tell whether the training successfully finished or failed
				# due to exceptions.
				self.iter += 1
			except Exception:
				logger.exception("Exception during training:")
				raise
			finally:
				self.after_train()
				
	def after_step(self):
		signal=None
		for h in self._hooks:
			signal = h.after_step()
		return signal




def register_and_load_trainer(train_data_path, test_data_path, data_name): ## ----> returns trainer
	## LOAD PARAMETERS RELATED TO DATASET	
	num_images, classes_, _ = register_data(train_data_path, test_data_path)
	
	cfg = get_custom_config(data_name)
	
	for dir_ in [cfg.OUTPUT_DIR, cfg.RESULTS_DIR]:
		os.makedirs(dir_, exist_ok=True)

	## set number of classes
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes_)

	## To run evaluation after 1 epoch
	EVAL_PERIOD = num_images // cfg.SOLVER.IMS_PER_BATCH
	PATIENCE = 5

	trainer = CustomTrainer(cfg)

	trainer.resume_or_load(resume=True)
	trainer.register_hooks([CustomEvalHook(EVAL_PERIOD, 
									lambda:early_stopping(cfg, trainer, PATIENCE))])
	
	print(f"NUM IMAGES PER BATCH: {cfg.SOLVER.IMS_PER_BATCH}")
	return trainer

