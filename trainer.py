from detectron2.engine import DefaultTrainer
import logging
from detectron2.utils.events import EventStorage
import json, os
from model_configs import get_custom_config
from eval import EvalHook
from early_stopper import early_stopping
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

### Modified TrainerHook ###

class DefaultTrainer(DefaultTrainer):

	def __init__(self,cfg):
		super(DefaultTrainer,self).__init__(cfg)
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
	
def load_dataset_desc(train_data_path):
	with open(f"{train_data_path}/train.json") as f:
		ann = json.load(f)
	classes = ann['categories']
	classes_ = [class_['name'] for class_ in classes]
	return len(ann['images']), classes_



def register_and_load_trainer(train_data_path, test_data_path): ## ----> returns trainer
	## LOAD PARAMETERS RELATED TO DATASET	
	num_images, classes_ = load_dataset_desc(train_data_path)

	### REGISTER DATASET
	DatasetCatalog.clear()
	MetadataCatalog.clear()
	register_coco_instances("custom_train",{},
							f"{train_data_path}/train.json",
							f"{train_data_path}")
	register_coco_instances("custom_test",{},
							f"{test_data_path}/test.json",
							f"{test_data_path}")


	cfg = get_custom_config()
	
	for dir_ in [cfg.OUTPUT_DIR, cfg.RESULTS_DIR]:
		os.makedirs(dir_, exist_ok=True)

	## set number of classes
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes_)

	## To run evaluation after 1 epoch
	EVAL_PERIOD = num_images // cfg.SOLVER.IMS_PER_BATCH
	PATIENCE = 3

	trainer = DefaultTrainer(cfg)

	trainer.resume_or_load(resume=False)
	trainer.register_hooks([EvalHook(EVAL_PERIOD, 
									lambda:early_stopping(cfg, trainer, PATIENCE))])
	
	print(f"NUM IMAGES PER BATCH: {cfg.SOLVER.IMS_PER_BATCH}")
	return trainer

