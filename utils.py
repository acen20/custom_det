import json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

def load_dataset_desc(train_data_path):
	with open(f"{train_data_path}/train.json") as f:
		ann = json.load(f)
	classes = ann['categories']
	classes_ = [class_['name'] for class_ in classes]
	return len(ann['images']), classes_


def register_data(train_data_path, test_data_path):
	## LOAD PARAMETERS RELATED TO DATASET	
	num_images, classes_ = load_dataset_desc(train_data_path)

	### CLEAR REGISTERS
	DatasetCatalog.clear()
	MetadataCatalog.clear()

	## REGISTER THE DATASET
	register_coco_instances("custom_train",{},
							f"{train_data_path}/train.json",
							f"{train_data_path}")
	register_coco_instances("custom_test",{},
							f"{test_data_path}/test.json",
							f"{test_data_path}")
	
	MetadataCatalog.get("custom_train").thing_classes=classes_
	MetadataCatalog.get("custom_train").thing_colors=[(1,0,0,1)]
	custom_metadata = MetadataCatalog.get("custom_train")
	return num_images, classes_, custom_metadata