from trainer import register_and_load_trainer
import warnings

warnings.filterwarnings("ignore")

def train(train_data_path, test_data_path, data_name):
	trainer = register_and_load_trainer(train_data_path, test_data_path, data_name)
	trainer.train(trainer.iter, trainer.max_iter)
	

if __name__ == "__main__":
    DATA_NAME = "GWD-X101"
    TRAIN_DATA_PATH = "../Dataset/GlobalWheatDetection/converted/train"
    TEST_DATA_PATH = "../Dataset/GlobalWheatDetection/converted/test"
    train(TRAIN_DATA_PATH, TEST_DATA_PATH, DATA_NAME)
