from trainer import register_and_load_trainer
import warnings

warnings.filterwarnings("ignore")

def train(train_data_path, test_data_path):
	trainer = register_and_load_trainer(train_data_path, test_data_path)
	trainer.train(trainer.iter, trainer.max_iter)
	

if __name__ == "__main__":
    TRAIN_DATA_PATH = "../Dataset/SPIKE Dataset/positive"
    TEST_DATA_PATH = "../Dataset/SPIKE Dataset/testImages_SPIKE"
    train(TRAIN_DATA_PATH, TEST_DATA_PATH)
