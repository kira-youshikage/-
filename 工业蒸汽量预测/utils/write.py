import txt
from config import DefaultConfig

opt = DefaultConfig()

def write_txt(test_label):
	txt_path = opt.txt_path
	txt.write_data(txt_path, test_label)