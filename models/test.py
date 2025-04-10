#from ..utils.data_loading import load_data
import sys
sys.path.append('FermBench\\utils')

from data_loading import load_data
data_path = "../data/dataset_curated.xlsx"
load_data(data_path)

