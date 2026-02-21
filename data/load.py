import pandas as pd 
class LoadData:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load_csv(self, file_path):
        df = pd.read_csv(file_path)
        return df