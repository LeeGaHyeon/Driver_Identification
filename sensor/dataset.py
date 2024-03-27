import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import glob
import numpy as np

class SI_dataset(Dataset):
    def __init__(self, base_path, drivers, courses, events, rounds, use_columns, train=True, window_size=5, stride=1):
        """
        Args:
            base_path (str): 데이터 파일의 기본 경로.
            drivers (list): 운전자 목록.
            courses (list): 코스 목록.
            events (list): 이벤트 목록.
            rounds (list): 라운드 목록.
            use_columns (dict): 이벤트별 사용할 컬럼 정보.
            train (bool): 학습용 데이터셋인지 검증용 데이터셋인지 여부.
            window_size (int): Sliding window의 크기.
            stride (int): Sliding window의 이동 간격.
        """
        self.window_size = window_size
        self.stride = stride
        self.data_sequences = []
        self.labels = []

        # 지정된 라운드에 따라 학습/검증 데이터셋 구분
        if train:
            train_rounds = [r for r in rounds if r != 5]
        else:
            train_rounds = [5]

        # 파일 경로 생성 및 데이터 파일 수집
        for driver in drivers:
            for event in events:
                for course in courses:
                    for round in train_rounds:
                        path_pattern = os.path.join(base_path, driver, event, course, str(round), "*.csv")
                        files = glob.glob(path_pattern)
                        for file_path in files:
                            data = pd.read_csv(file_path)

                            # 이벤트 유형에 따라 사용할 컬럼 선택
                            event_type = file_path.split(os.sep)[-4]
                            use_col = use_columns[event_type]

                            # 선택된 컬럼의 데이터만 사용
                            data = data[use_col].values.flatten()
                            # Sliding window 적용
                            for start in range(0, len(data) - self.window_size + 1, self.stride):
                                sequence = data[start:start+self.window_size]
                                self.data_sequences.append(sequence)
                                # 파일별로 라벨(운전자 인덱스) 추가
                                self.labels.append(drivers.index(driver))

    def __len__(self):
        return len(self.data_sequences)

    def __getitem__(self, idx):
        sequence = self.data_sequences[idx]
        label = self.labels[idx]

        # 데이터를 텐서로 변환
        data_tensor = torch.tensor(sequence, dtype=torch.float)
        return data_tensor, label
