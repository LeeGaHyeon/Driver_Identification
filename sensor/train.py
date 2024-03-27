
from dataset import SI_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, feature_size, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.feature_extractor = nn.Linear(hidden_size, feature_size)  # 1280 특성 추출
        self.classifier = nn.Linear(feature_size, num_classes)  # 최종 분류 레이어

    def forward(self, x):
        # 초기 상태 설정
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 전달
        out, _ = self.lstm(x, (h0, c0))

        # LSTM의 마지막 시간 단계의 출력을 특성 추출 레이어로 전달
        features = self.feature_extractor(out[:, -1, :])

        # 특성을 기반으로 최종 분류
        output = self.classifier(features)
        return output, features  # 분류 결과와 1280 특성 모두 반환


input_size = 1  # 'y' 또는 'z' 축 값의 차원
hidden_size = 128
num_layers = 5
feature_size = 1280
num_classes = 3

model = LSTM(input_size, hidden_size, num_layers, feature_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


# 수정된 학습 함수
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}, Training"):
            sequences = sequences.unsqueeze(-1)
            outputs, features = model(sequences)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for sequences, labels in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs}, Validation"):
                sequences = sequences.unsqueeze(-1)
                outputs, features = model(sequences)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                valid_acc += calculate_accuracy(outputs, labels)

            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

            save_model(model, f'./model/MI_{epoch}epoch_model.pt')  # .pt로 저장
            save_model(model, f'./model/MI_{epoch}epoch_model.pth')  # .pth로 저장

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')


num_epochs = 30
base_path = "C:/Users/user/jupyterlab/data_preprocessing/split_data/sensor"
drivers = ['leeseunglee', 'leegihun', 'leeyunguel']
courses = ['A', 'B', 'C']
events = ['bump', 'corner']
rounds = [1, 2, 3, 4, 5, 6]
use_columns = {'bump': 'y_change', 'corner': 'z_change'}

# 데이터셋 인스턴스 생성 (학습용 및 검증용)
train_dataset = SI_dataset(base_path, drivers, courses, events, rounds, use_columns, train=True)
valid_dataset = SI_dataset(base_path, drivers, courses, events, rounds, use_columns, train=False)

# 데이터 로더 설정 (학습용 및 검증용)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 모델 학습 시작
train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs)

