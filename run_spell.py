import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque, Counter
import serial
import time
import serial.tools.list_ports
import argparse
parser = argparse.ArgumentParser(description='spell recognition')
parser.add_argument('--arduino', action='store_true', help='Activate Arduino')
parser.add_argument('--path', type=str, help='Model Path')
args = parser.parse_args()

time.sleep(2)


def find_available_cameras(max_tests=10):
    available_cameras = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Test the first 10 indices.
available_cameras = find_available_cameras(10)
print("Available Cameras:", available_cameras)



def detect_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    arduino_ports = []

    for port in ports:
        if 'Arduino' in port.description:  # 檢查描述中是否包含 "Arduino" 字串
            arduino_ports.append(port.device)  # 獲取COM端口

    return arduino_ports



if args.arduino:
    arduino_ports = detect_arduino_port()
    if arduino_ports:
        print("找到以下Arduino的COM端口：")
        for port in arduino_ports:
            print(port)
        port = arduino_ports[0]  # 使用第一個找到的端口
        baud_rate = 9600
        ser = serial.Serial(port, baud_rate, timeout=1)
    else:
        print("找不到任何Arduino的COM端口。")
else:
    print("Arduino連接未啟用。")
#port  = 'COM1'
#port = port  # 埠號（Windows作業系統通常為COMX，X為埠號；Linux作業系統通常為/dev/ttyUSBX，X為埠號）

#baud_rate = 9600  # （Arduino程式中的Serial.begin的數值）

# 建立串列連接
#ser = serial.Serial(port, baud_rate, timeout=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

def crop_center(img, desired_width, desired_height):
    h, w = img.shape[:2]  # 獲取圖像的高度和寬度
    start_x = w//2 - desired_width//2  # 計算裁切起始點的X坐標
    start_y = h//2 - desired_height//2  # 計算裁切起始點的Y坐標
    return img[start_y:start_y+desired_height, start_x:start_x+desired_width]  # 返回裁切後的圖像


def check_movement(coordinates):
    last_coords = list(coordinates)[-10:] if len(coordinates) >= 10 else list(coordinates)  # 獲取最近的10個坐標點
    center_x = np.mean([coord[0] for coord in last_coords])  # 計算這些點的X坐標平均值
    center_y = np.mean([coord[1] for coord in last_coords])  # 計算這些點的Y坐標平均值
    distances = [np.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in last_coords]  # 計算每個點到平均位置的距離
    var = np.var(distances)  # 計算距離的變異數

    if np.isnan(var):
        var = 1000  # 如果變異數為NaN，則設置為1000

    return var  # 返回變異數


def get_coord(frame, cord_to_remove=[]):
    coord_list = []  # 初始化坐標列表

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 將圖像轉換為灰階
    _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # 應用二值化閾值
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 檢測輪廓

    for contour in contours:
        if cv2.contourArea(contour) < 2:  # 如果輪廓面積太小，則忽略
            continue
        x, y, w, h = cv2.boundingRect(contour)  # 計算輪廓的邊界框
        coord = (x + w // 2, y + h // 2)  # 計算邊界框的中心點
        if coord not in cord_to_remove:  # 如果這個坐標不在移除列表中
            coord_list.append(coord)  # 則添加到坐標列表

    return coord_list  # 返回坐標列表


def get_predict(black_img):
    rblack_img = cv2.resize(black_img, (28, 28))
    rblack_img = cv2.cvtColor(rblack_img, cv2.COLOR_BGR2GRAY) / 255.0
    rblack_img = rblack_img.reshape(1, 1, 28, 28)
    img_tensor = torch.FloatTensor(rblack_img).to(device)
    output = model(img_tensor)
    prediction = torch.argmax(output).item()
    softmax_output = torch.nn.functional.softmax(output, dim=1)
    prob = torch.max(softmax_output).item()

    return prediction, prob


cap = cv2.VideoCapture(available_cameras[0])
fps = 30

desired_width = 640
desired_height = 480
#(1280, 720)
desired_exposure = -11
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
cap.set(cv2.CAP_PROP_EXPOSURE, desired_exposure)

# get 50 frames for init check bright spots

coord_to_check = []

for ii in range(50):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = crop_center(frame, 350, 350)
    coord_to_check += get_coord(frame, cord_to_remove=[])

coordinate_counter = Counter()
for kk in coord_to_check:
    coordinate_counter[kk] += 1

cord_to_remove = []
print('Top 10 most frequent coordinates in the first 100 frames:')
for coord, count in coordinate_counter.most_common(10):
    print(f'Coordinate: {coord}, Frequency: {count}')
    cord_to_remove.append(coord)

black_img = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8) + 255


coordinates = deque(maxlen=50)

#coordinate_counter_after_100 = Counter()
check_counter = 0
cord_to_remove = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args.path, map_location=device)
model.eval()

ff = 0
no_cord = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = crop_center(frame, 350, 350)
    cord_list = get_coord(frame, cord_to_remove=[])
    if len(cord_list) > 0:
        no_cord = 0
        coordinates.append(cord_list[0])
        black_img = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8) + 255

        for i in range(1, len(coordinates)):
            cv2.line(black_img, coordinates[i-1],
                     coordinates[i], (0, 0, 0), 6)
                     
    else:
        no_cord += 1
        #print(no_cord)
    kernel = np.ones((8, 8), np.uint8)

    # Apply dilation
    frame = cv2.dilate(frame, kernel, iterations=1)

    concatenated_image = np.hstack((frame, black_img))
    cv2.imshow('Frame', concatenated_image)
    
    #cv2.imwrite(
    #    f'C:\\Users\\tyhua\\Desktop\\openai\\jpg\\saved_gogo_{ff}.jpg', concatenated_image)
    
    
    check_counter += 1
    if check_counter > 10: #重設過後等待10次才再檢查，以免圖形一直被消除掉
        move = check_movement(coordinates)
        if move < 10 or no_cord >8:
            #print(move, no_cord)
            if len(list(coordinates)) >= 15:
                ff += 1
                #cv2.imwrite(f'C:\\Gdrive_tudo\\code\\project\\moretraining\morev5_{ff:05d}.png', black_img)  
                #print(ff)
                prediction, prob = get_predict(black_img)
                if prob > 0.99:
                    print('Prediction:', prediction, prob)
                    if prediction == 1:
                        spell = 'Incedio🔥'
                        command = f"incendio"
                        #ser.write(command.encode())
                        #print("incendio")
                    elif prediction == 2:
                        spell = 'aqua🌊'
                        command = f"aqua"
                        #ser.write(command.encode())
                        #print("aqua")
                    elif prediction == 3:
                        spell = 'leviosa🪶'
                        command = f'arresto'
                        #ser.write(command.encode())
                        #print("arresto")
                    elif prediction == 4:
                        spell = 'arresto🖐️'
                        command=f"arresto"
                        #ser.write(command.encode())
                        #print("arresto")
                    elif prediction == 5:
                        spell = 'alohomora🔓'
                        command = f"alohomora"
                        #ser.write(command.encode())
                        #print("alohomora")
                    elif prediction == 6:
                        spell = 'lumos💡' 
                        command = f"lumos"
                        #ser.write(command.encode())
                        #print("lumos")
                    #ser.write(command.encode())
 
                    if args.arduino:
                        ser.write(command.encode())
                    print(spell)
                        #print(command)
                else:
                    print("NULL")
            coordinates = deque(maxlen=50)
            check_counter = 0
            no_cord = 0

            black_img = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8) + 255
            concatenated_image = np.hstack((frame, black_img))
            cv2.imshow('Frame', concatenated_image)

    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
