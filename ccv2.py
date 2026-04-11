import cv2
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=3000, varThreshold=160, detectShadows=True)
cap = cv2.VideoCapture('')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
diag = np.sqrt(width**2 + height**2)

# Параметры для записи 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, fps, (width * 2, height))

# Для удобного отображения в демо
cv2.namedWindow('AOD System', cv2.WINDOW_NORMAL)
cv2.resizeWindow('AOD System', 1280, 480)

STATIONARY_TIME = int(3*fps)                # Сколько кадров объект должен быть неподвижен
DISTANCE_THRESHOLD = int(diag * 0.05)       # Порог смещения центра объекта
OBJECT_AREA = int((width * height)*0.0005)  # Минимальная область объекта для отслеживания
MAX_HIDDEN_TIME = int(3*fps)                # Сколько кадров объект может быть вне поля зрения
LEARN_BACK_TIME = int(fps*4)                # Период адаптации фона от начала видео

tracked_items = {}          # формат - {center: {'frames': int, 'rect': (x,y,w,h), 'is_human': bool, 'hidden_frames': int}}
frame_count = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1

    # Обучение фона
    if frame_count < LEARN_BACK_TIME:
        fg_mask = bg_subtractor.apply(frame, learningRate=-1)
        cv2.putText(frame, "Training background", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        fg_mask = bg_subtractor.apply(frame, learningRate=0.00001)
        cv2.putText(frame, "Monitoring", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    _, fg_mask = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.erode(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
    sq = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, sq)

    if frame_count >= LEARN_BACK_TIME:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        active_in_this_frame = []

        matched_old_centers = set() # Список для отслеживания тех, кого мы уже нашли в этом кадре

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < OBJECT_AREA: continue

            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w//2, y + h//2)
            active_in_this_frame.append((center, (x, y, w, h)))

            found_match = False
            for old_center in list(tracked_items.keys()):
                dist = np.linalg.norm(np.array(center) - np.array(old_center))

                if dist < DISTANCE_THRESHOLD:
                    tracked_items[old_center]['frames'] += 1
                    tracked_items[old_center]['rect'] = (x, y, w, h)
                    tracked_items[old_center]['hidden_frames'] = 0 
                    matched_old_centers.add(old_center)
                    
                    found_match = True
                    
                    # Если объект стоит достаточно долго, проверяем его на человека
                    if tracked_items[old_center]['frames'] == fps:
                        roi = frame[y:y+h, x:x+w]
                        if roi.size > 0:
                            (rects, weights) = hog.detectMultiScale(frame, winStride=(6, 6), padding=(8, 8), scale=1.2)
                            for (hx, hy, hw, hh) in rects:
                                if (x < hx + hw and x + w > hx and y < hy + hh and y + h > hy):
                                    tracked_items[old_center]['is_human'] = True
                                    # print(tracked_items[old_center])
                                    break

                    if tracked_items[old_center]['frames'] > STATIONARY_TIME and not tracked_items[old_center]['is_human']:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "ALARM", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                    break

            if not found_match:
                tracked_items[center] = {'frames': 1, 'rect': (x, y, w, h), 'is_human': False, 'hidden_frames': 0}

        for old_center in list(tracked_items.keys()):
            if old_center not in matched_old_centers:
                tracked_items[old_center]['hidden_frames'] += 1
                if tracked_items[old_center]['hidden_frames'] > MAX_HIDDEN_TIME:
                    del tracked_items[old_center]

    # Параметры для вывода записи
    mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    combined_view = np.hstack((frame, mask_bgr))
    out.write(combined_view)
    cv2.imshow('AOD System', combined_view)
    
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
