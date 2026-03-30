import cv2
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=3000, varThreshold=180, detectShadows=True)
STATIONARY_TIME = 120   # Сколько кадров объект должен быть неподвижен
DISTANCE_THRESHOLD = 20 # Порог смещения центра объекта
OBJECT_AREA = 20        # Минимальная область объекта для отслеживания
MAX_HIDDEN_TIME = 90    # Сколько кадров объект может быть вне поля зрения

tracked_items = {}      #формат - {center: {'frames': int, 'rect': (x,y,w,h), 'is_human': bool, 'hidden_frames': int}}

cap = cv2.VideoCapture('sourse_video\streetcam2.mp4')


# Параметры для записи 
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('recording_demo.avi', fourcc, fps, (width * 2, height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    fg_mask = bg_subtractor.apply(frame, learningRate=0.00001)
    _, fg_mask = cv2.threshold(fg_mask, 220, 255, cv2.THRESH_BINARY) # Убираем тени 
    sq = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))     #склеиваем объект 
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, sq)

    cv2.imshow('MOG2 Mask', fg_mask)
    # if cv2.waitKey(10) & 0xFF == ord('q'): break

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    active_in_this_frame = []
    
    matched_old_centers = set() #Список для отслеживания тех, кого мы уже нашли в этом кадре

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
                
                # Если объект стоит достаточно долго, проверяем его на человека (на 20 кадре)
                if tracked_items[old_center]['frames'] == 20:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        (rects, weights) = hog.detectMultiScale(frame, winStride=(6, 6), padding=(8, 8), scale=1.2)
                        for (hx, hy, hw, hh) in rects:
                            if (x < hx + hw and x + w > hx and y < hy + hh and y + h > hy):
                                tracked_items[old_center]['is_human'] = True
                                break

                if tracked_items[old_center]['frames'] > STATIONARY_TIME and not tracked_items[old_center]['is_human']:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "ALARM", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break
        
        if not found_match:
            tracked_items[center] = {'frames': 1, 'rect': (x, y, w, h), 'is_human': False, 'hidden_frames': 0}

    #Логика обработки "пропавших" объектов
    for old_center in list(tracked_items.keys()):
        if old_center not in matched_old_centers:
            tracked_items[old_center]['hidden_frames'] += 1
            if tracked_items[old_center]['hidden_frames'] > MAX_HIDDEN_TIME:
                del tracked_items[old_center]

    # Очистка старых данных (если объект исчез из маски)
    # Очистка грубая, в реальной задаче стоило бы удалять объекты по возрастанию 
    # количесвта кадров, в которых он найден подряд
    if len(tracked_items) > 50:
        tracked_items.clear()


    #Параметры для вывода записи
    # mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    # combined_view = np.hstack((frame, mask_bgr))

    # out.write(combined_view)
    # cv2.imshow('Demo View', combined_view)
    # if cv2.waitKey(10) & 0xFF == ord('q'): break


    cv2.imshow('AOD Module', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
# out.release()
cv2.destroyAllWindows()
