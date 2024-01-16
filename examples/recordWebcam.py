# import library OpenCV
import cv2

# membaca webcam
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('examples/video/record.avi',
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        30, size)

while True:
    # baca tiap frame dari kamera
    ret, frame = cap.read()

    if not ret: break

    result.write(frame)
    cv2.imshow('WEBCAM', frame) # menampilkan frame

    # menutup window dengan menekan tombol 'esc'
    if cv2.waitKey(1) == 27: break
cv2.VideoWriter()

cap.release() # menutup video
result.release()
cv2.destroyAllWindows() # menutup semua window