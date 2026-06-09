import cv2

THRESHOLD = 76.5  # 70% match

# Load face detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the authorized face photo
ref_img = cv2.imread("WhatsApp Image 2026-05-11 at 1.42.48 AM.jpeg", cv2.IMREAD_GRAYSCALE)
if ref_img is None:
    print("ERROR: Put a file called 'authorized_face.jpg' in the same folder.")
    exit()

ref_faces = detector.detectMultiScale(ref_img, 1.1, 5, minSize=(60, 60))
if len(ref_faces) == 0:
    print("ERROR: No face found in authorized_face.jpg")
    exit()

x, y, w, h = ref_faces[0]
ref_face = cv2.equalizeHist(cv2.resize(ref_img[y:y+h, x:x+w], (100, 100)))
print("Reference face loaded. Opening camera...")

# Open camera and compare
cam = cv2.VideoCapture(0)

while True:
    ok, frame = cam.read()
    if not ok:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        cam_face = cv2.equalizeHist(cv2.resize(gray[y:y+h, x:x+w], (100, 100)))
        diff = cv2.absdiff(ref_face, cam_face).mean()
        match_percent = max(0, 100 - (diff / 255 * 100))

        if match_percent >= 70:
            color = (0, 255, 0)  # Green
            label = f"DOOR OPEN ({match_percent:.0f}%)"
        else:
            color = (0, 0, 255)  # Red
            label = f"DOOR LOCKED ({match_percent:.0f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Test - Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
