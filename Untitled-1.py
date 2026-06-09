import cv2
import serial
import time

# -----------------------------
# Arduino Serial Connection
# -----------------------------

ARDUINO_PORT = "COM5"
BAUD_RATE = 9600

arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)

time.sleep(2)

# -----------------------------
# Face Detection
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Load Owner Face
# -----------------------------

reference_frame = cv2.imread("owner.jpg")

if reference_frame is None:
    print("owner.jpg not found")
    exit()

# -----------------------------
# Convert face to gray
# -----------------------------

def extract_face(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (100, 100))

    return face

# -----------------------------
# Prepare reference face
# -----------------------------

reference_face = extract_face(reference_frame)

if reference_face is None:
    print("No face in owner.jpg")
    exit()

# -----------------------------
# Face Compare Function
# -----------------------------

def compare_faces(face1, face2):

    result = cv2.matchTemplate(
        face1,
        face2,
        cv2.TM_CCOEFF_NORMED
    )

    score = result[0][0]

    return score

# -----------------------------
# Camera Start
# -----------------------------

camera = cv2.VideoCapture(0)

print("Python system ready")

# -----------------------------
# Main Loop
# -----------------------------

while True:

    if arduino.in_waiting > 0:

        message = arduino.readline().decode().strip()

        print("Arduino:", message)

        # Arduino sends SCAN
        if message == "SCAN":

            print("Checking face...")

            success, frame = camera.read()

            if not success:
                print("Camera error")
                arduino.write(b"FACE_FAIL\n")
                continue

            detected_face = extract_face(frame)

            if detected_face is None:

                print("No face detected")

                arduino.write(b"FACE_FAIL\n")

                continue

            score = compare_faces(
                reference_face,
                detected_face
            )

            percentage = score * 100

            print("Match:", round(percentage, 2), "%")

            # 80% match required
            if percentage >= 80:

                print("FACE_OK sent")

                arduino.write(b"FACE_OK\n")

            else:

                print("FACE_FAIL sent")

                arduino.write(b"FACE_FAIL\n")

    # Press Q to exit
    if cv2.waitKey(1) == ord('q'):
        break

# -----------------------------
# Close Everything
# -----------------------------

camera.release()

cv2.destroyAllWindows()

arduino.close()