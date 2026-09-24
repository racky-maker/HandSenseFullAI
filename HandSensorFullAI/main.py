import cv2
import mediapipe as mp
import math
import pyautogui
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# FUNGSI MATEMATIKA
# =========================================================

def sudut(a, b, c):
    """
    Menghitung sudut ABC dari tiga landmark.
    """

    ba = (a.x - b.x, a.y - b.y)
    bc = (c.x - b.x, c.y - b.y)

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    panjang_ba = math.sqrt(
        ba[0] ** 2 + ba[1] ** 2
    )

    panjang_bc = math.sqrt(
        bc[0] ** 2 + bc[1] ** 2
    )

    if panjang_ba == 0 or panjang_bc == 0:
        return 0

    nilai = dot / (panjang_ba * panjang_bc)

    nilai = max(-1, min(1, nilai))

    return math.degrees(
        math.acos(nilai)
    )


def jarak(a, b):
    """
    Menghitung jarak antara dua landmark.
    """

    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


# =========================================================
# HITUNG JUMLAH JARI
# =========================================================

def hitung_jari(hand):

    jumlah = 0

    # -------------------------
    # Jempol
    # -------------------------

    sudut_jempol = sudut(
        hand[2],
        hand[3],
        hand[4]
    )

    if sudut_jempol > 150:
        jumlah += 1

    # -------------------------
    # Telunjuk
    # -------------------------

    if sudut(
        hand[5],
        hand[6],
        hand[8]
    ) > 160:
        jumlah += 1

    # -------------------------
    # Jari tengah
    # -------------------------

    if sudut(
        hand[9],
        hand[10],
        hand[12]
    ) > 160:
        jumlah += 1

    # -------------------------
    # Jari manis
    # -------------------------

    if sudut(
        hand[13],
        hand[14],
        hand[16]
    ) > 160:
        jumlah += 1

    # -------------------------
    # Kelingking
    # -------------------------

    if sudut(
        hand[17],
        hand[18],
        hand[20]
    ) > 160:
        jumlah += 1

    return jumlah


# =========================================================
# NAMA GESTURE BERDASARKAN JUMLAH JARI
# =========================================================

def nama_gesture(jumlah_jari):

    gesture = {
        0: "FIST",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "OPEN HAND"
    }

    return gesture.get(
        jumlah_jari,
        "UNKNOWN"
    )


# =========================================================
# DETEKSI THUMBS UP
# =========================================================

def is_thumbs_up(hand):

    index_closed = hand[8].y > hand[6].y
    middle_closed = hand[12].y > hand[10].y
    ring_closed = hand[16].y > hand[14].y
    pinky_closed = hand[20].y > hand[18].y

    thumb_angle = sudut(
        hand[2],
        hand[3],
        hand[4]
    )

    thumb_open = thumb_angle > 150

    return (
        thumb_open
        and index_closed
        and middle_closed
        and ring_closed
        and pinky_closed
    )


# =========================================================
# DETEKSI PEACE
# =========================================================

def is_peace(hand):

    index_open = hand[8].y < hand[6].y
    middle_open = hand[12].y < hand[10].y

    ring_closed = hand[16].y > hand[14].y
    pinky_closed = hand[20].y > hand[18].y

    return (
        index_open
        and middle_open
        and ring_closed
        and pinky_closed
    )


# =========================================================
# DETEKSI OK
# =========================================================

def is_ok(hand):

    thumb_index_close = (
        jarak(hand[4], hand[8]) < 0.08
    )

    middle_open = hand[12].y < hand[10].y
    ring_open = hand[16].y < hand[14].y
    pinky_open = hand[20].y < hand[18].y

    return (
        thumb_index_close
        and middle_open
        and ring_open
        and pinky_open
    )


# =========================================================
# DETEKSI GESTURE
# =========================================================

def deteksi_gesture(hand):

    jumlah_jari = hitung_jari(hand)

    # Gesture khusus diperiksa terlebih dahulu

    if is_ok(hand):
        gesture = "OK"

    elif is_thumbs_up(hand):
        gesture = "THUMBS UP"

    elif is_peace(hand):
        gesture = "PEACE"

    else:
        gesture = nama_gesture(jumlah_jari)

    return jumlah_jari, gesture

# =========================================================
# AKSI GESTURE
# =========================================================

def aksi_gesture(gesture):

    aksi = {
        "FIST": "STOP",
        "ONE": "NEXT",
        "TWO": "PLAY",
        "THREE": "MODE 3",
        "FOUR": "MODE 4",
        "OPEN HAND": "MENU",
        "THUMBS UP": "OK",
        "PEACE": "PAUSE",
        "OK": "SELECT"
    }

    return aksi.get(
        gesture,
        "NO ACTION"
    )

# =========================================================
# KONTROL MEDIA
# =========================================================

def jalankan_aksi(gesture):

    if gesture == "THUMBS UP":
        pyautogui.press("playpause")

    elif gesture == "PEACE":
        pyautogui.press("playpause")

    elif gesture == "ONE":
        pyautogui.press("nexttrack")

    elif gesture == "FIST":
        pyautogui.press("volumemute")

    elif gesture == "OPEN HAND":
        pyautogui.press("playpause")

# =========================================================
# KONTROL MOUSE
# =========================================================

screen_w, screen_h = pyautogui.size()

last_mouse_x = -1
last_mouse_y = -1

def gerakkan_mouse(hand):
    global last_mouse_x, last_mouse_y

    x = hand[8].x
    y = hand[8].y

    margin = 0.1

    x = (x - margin) / (1 - 2 * margin)
    y = (y - margin) / (1 - 2 * margin)

    x = max(0, min(1, x))
    y = max(0, min(1, y))

    mouse_x = int(x * screen_w)
    mouse_y = int(y * screen_h)

    # Jangan menggerakkan mouse kalau perubahan terlalu kecil
    if abs(mouse_x - last_mouse_x) >= 3 or abs(mouse_y - last_mouse_y) >= 3:
        pyautogui.moveTo(mouse_x, mouse_y, duration=0)

        last_mouse_x = mouse_x
        last_mouse_y = mouse_y

# =========================================================
# MEDIA PIPE
# =========================================================

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# =========================================================
# KAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

timestamp = 0

# =========================================================
# COOLDOWN GESTURE
# =========================================================

gesture_sebelumnya = ""
gesture_terakhir_dijalankan = ""

waktu_gesture = 0

stabilitas = 15

# =========================================================
# LOOP UTAMA
# =========================================================

fps = 0
frame_count = 0
fps_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        print("Kamera tidak dapat dibuka!")
        break

    # FPS
    frame_count += 1

    if time.time() - fps_time >= 1:
        fps = frame_count
        frame_count = 0
        fps_time = time.time()

    # Membuat kamera seperti cermin
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 1

    # Deteksi tangan
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    # Ukuran frame
    h, w, _ = frame.shape


    # =========================================================
    # DATA TANGAN
    # =========================================================

    tangan_data = []

    if not result.hand_landmarks:

        gesture_sebelumnya = ""
        gesture_terakhir_dijalankan = ""
        waktu_gesture = timestamp   

    if result.hand_landmarks:

        for i, hand in enumerate(result.hand_landmarks):

            # =================================================
            # LANDMARK
            # =================================================

            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # CONNECTION
            # =================================================

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9),
                (9, 13),
                (13, 17)
            ]


            for start, end in connections:

                x1 = int(hand[start].x * w)
                y1 = int(hand[start].y * h)

                x2 = int(hand[end].x * w)
                y2 = int(hand[end].y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


            # =================================================
            # DETEKSI
            # =================================================

            handedness = result.handedness[i][0].category_name

            jumlah_jari, gesture = deteksi_gesture(hand)

            if gesture == "OPEN HAND" and handedness == "Right":
                gerakkan_mouse(hand)

            aksi = aksi_gesture(gesture)

            # =========================================================
            # JALANKAN ACTION
            # =========================================================

            if gesture != gesture_sebelumnya:

                gesture_sebelumnya = gesture
                waktu_gesture = timestamp

            else:

                if (
                    timestamp - waktu_gesture >= stabilitas
                    and gesture != gesture_terakhir_dijalankan
                ):

                    jalankan_aksi(gesture)

                    gesture_terakhir_dijalankan = gesture

            # =================================================
            # LEFT / RIGHT
            # =================================================

            side = result.handedness[i][0].category_name


            tangan_data.append(
                {
                    "side": side,
                    "fingers": jumlah_jari,
                    "gesture": gesture,
                    "action": aksi
                }
            )


    # =========================================================
    # PANEL
    # =========================================================

    jumlah_tangan = len(tangan_data)

    panel_height = 150 + (jumlah_tangan * 70)


    cv2.rectangle(
        frame,
        (15, 15),
        (600, panel_height),
        (30, 30, 30),
        -1
    )


    # =========================================================
    # JUDUL
    # =========================================================

    cv2.putText(
        frame,
        "HAND SENSOR AI",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # =========================================================
    # STATUS
    # =========================================================

    cv2.putText(
        frame,
        f"Total Hands: {jumlah_tangan}",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if jumlah_tangan > 0 else (0, 255, 255),
        2
    )


    # =========================================================
    # INFORMASI TANGAN
    # =========================================================

    for i, data in enumerate(tangan_data):

        teks = (
            f"{data['side']}: "
            f"{data['fingers']} jari - "
            f"{data['gesture']}"
        )

        aksi_teks = f"Action: {data['action']}"

        cv2.putText(
            frame,
            teks,
            (35, 125 + (i * 70)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            aksi_teks,
            (35, 150 + (i * 70)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
            

    # =====================================================
    # TAMPILKAN KAMERA
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Hand Sensor",
        frame
    )


    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# TUTUP PROGRAM
# =========================================================

cap.release()

detector.close()

cv2.destroyAllWindows()