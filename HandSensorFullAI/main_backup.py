import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# Fungsi menghitung jari
# =========================




def sudut(a, b, c):
    """
    Menghitung sudut ABC
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

    return math.degrees(math.acos(nilai))


def hitung_jari(hand, handedness):

    jumlah = 0

    # =========================
    # JEMPOL
    # =========================

    sudut_jempol = sudut(
        hand[2],
        hand[3],
        hand[4]
    )

    if sudut_jempol > 150:
        jumlah += 1


    # =========================
    # TELUNJUK
    # =========================

    sudut_telunjuk = sudut(
        hand[5],
        hand[6],
        hand[8]
    )

    if sudut_telunjuk > 160:
        jumlah += 1


    # =========================
    # JARI TENGAH
    # =========================

    sudut_tengah = sudut(
        hand[9],
        hand[10],
        hand[12]
    )

    if sudut_tengah > 160:
        jumlah += 1


    # =========================
    # JARI MANIS
    # =========================

    sudut_manis = sudut(
        hand[13],
        hand[14],
        hand[16]
    )

    if sudut_manis > 160:
        jumlah += 1


    # =========================
    # KELINGKING
    # =========================

    sudut_kelingking = sudut(
        hand[17],
        hand[18],
        hand[20]
    )

    if sudut_kelingking > 160:
        jumlah += 1


    return jumlah

def nama_gesture(jumlah_jari):
    if jumlah_jari == 0:
        return "FIST"
    elif jumlah_jari == 1:
        return "ONE"
    elif jumlah_jari == 2:
        return "TWO"
    elif jumlah_jari == 3:
        return "THREE"
    elif jumlah_jari == 4:
        return "FOUR"
    elif jumlah_jari == 5:
        return "OPEN HAND"
    else:
        return "UNKNOWN"

def is_thumbs_up(hand):
    # Jari lain harus tertutup
    index_closed = hand[8].y > hand[6].y
    middle_closed = hand[12].y > hand[10].y
    ring_closed = hand[16].y > hand[14].y
    pinky_closed = hand[20].y > hand[18].y

    # Jempol harus terbuka
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

def jarak(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def is_ok(hand):
    # Jempol dan telunjuk harus berdekatan
    thumb_index_close = jarak(hand[4], hand[8]) < 0.08

    # Tiga jari lainnya terbuka
    middle_open = hand[12].y < hand[10].y
    ring_open = hand[16].y < hand[14].y
    pinky_open = hand[20].y < hand[18].y

    return (
        thumb_index_close
        and middle_open
        and ring_open
        and pinky_open
    )

def is_three(hand):
    index_open = hand[8].y < hand[6].y
    middle_open = hand[12].y < hand[10].y
    ring_open = hand[16].y < hand[14].y

    pinky_closed = hand[20].y > hand[18].y
    thumb_closed = hand[4].y > hand[3].y

    return (
        index_open
        and middle_open
        and ring_open
        and pinky_closed
        and thumb_closed
    )

# =========================
# MediaPipe
# =========================

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# =========================
# Kamera
# =========================

cap = cv2.VideoCapture(0)

timestamp = 0


while True:

    success, frame = cap.read()

    if not success:
        print("Kamera tidak dapat dibuka!")
        break

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


    # =========================
    # Jika tangan terdeteksi
    # =========================

    if result.hand_landmarks:

        for i, hand in enumerate(result.hand_landmarks):

            h, w, _ = frame.shape


            # -------------------------
            # Gambar titik
            # -------------------------

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


            # -------------------------
            # Garis tangan
            # -------------------------

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


            # -------------------------
            # Menentukan tangan kanan/kiri
            # -------------------------

            handedness = "Right"

            if result.handedness:

                handedness = result.handedness[i][0].category_name


            # -------------------------
            # Hitung jari
            # -------------------------

            jumlah_jari = hitung_jari(
                hand,
                handedness
            )

            if is_ok(hand):
                gesture = "OK"

            elif is_thumbs_up(hand):
                gesture = "THUMBS UP"

            elif is_peace(hand):
                gesture = "PEACE"

            elif is_three(hand):
                gesture = "THREE"

            else:
                gesture = nama_gesture(jumlah_jari)


            # -------------------------
            # Tampilkan jumlah jari
            # -------------------------

            cv2.putText(
                frame,
                f"Jari: {jumlah_jari}",
                (30, 60 + i * 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (30, 105 + i * 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )


    # =========================
    # Tampilkan kamera
    # =========================

    cv2.imshow(
        "Hand Sensor",
        frame
    )


    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# Tutup
# =========================

cap.release()
detector.close()
cv2.destroyAllWindows()