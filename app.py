import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import pandas as pd
import streamlit as st

# Fungsi untuk memuat data wajah
def load_face_data(data_path):
    images = []
    names = []
    for person in os.listdir(data_path):
        image = face_recognition.load_image_file(os.path.join(data_path, person))
        images.append(image)
        names.append(os.path.splitext(person)[0])
    return images, names

# Fungsi untuk mengenkoding wajah
def encode_faces(images):
    encode_list = []
    for img in images:
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encode_list.append(face_encodings[0])
    return encode_list

# Fungsi untuk memuat data kehadiran yang sudah tercatat
def load_attendance_data():
    df = pd.read_csv('laporan.csv')
    attendance_list = df['Name'].tolist()
    return attendance_list

# Fungsi untuk menandai kehadiran
def mark_attendance(jam_masuk, jam_pulang):
    def mark_attendance_internal(name, attendance_list, jam_masuk, jam_pulang):
        with open('laporan.csv', 'a') as f:
            now = datetime.now()
            tanggal = now.strftime('%Y-%b-%d')
            lokasi_kerja = "KSTR"

            if jam_masuk:  # Jika melakukan absensi masuk
                keterangan = "Hadir" if name in attendance_list else "Hadir"
                note = ""  # Tambahkan placeholder untuk catatan
                f.write(f'{name},{jam_masuk},{""},{tanggal},{keterangan},{lokasi_kerja},{note}\n')
            elif jam_pulang:  # Jika melakukan absensi pulang
                keterangan = "Hadir" if name in attendance_list else "Hadir"
                note = ""  # Tambahkan placeholder untuk catatan
                f.write(f'{name},{""},{jam_pulang},{tanggal},{keterangan},{lokasi_kerja},{note}\n')

    # Muat data kehadiran dari file laporan.csv
    attendance_list = load_attendance_data()

    # Muat data wajah
    data_path = "data"
    images, names = load_face_data(data_path)

    # Encode wajah
    encode_known_faces = encode_faces(images)

    # Mulai akses webcam
    cap = cv2.VideoCapture(0)
    laporan_dibuat = False
    img_placeholder = st.empty()
    absen_berhasil = False

    while True:
        success, img = cap.read()
        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        faces_cur_frame = face_recognition.face_locations(img_small_rgb)
        encode_cur_frame = face_recognition.face_encodings(img_small_rgb, faces_cur_frame)

        for encode_face, face_loc in zip(encode_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encode_known_faces, encode_face)
            face_distances = face_recognition.face_distance(encode_known_faces, encode_face)
            match_index = np.argmin(face_distances)

            if matches[match_index]:
                identity = names[match_index]
                if not absen_berhasil:  # Pastikan notifikasi hanya muncul sekali setelah absen berhasil
                    st.write("Absen Berhasil: " + identity)
                    absen_berhasil = True
            else:
                identity = "Wajah tidak dikenali"

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), 2, cv2.FILLED)
            cv2.putText(img, identity, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if not laporan_dibuat:
                mark_attendance_internal(identity, attendance_list, jam_masuk, jam_pulang)
                laporan_dibuat = True

        img_placeholder.image(img, channels="BGR", use_column_width=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Tampilkan notifikasi absen berhasil atau gagal
    if jam_masuk or jam_pulang:
        st.write("Absen Berhasil")
    else:
        st.write("Absen Gagal")

# Fungsi main
def main():
    st.title("ABSENSI")
    st.image("logokens.png", caption="The Kensington Royal Suites")
    st.write("Tekan tombol di bawah untuk memulai presensi")

    jam_masuk = datetime.now().strftime('%H:%M:%S')
    jam_pulang = datetime.now().strftime('%H:%M:%S')

    if st.button("Masuk"):
        mark_attendance(jam_masuk, None)

    elif st.button("Pulang"):
        mark_attendance(None, jam_pulang)

# Panggil fungsi main
if __name__ == "__main__":
    main()
