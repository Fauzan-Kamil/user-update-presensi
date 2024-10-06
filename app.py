import cv2
import face_recognition
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from datetime import datetime
import pandas as pd
import streamlit as st

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="User Absensi")

# Fungsi untuk memuat data wajah
def load_face_data(data_path):
    images = []
    names = []
    for person in os.listdir(data_path):
        image_path = os.path.join(data_path, person)
        if os.path.isfile(image_path):  # Pastikan hanya file yang diproses
            image = face_recognition.load_image_file(image_path)
            images.append(image)
            names.append(os.path.splitext(person)[0])
    return images, names

# Fungsi untuk mengenkoding wajah
def encode_faces(images):
    encode_list = []
    for img in images:
        face_encodings = face_recognition.face_encodings(img)
        if face_encodings:
            encode_list.append(face_encodings[0])
    return encode_list

# Fungsi untuk memuat data kehadiran yang sudah tercatat
def load_attendance_data():
    try:
        df = pd.read_csv('laporan.csv')
        return df['Name'].tolist()
    except FileNotFoundError:
        return []

# Fungsi untuk menandai kehadiran
def mark_attendance(jam_masuk, jam_pulang):
    def mark_attendance_internal(name, attendance_list, jam_masuk, jam_pulang):
        now = datetime.now()
        tanggal = now.strftime('%Y-%b-%d')
        lokasi_kerja = "KSTR"
        keterangan = "Hadir"
        note = ""

        new_entry = {
            'Name': name,
            'Jam Masuk': jam_masuk if jam_masuk else "",
            'Jam Pulang': jam_pulang if jam_pulang else "",
            'Tanggal': tanggal,
            'Keterangan': keterangan,
            'Lokasi Kerja': lokasi_kerja,
            'Note': note
        }

        # Baca file CSV jika ada, jika tidak buat DataFrame baru
        try:
            df = pd.read_csv('laporan.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['Name', 'Jam Masuk', 'Jam Pulang', 'Tanggal', 'Keterangan', 'Lokasi Kerja', 'Note'])

        # Tambahkan entri baru
        df = df.append(new_entry, ignore_index=True)

        # Simpan kembali ke CSV
        df.to_csv('laporan.csv', index=False)

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
        
        if not success:
            st.error("Gagal membaca frame dari webcam. Pastikan webcam terhubung dan berfungsi dengan baik.")
            break

        img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        faces_cur_frame = face_recognition.face_locations(img_small_rgb)
        encode_cur_frame = face_recognition.face_encodings(img_small_rgb, faces_cur_frame)

        for encode_face, face_loc in zip(encode_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encode_known_faces, encode_face)
            face_distances = face_recognition.face_distance(encode_known_faces, encode_face)
            
            if len(face_distances) > 0:
                match_index = np.argmin(face_distances)
                if matches[match_index]:
                    identity = names[match_index]
                    if not absen_berhasil:
                        st.write("Absen Berhasil: " + identity)
                        absen_berhasil = True
                else:
                    identity = "Wajah tidak dikenali"
            else:
                identity = "Wajah tidak dikenali"

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, identity, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if not laporan_dibuat:
                mark_attendance_internal(identity, attendance_list, jam_masuk, jam_pulang)
                laporan_dibuat = True

        img_placeholder.image(img, channels="BGR", use_column_width=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if jam_masuk or jam_pulang:
        st.success("Absen Berhasil")
    else:
        st.error("Absen Gagal")

def main():
    st.title("ABSENSI")
    st.image("logokens.png", caption="The Kensington Royal Suites")
    st.write("Tekan tombol di bawah untuk memulai presensi")

    jam_masuk = datetime.now().strftime('%H:%M:%S')
    jam_pulang = datetime.now().strftime('%H:%M:%S')
    
    # Define custom CSS for button colors
    st.markdown("""
        <style>
            #button1 {
                color: white;
                background-color: #00FF00;
                border-color: #00FF00;
            }
            #button2 {
                color: white;
                background-color: #FF0000;
                border-color: #FF0000;
            }
        </style>
    """, unsafe_allow_html=True)

    # Create buttons with st.button
    col1, col2 = st.columns(2)
    with col1:
        button1_clicked = st.button("Masuk", key="button1", help="Klik untuk absen masuk")
    with col2:
        button2_clicked = st.button("Pulang", key="button2", help="Klik untuk absen pulang")

    if button1_clicked:
        mark_attendance(jam_masuk, None)
        st.success("Anda telah berhasil melakukan absensi masuk.")
    elif button2_clicked:
        mark_attendance(None, jam_pulang)
        st.success("Anda telah berhasil melakukan absensi pulang.")

# Panggil fungsi main
if __name__ == "__main__":
    main()
