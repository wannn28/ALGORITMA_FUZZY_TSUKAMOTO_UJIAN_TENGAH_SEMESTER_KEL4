import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Daftar aturan fuzzy
rules = [
    {'antecedent': ("Rendah", "Rendah", "Rendah"), 'consequent': "Sedang"},
    {'antecedent': ("Rendah", "Rendah", "Sedang"), 'consequent': "Tinggi"},
    {'antecedent': ("Rendah", "Rendah", "Tinggi"), 'consequent': "Tinggi"},
    {'antecedent': ("Rendah", "Sedang", "Rendah"), 'consequent': "Sedang"},
    {'antecedent': ("Rendah", "Sedang", "Sedang"), 'consequent': "Sedang"},
    {'antecedent': ("Rendah", "Sedang", "Tinggi"), 'consequent': "Tinggi"},
    {'antecedent': ("Rendah", "Tinggi", "Rendah"), 'consequent': "Rendah"},
    {'antecedent': ("Rendah", "Tinggi", "Sedang"), 'consequent': "Sedang"},
    {'antecedent': ("Rendah", "Tinggi", "Tinggi"), 'consequent': "Tinggi"},
    {'antecedent': ("Sedang", "Rendah", "Rendah"), 'consequent': "Sedang"},
    {'antecedent': ("Sedang", "Rendah", "Sedang"), 'consequent': "Tinggi"},
    {'antecedent': ("Sedang", "Rendah", "Tinggi"), 'consequent': "Tinggi"},
    {'antecedent': ("Sedang", "Sedang", "Rendah"), 'consequent': "Sedang"},
    {'antecedent': ("Sedang", "Sedang", "Sedang"), 'consequent': "Tinggi"},
    {'antecedent': ("Sedang", "Sedang", "Tinggi"), 'consequent': "Tinggi"},
    {'antecedent': ("Sedang", "Tinggi", "Rendah"), 'consequent': "Sedang"},
    {'antecedent': ("Sedang", "Tinggi", "Sedang"), 'consequent': "Tinggi"},
    {'antecedent': ("Sedang", "Tinggi", "Tinggi"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Rendah", "Rendah"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Rendah", "Sedang"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Rendah", "Tinggi"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Sedang", "Rendah"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Sedang", "Sedang"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Sedang", "Tinggi"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Tinggi", "Rendah"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Tinggi", "Sedang"), 'consequent': "Tinggi"},
    {'antecedent': ("Tinggi", "Tinggi", "Tinggi"), 'consequent': "Tinggi"},
]

# Fungsi keanggotaan segitiga
def triangle_membership(x, points):
    a, b, c = points
    if x <= a:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    else:
        return 0.0

# Level keanggotaan untuk setiap variabel
request_count_levels = {
    "Rendah": [0, 0, 500],
    "Sedang": [0, 500, 1000],
    "Tinggi": [500, 1000, 1000],
}

system_security_level_levels = {
    "Rendah": [0, 0, 5],
    "Sedang": [0, 5, 10],
    "Tinggi": [5, 10, 10],
}

anomalous_data_volume_levels = {
    "Rendah": [0, 0, 250],
    "Sedang": [0, 250, 500],
    "Tinggi": [250, 500, 500],
}

cyber_attack_risk_level_levels = {
    "Rendah": [0, 0, 50],
    "Sedang": [0, 50, 100],
    "Tinggi": [50, 100, 100],
}

# Fungsi untuk mendapatkan derajat keanggotaan input
def get_input_membership(levels, x):
    memberships = {}
    for level, points in levels.items():
        memberships[level] = triangle_membership(x, points)
    return memberships

# Fungsi untuk menghitung nilai z berdasarkan α pada fungsi segitiga yang monoton
def get_z_value(level, alpha):
    a, b, c = cyber_attack_risk_level_levels[level]
    if alpha == 0:
        return 0
    if b == a:
        # Monoton naik dari a ke c
        z = alpha * (c - a) + a
    elif b == c:
        # Monoton turun dari a ke c
        z = c - alpha * (c - a)
    else:
        if alpha <= triangle_membership(b, (a, b, c)):
            # Bagian naik
            z = alpha * (b - a) + a
        else:
            # Bagian turun
            z = c - alpha * (c - b)
    return z

# Fungsi untuk menggambar grafik fungsi keanggotaan
def plot_membership(levels, var_name, input_value=None, show_input=False):
    x_min = min([p[0] for p in levels.values()])
    x_max = max([p[2] for p in levels.values()])
    x_values = np.linspace(x_min, x_max, 1000)
    plt.figure(figsize=(8, 4))
    for level, points in levels.items():
        y_values = [triangle_membership(x, points) for x in x_values]
        plt.plot(x_values, y_values, label=f"{level}")
    if show_input and input_value is not None:
        y_marker = []
        for level, points in levels.items():
            y = triangle_membership(input_value, points)
            y_marker.append(y)
            plt.plot([input_value, input_value], [0, y], 'k--')
            plt.plot(input_value, y, 'ro')
        plt.text(input_value, max(y_marker) + 0.05, f'Input = {input_value}', ha='center')
    plt.title(f"Fungsi Keanggotaan - {var_name}")
    plt.xlabel(var_name)
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

# Fungsi inferensi dengan metode Tsukamoto
def inferensi(request_count, security_level, anomalous_data):
    steps = {}
    # Fuzzifikasi
    request_memberships = get_input_membership(request_count_levels, request_count)
    security_memberships = get_input_membership(system_security_level_levels, security_level)
    data_memberships = get_input_membership(anomalous_data_volume_levels, anomalous_data)
    steps['fuzzification'] = {
        'Jumlah Permintaan Akses': request_memberships,
        'Tingkat Keamanan Sistem': security_memberships,
        'Volume Data Anomali': data_memberships
    }

    # Inferensi dan perhitungan z untuk setiap aturan
    z_values = []
    alphas = []
    rule_steps = []
    for idx, rule in enumerate(rules):
        antecedent = rule['antecedent']
        consequent = rule['consequent']
        alpha = min(request_memberships[antecedent[0]], security_memberships[antecedent[1]], data_memberships[antecedent[2]])
        if alpha > 0:
            z = get_z_value(consequent, alpha)
            z_values.append(z)
            alphas.append(alpha)
            rule_steps.append({
                'rule_number': idx + 1,
                'antecedent': antecedent,
                'consequent': consequent,
                'alpha': alpha,
                'z': z
            })
    steps['rule_evaluation'] = rule_steps

    # Defuzzifikasi (rata-rata tertimbang)
    if sum(alphas) == 0:
        z_final = 0
    else:
        z_final = sum([alpha * z for alpha, z in zip(alphas, z_values)]) / sum(alphas)
    steps['defuzzification'] = {
        'alphas': alphas,
        'z_values': z_values,
        'z_final': z_final
    }
    return z_final, steps

# Dataset dari gambar
data = [
    [200, 3, 50],
    [400, 2, 150],
    [150, 8, 30],
    [900, 1, 400],
    [250, 5, 80],
    [700, 3, 300],
    [100, 9, 20],
    [500, 7, 100],
    [800, 4, 350],
    [300, 6, 60]
]
actual_risks = [30, 60, 20, 95, 40, 85, 10, 50, 90, 35]

# Menu pilihan
st.title("Sistem Prediksi Tingkat Resiko Serangan Siber Menggunakan Metode Fuzzy Tsukamoto")
menu = st.sidebar.selectbox("Pilih Menu", ["Tampilkan Hasil Dataset", "Input Manual", "Lihat Perhitungan Manual"])

# Jika pilih menu "Tampilkan Hasil Dataset"
if menu == "Tampilkan Hasil Dataset":
    st.header("Hasil Prediksi dari Dataset")
    predicted_risks = []
    results = []
    for i, (request_count, security_level, anomalous_data) in enumerate(data):
        predicted_risk, _ = inferensi(request_count, security_level, anomalous_data)
        predicted_risks.append(predicted_risk)
        error = abs(predicted_risk - actual_risks[i])
        results.append({
            'Hari': f"Hari {i+1}",
            'Jumlah Permintaan Akses': request_count,
            'Tingkat Keamanan Sistem': security_level,
            'Volume Data Anomali': anomalous_data,
            'Prediksi': round(predicted_risk, 2),
            'Aktual': actual_risks[i],
            'Error': round(error, 2)
        })
    # Tampilkan tabel hasil
    df_results = pd.DataFrame(results)
    df_results.set_index('Hari', inplace=True)
    st.dataframe(df_results.style.format({
        'Jumlah Permintaan Akses': '{:,.0f}',
        'Tingkat Keamanan Sistem': '{:,.0f}',
        'Volume Data Anomali': '{:,.0f}',
        'Prediksi': '{:,.2f}',
        'Aktual': '{:,.0f}',
        'Error': '{:,.2f}'
    }))

    # Hitung MAE
    errors = [abs(pred - actual) for pred, actual in zip(predicted_risks, actual_risks)]
    mae = sum(errors) / len(errors)
    st.write(f"\n**Mean Absolute Error (MAE): {mae:.2f}**")

    st.write("\nPilih hari untuk melihat proses perhitungan:")
    selected_day = st.selectbox("Pilih Hari", [f"Hari {i+1}" for i in range(len(data))])
    selected_index = int(selected_day.split()[1]) - 1
    request_count, security_level, anomalous_data = data[selected_index]
    predicted_risk, steps = inferensi(request_count, security_level, anomalous_data)
    st.write(f"\n### Proses perhitungan untuk {selected_day}:")
    st.write(f"**Input:**")
    st.write(f"- Jumlah Permintaan Akses: {request_count}")
    st.write(f"- Tingkat Keamanan Sistem: {security_level}")
    st.write(f"- Volume Data Anomali: {anomalous_data}")
    st.write(f"**Prediksi:** {predicted_risk:.2f}")
    st.write(f"**Aktual:** {actual_risks[selected_index]}")

    # Checkbox untuk menampilkan langkah-langkah perhitungan
    show_steps = st.checkbox("Lihat Step")

    if show_steps:
        # Tampilkan langkah-langkah perhitungan
        st.subheader("Proses dan Tahap Algoritma Fuzzy Tsukamoto")

        st.write("### Fuzzifikasi")
        st.write("Derajat Keanggotaan untuk setiap variabel input:")
        for var_name, memberships in steps['fuzzification'].items():
            st.write(f"**{var_name}:**")
            df_memberships = pd.DataFrame(memberships.items(), columns=['Level', 'Derajat Keanggotaan'])
            st.dataframe(df_memberships.set_index('Level').style.format({'Derajat Keanggotaan': '{:.2f}'}))

        st.write("### Evaluasi Aturan")
        st.write("Aturan yang aktif dan perhitungan nilai α dan z:")
        if steps['rule_evaluation']:
            df_rules = pd.DataFrame([
                {
                    'Aturan': f"Aturan {rule_step['rule_number']}",
                    'Jika': f"{rule_step['antecedent']}",
                    'Maka': rule_step['consequent'],
                    'α (Derajat Kebenaran)': round(rule_step['alpha'], 2),
                    'Nilai z': round(rule_step['z'], 2)
                }
                for rule_step in steps['rule_evaluation']
            ])
            st.dataframe(df_rules.set_index('Aturan'))
        else:
            st.write("Tidak ada aturan yang aktif.")

        st.write("### Defuzzifikasi")
        total_alpha = sum(steps['defuzzification']['alphas'])
        if total_alpha == 0:
            st.write("Tidak ada aturan yang aktif.")
        else:
            st.write(f"Nilai z akhir (Tingkat Resiko Serangan Siber): **{predicted_risk:.2f}**")

    # Tampilkan grafik fungsi keanggotaan dengan atau tanpa posisi input
    st.header("Grafik Fungsi Keanggotaan")
    st.subheader("Jumlah Permintaan Akses")
    plot_membership(request_count_levels, "Jumlah Permintaan Akses", request_count, show_input=show_steps)

    st.subheader("Tingkat Keamanan Sistem")
    plot_membership(system_security_level_levels, "Tingkat Keamanan Sistem", security_level, show_input=show_steps)

    st.subheader("Volume Data Anomali")
    plot_membership(anomalous_data_volume_levels, "Volume Data Anomali", anomalous_data, show_input=show_steps)

    st.subheader("Tingkat Resiko Serangan Siber")
    plot_membership(cyber_attack_risk_level_levels, "Tingkat Resiko Serangan Siber", predicted_risk, show_input=show_steps)

# Jika pilih menu "Input Manual"
elif menu == "Input Manual":
    st.header("Input Data Manual")
    request_count = st.number_input("Jumlah Permintaan Akses (0-1000):", min_value=0, max_value=1000, value=500)
    security_level = st.number_input("Tingkat Keamanan Sistem (0-10):", min_value=0, max_value=10, value=5)
    anomalous_data = st.number_input("Volume Data Anomali (0-500):", min_value=0, max_value=500, value=250)

    # Hasil prediksi
    predicted_risk, steps = inferensi(request_count, security_level, anomalous_data)
    st.write(f"Tingkat Resiko Serangan Siber untuk input ini adalah: **{predicted_risk:.2f}**")

    # Checkbox untuk menampilkan langkah-langkah perhitungan
    show_steps = st.checkbox("Lihat Step")

    if show_steps:
        # Tampilkan langkah-langkah perhitungan
        st.subheader("Proses dan Tahap Algoritma Fuzzy Tsukamoto")

        st.write("### Fuzzifikasi")
        st.write("Derajat Keanggotaan untuk setiap variabel input:")
        for var_name, memberships in steps['fuzzification'].items():
            st.write(f"**{var_name}:**")
            df_memberships = pd.DataFrame(memberships.items(), columns=['Level', 'Derajat Keanggotaan'])
            st.dataframe(df_memberships.set_index('Level').style.format({'Derajat Keanggotaan': '{:.2f}'}))

        st.write("### Evaluasi Aturan")
        st.write("Aturan yang aktif dan perhitungan nilai α dan z:")
        if steps['rule_evaluation']:
            df_rules = pd.DataFrame([
                {
                    'Aturan': f"Aturan {rule_step['rule_number']}",
                    'Jika': f"{rule_step['antecedent']}",
                    'Maka': rule_step['consequent'],
                    'α (Derajat Kebenaran)': round(rule_step['alpha'], 2),
                    'Nilai z': round(rule_step['z'], 2)
                }
                for rule_step in steps['rule_evaluation']
            ])
            st.dataframe(df_rules.set_index('Aturan'))
        else:
            st.write("Tidak ada aturan yang aktif.")

        st.write("### Defuzzifikasi")
        total_alpha = sum(steps['defuzzification']['alphas'])
        if total_alpha == 0:
            st.write("Tidak ada aturan yang aktif.")
        else:
            st.write(f"Nilai z akhir (Tingkat Resiko Serangan Siber): **{predicted_risk:.2f}**")

    # Tampilkan grafik fungsi keanggotaan dengan atau tanpa posisi input
    st.header("Grafik Fungsi Keanggotaan")
    st.subheader("Jumlah Permintaan Akses")
    plot_membership(request_count_levels, "Jumlah Permintaan Akses", request_count, show_input=show_steps)

    st.subheader("Tingkat Keamanan Sistem")
    plot_membership(system_security_level_levels, "Tingkat Keamanan Sistem", security_level, show_input=show_steps)

    st.subheader("Volume Data Anomali")
    plot_membership(anomalous_data_volume_levels, "Volume Data Anomali", anomalous_data, show_input=show_steps)

    st.subheader("Tingkat Resiko Serangan Siber")
    plot_membership(cyber_attack_risk_level_levels, "Tingkat Resiko Serangan Siber", predicted_risk, show_input=show_steps)

# Jika pilih menu "Lihat Perhitungan Manual"
elif menu == "Lihat Perhitungan Manual":
    st.header("Perhitungan Manual Sistem Fuzzy Tsukamoto")
    
    st.write("### Pilih Data untuk Perhitungan Manual")
    # Pilih data dari dataset atau masukkan data manual
    perhitungan_menu = st.selectbox("Pilih Metode Perhitungan", ["Dataset", "Input Manual"])

    if perhitungan_menu == "Dataset":
        selected_day = st.selectbox("Pilih Hari", [f"Hari {i+1}" for i in range(len(data))])
        selected_index = int(selected_day.split()[1]) - 1
        request_count, security_level, anomalous_data = data[selected_index]
        actual_risk = actual_risks[selected_index]
    else:
        st.write("### Input Data Manual")
        request_count = st.number_input("Jumlah Permintaan Akses (0-1000):", min_value=0, max_value=1000, value=500, key='manual_request')
        security_level = st.number_input("Tingkat Keamanan Sistem (0-10):", min_value=0, max_value=10, value=5, key='manual_security')
        anomalous_data = st.number_input("Volume Data Anomali (0-500):", min_value=0, max_value=500, value=250, key='manual_data')
        actual_risk = st.number_input("Tingkat Resiko Aktual:", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key='manual_actual')

    # Tampilkan data yang dipilih
    st.write(f"\n### Data yang Dipilih:")
    st.write(f"- **Jumlah Permintaan Akses:** {request_count}")
    st.write(f"- **Tingkat Keamanan Sistem:** {security_level}")
    st.write(f"- **Volume Data Anomali:** {anomalous_data}")
    if perhitungan_menu == "Dataset":
        st.write(f"- **Tingkat Resiko Aktual:** {actual_risk}")

    # Lakukan inferensi
    predicted_risk, steps = inferensi(request_count, security_level, anomalous_data)
    error = abs(predicted_risk - actual_risk) if perhitungan_menu == "Dataset" else abs(predicted_risk - actual_risk)
    st.write(f"- **Prediksi Tingkat Resiko:** {predicted_risk:.2f}")
    if perhitungan_menu == "Dataset":
        st.write(f"- **Tingkat Resiko Aktual:** {actual_risk}")
        st.write(f"- **Error (|Prediksi - Aktual|):** {error:.2f}")

    st.write("\n### Langkah 1: Fuzzifikasi Input")
    st.write("Fuzzifikasi adalah proses mengubah nilai numerik input menjadi derajat keanggotaan pada himpunan fuzzy masing-masing variabel.")
    
    # Tampilkan rumus fungsi keanggotaan segitiga
    st.latex(r"""
        \mu_{\text{Level}}(x) =
        \begin{cases} 
        0 & \text{jika } x \leq a \\
        \frac{x - a}{b - a} & \text{jika } a < x \leq b \\
        \frac{c - x}{c - b} & \text{jika } b < x < c \\
        0 & \text{jika } x \geq c 
        \end{cases}
    """)

    st.write("#### Contoh Perhitungan Derajat Keanggotaan:")
    st.write("Misalnya, untuk **Jumlah Permintaan Akses** dengan input \( x = {} \).".format(request_count))
    
    # Menampilkan perhitungan derajat keanggotaan untuk setiap variabel
    for var_name, levels in zip(
        ["Jumlah Permintaan Akses", "Tingkat Keamanan Sistem", "Volume Data Anomali"],
        [request_count_levels, system_security_level_levels, anomalous_data_volume_levels]
    ):
        x = request_count if var_name == "Jumlah Permintaan Akses" else security_level if var_name == "Tingkat Keamanan Sistem" else anomalous_data
        st.write(f"**{var_name}:**")
        for level, points in levels.items():
            a, b, c = points
            # Menentukan rumus yang digunakan berdasarkan x
            if x <= a:
                degree = 0.0
                explanation = f"Jika {x} ≤ {a}, maka μ({level}) = 0"
            elif a < x <= b:
                degree = (x - a) / (b - a) if (b - a) != 0 else 0.0
                explanation = f"Jika {a} < {x} ≤ {b}, maka μ({level}) = ({x} - {a}) / ({b} - {a}) = {x - a} / {b - a} = {degree:.2f}"
            elif b < x < c:
                degree = (c - x) / (c - b) if (c - b) != 0 else 0.0
                explanation = f"Jika {b} < {x} < {c}, maka μ({level}) = ({c} - {x}) / ({c} - {b}) = {c - x} / {c - b} = {degree:.2f}"
            else:
                degree = 0.0
                explanation = f"Jika {x} ≥ {c}, maka μ({level}) = 0"
            st.write(f"- **{level}:** {degree:.2f} ({explanation})")
    st.write("")  # Spasi

    # Tampilkan dataframe fuzzifikasi
    st.write("#### Derajat Keanggotaan:")
    fuzz_df = pd.DataFrame({
        "Variabel": ["Jumlah Permintaan Akses", "Tingkat Keamanan Sistem", "Volume Data Anomali"],
        "Rendah": [
            triangle_membership(request_count, request_count_levels["Rendah"]),
            triangle_membership(security_level, system_security_level_levels["Rendah"]),
            triangle_membership(anomalous_data, anomalous_data_volume_levels["Rendah"])
        ],
        "Sedang": [
            triangle_membership(request_count, request_count_levels["Sedang"]),
            triangle_membership(security_level, system_security_level_levels["Sedang"]),
            triangle_membership(anomalous_data, anomalous_data_volume_levels["Sedang"])
        ],
        "Tinggi": [
            triangle_membership(request_count, request_count_levels["Tinggi"]),
            triangle_membership(security_level, system_security_level_levels["Tinggi"]),
            triangle_membership(anomalous_data, anomalous_data_volume_levels["Tinggi"])
        ]
    })
    fuzz_df = fuzz_df.set_index("Variabel")
    st.dataframe(fuzz_df.style.format("{:.2f}"))

    st.write("\n### Langkah 2: Evaluasi Aturan Fuzzy")
    st.write("Aturan yang aktif dan perhitungan nilai α dan z berdasarkan himpunan fuzzy input.")
    
    st.latex(r"""
        \alpha = \min(\mu_{\text{Permintaan Akses}}(\text{Level A}), \mu_{\text{Keamanan Sistem}}(\text{Level B}), \mu_{\text{Data Anomali}}(\text{Level C}))
    """)

    # Menampilkan aturan yang aktif
    if steps['rule_evaluation']:
        st.write("#### Aturan yang Aktif:")
        df_rules = pd.DataFrame([
            {
                'Aturan': f"Aturan {rule_step['rule_number']}",
                'Jika': f"{rule_step['antecedent']}",
                'Maka': rule_step['consequent'],
                'α (Derajat Kebenaran)': round(rule_step['alpha'], 2),
                'Nilai z': round(rule_step['z'], 2)
            }
            for rule_step in steps['rule_evaluation']
        ])
        st.dataframe(df_rules.set_index('Aturan'))
    else:
        st.write("Tidak ada aturan yang aktif.")

    st.write("\n### Langkah 3: Agregasi Output Aturan")
    st.write("Mengumpulkan semua nilai \( z \) dan \( \alpha \) dari aturan yang diaktifkan.")
    st.latex(r"""
        z_{\text{final}} = \frac{\sum (\alpha_i \times z_i)}{\sum \alpha_i}
    """)

    # Menampilkan agregasi
    if steps['rule_evaluation']:
        st.write("#### Perhitungan Defuzzifikasi:")
        sum_alpha_z = sum([alpha * z for alpha, z in zip(steps['defuzzification']['alphas'], steps['defuzzification']['z_values'])])
        sum_alpha = sum(steps['defuzzification']['alphas'])
        st.latex(rf"""
            z_{{\text{{final}}}} = \frac{{{sum_alpha_z:.2f}}}{{{sum_alpha:.2f}}} = {steps['defuzzification']['z_final']:.2f}
        """)
    else:
        st.write("Tidak ada aturan yang aktif untuk diaggregasi.")

    # Menampilkan hasil defuzzifikasi
    st.write("#### Hasil Defuzzifikasi:")
    if sum(steps['defuzzification']['alphas']) == 0:
        st.write("Tidak ada aturan yang aktif, sehingga nilai z akhir adalah 0.")
    else:
        st.write(f"**Tingkat Resiko Serangan Siber:** {steps['defuzzification']['z_final']:.2f}")

    # Tampilkan grafik fungsi keanggotaan
    st.write("\n### Langkah 4: Grafik Fungsi Keanggotaan")
    st.subheader("Jumlah Permintaan Akses")
    plot_membership(request_count_levels, "Jumlah Permintaan Akses", request_count, show_input=True)

    st.subheader("Tingkat Keamanan Sistem")
    plot_membership(system_security_level_levels, "Tingkat Keamanan Sistem", security_level, show_input=True)

    st.subheader("Volume Data Anomali")
    plot_membership(anomalous_data_volume_levels, "Volume Data Anomali", anomalous_data, show_input=True)

    st.subheader("Tingkat Resiko Serangan Siber")
    plot_membership(cyber_attack_risk_level_levels, "Tingkat Resiko Serangan Siber", steps['defuzzification']['z_final'], show_input=True)

    # Tampilkan perbandingan jika data dari dataset
    if perhitungan_menu == "Dataset":
        st.write("\n### Perbandingan dengan Tingkat Resiko Aktual")
        st.write(f"- **Prediksi:** {steps['defuzzification']['z_final']:.2f}")
        st.write(f"- **Aktual:** {actual_risk}")
        st.write(f"- **Error:** {error:.2f}")

    elif perhitungan_menu == "Input Manual":
        st.write("\n### Perbandingan dengan Tingkat Resiko Aktual")
        st.write(f"- **Prediksi:** {steps['defuzzification']['z_final']:.2f}")
        st.write(f"- **Aktual:** {actual_risk}")
        st.write(f"- **Error:** {error:.2f}")
