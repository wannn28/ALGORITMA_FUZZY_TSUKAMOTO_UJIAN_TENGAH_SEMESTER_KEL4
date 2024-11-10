import numpy as np
import pandas as pd
from fuzzy_logic.rules import rules
from fuzzy_logic.membership_functions import triangle_membership

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

def get_input_membership(levels, x):
    memberships = {}
    for level, points in levels.items():
        memberships[level] = triangle_membership(x, points)
    return memberships

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