import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from fuzzy_logic.membership_functions import triangle_membership
from fuzzy_logic.inference import cyber_attack_risk_level_levels

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