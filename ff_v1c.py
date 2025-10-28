import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
DATA_FILE = "data.dat"
DEFAULT_G = 9.8069
DEFAULT_M = 0.005
DEFAULT_K = 0.0001
AIR = [DEFAULT_M, DEFAULT_K]

# ---------------------------------------------------------------------
# Air Friction Calculation
# ---------------------------------------------------------------------
def calculate_k(rho, Cd, r):
    A = math.pi * r * r
    return 0.5 * rho * Cd * A

# ---------------------------------------------------------------------
# UNIVERSAL FILE I/O HANDLER
# ---------------------------------------------------------------------
def handle_file_io(mode, default_values=None, write_values=None):
    
##    Reads or writes g, m, k values to/from a single file: data.dat
   
    try:
        if mode == 'read':
            if not os.path.exists(DATA_FILE):
                print(f"No data file found. Using defaults: g={DEFAULT_G}, m={DEFAULT_M}, k={DEFAULT_K}")
                handle_file_io('write', write_values=default_values)
                return default_values

            with open(DATA_FILE, 'r') as file:
                lines = file.readlines()
                if len(lines) < 3:
                    raise ValueError("data.dat is malformed or missing values.")
                g = float(lines[0].strip())
                m = float(lines[1].strip())
                k = float(lines[2].strip())
                print(f"Loaded values from {DATA_FILE}: g={g:.4f}, m={m:.3f}, k={k:.5f}")
                return [g, m, k]

        elif mode == 'write':
            with open(DATA_FILE, 'w') as file:
                for val in write_values:
                    file.write(f"{val}\n")
            print(f"Values saved to {DATA_FILE}: g={write_values[0]:.4f}, m={write_values[1]:.3f}, k={write_values[2]:.5f}")
    except Exception as e:
        print(f"File error on {DATA_FILE}: {e}")
        return default_values if mode == 'read' else None

# ---------------------------------------------------------------------
# MAIN MENU DISPLAY
# ---------------------------------------------------------------------
def display_menu(g):
    print("\n" + "*" * 80)
    print("Welcome to Synthetic Free Fall Data Generator")
    print(f"Current Local values - g: {g:.4f} m/s², m: {AIR[0]:.3f} kg, k: {AIR[1]:.5f} Ns/m")
    print("*" * 80 + "\n")
    print("1 - Change local g value")
    print("2 - Single Student Data (No air friction)")
    print("3 - Single Student Data with air friction")
    print("4 - Multiple Students Data (No air friction)")
    print("5 - Multiple Students Data with air friction")
    print("6 - Change object parameters for air friction")
    print("7 - Exit\n")

# ---------------------------------------------------------------------
# INPUT HANDLER
# ---------------------------------------------------------------------
def get_float_input(prompt, convert_cm_to_m=False):
    try:
        value = float(input(prompt))
        return value / 100 if convert_cm_to_m else value
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return get_float_input(prompt, convert_cm_to_m)

# ---------------------------------------------------------------------
# DATA GENERATION FUNCTIONS
# ---------------------------------------------------------------------
def generate_data_no_friction(g, filename, h_initial, h_final,
                              h_increment, n_trials, h_noise_level, t_noise_level):
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Height_m", "Time_s", "Time^2_s2"])

            h_current = h_initial
            while h_current <= h_final:
                h_with_noise = h_current + np.random.normal(0, h_noise_level)

                for _ in range(n_trials):
                    t_theoretical = math.sqrt(2 * max(h_with_noise, 0) / g)
                    t_measured = t_theoretical + np.random.normal(0, t_noise_level)
                    writer.writerow([h_current, t_measured, t_measured ** 2])

                h_current += h_increment

        print(f"Data successfully saved to '{filename}'.")
        draw_graph(filename)

    except PermissionError:
        print(f"Permission denied when writing '{filename}'. Please close the file.")
    except OSError as e:
        print(f"File system error while writing '{filename}': {e}")
    except Exception as e:
        print(f"Unexpected error while writing '{filename}': {e}")

def generate_data_with_friction(g, m, k, filename, h_initial, h_final,
                                h_increment, n_trials, h_noise_level, t_noise_level):
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Height_m", "Time_s", "Time^2_s2"])

            h_current = h_initial
            while h_current <= h_final:
                h_with_noise = h_current + np.random.normal(0, h_noise_level)

                for _ in range(n_trials):
                    try:
                        tf = math.sqrt(m / (g * k)) * math.acosh(math.exp(k * max(h_with_noise, 0) / m))
                    except ValueError:
                        tf = math.nan

                    t_measured = tf + np.random.normal(0, t_noise_level)
                    writer.writerow([h_current, t_measured, t_measured ** 2])

                h_current += h_increment

        print(f"Data with air friction successfully saved to '{filename}'.")
        draw_graph(filename)

    except PermissionError:
        print(f"Permission denied when writing '{filename}'. Please close the file.")
    except OSError as e:
        print(f"File system error while writing '{filename}': {e}")
    except Exception as e:
        print(f"Unexpected error while writing '{filename}': {e}")

# ---------------------------------------------------------------------
# GRAPH PLOTTING
# ---------------------------------------------------------------------
def draw_graph(filename):
    try:
        if not os.path.exists(filename):
            print(f"File '{filename}' not found for plotting.")
            return

        df = pd.read_csv(filename)
        if df.empty:
            print(f"No data found in '{filename}'.")
            return

        x = df["Time^2_s2"]
        y = df["Height_m"]

        coeffs = np.polyfit(x, y, 1)
        a, b = coeffs
        trendline = np.poly1d(coeffs)

        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, color='blue', alpha=0.7, label='Data Points')
        plt.plot(x, trendline(x), color='red', label=f'Fit: h = {a:.3f}·t² + {b:.4f}')
        plt.xlabel("Time² (s²)")
        plt.ylabel("Height (m)")
        plt.title(f"Free Fall: {os.path.basename(filename)}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_name = filename.replace(".csv", ".png")
        plt.savefig(plot_name, dpi=300)
        plt.show()

        print(f"Graph saved as '{plot_name}'.")
        print(f"Linear Fit: h = {a:.4f}·t² + {b:.5f}\n")
    except pd.errors.EmptyDataError:
        print(f"️'{filename}' is empty or corrupted.")
    except Exception as e:
        print(f"Error while plotting '{filename}': {e}")

# ---------------------------------------------------------------------
# EXPERIMENT MODES
# ---------------------------------------------------------------------
def run_experiment(mode, g, m=None, k=None):
    is_multi = mode.startswith("multi")
    has_friction = "with_friction" in mode

    num_students = int(get_float_input("Enter number of students: ")) if is_multi else 1

    h_initial = get_float_input("Enter initial height (cm): ", True)
    h_final = get_float_input("Enter final height (cm): ", True)
    h_increment = get_float_input("Enter height increment (cm): ", True)
    n_trials = int(get_float_input("Enter number of trials per student: "))

    print("\nSelect noise type:")
    print("1 - No noise")
    print("2 - Noise in height")
    print("3 - Noise in time")
    print("4 - Noise in height and time\n")
    noise_option = input("Enter your choice: ")

    h_noise_level = t_noise_level = 0
    if noise_option in ("2", "4"):
        h_noise_level = get_float_input("Enter height noise coefficient: ")
    if noise_option in ("3", "4"):
        t_noise_level = get_float_input("Enter time noise coefficient: ")

    for i in range(1, num_students + 1):
        filename = f"student_{i}_free_fall.csv" if is_multi else "synthetic_free_fall_data.csv"
        if is_multi:
            print(f"\nGenerating data for Student {i}...")

        if has_friction:
            if m is None or k is None:
                raise ValueError("Mass (m) and friction coefficient (k) are required for air friction experiments.")
            generate_data_with_friction(g, m, k, filename, h_initial, h_final,
                                        h_increment, n_trials, h_noise_level, t_noise_level)
        else:
            generate_data_no_friction(g, filename, h_initial, h_final,
                                      h_increment, n_trials, h_noise_level, t_noise_level)

# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------
def main():
    while True:
        g, AIR[0], AIR[1] = handle_file_io('read', default_values=[DEFAULT_G, DEFAULT_M, DEFAULT_K])
        display_menu(g)
        choice = input("Enter your choice: ").strip()

        match choice:
            case "1":
                g = get_float_input("Enter new local g value (m/s²): ")
                handle_file_io('write', write_values=[g, AIR[0], AIR[1]])
            case "2":
                run_experiment("single_no_friction", g)
            case "3":
                run_experiment("single_with_friction", g, AIR[0], AIR[1])
            case "4":
                run_experiment("multi_no_friction", g)
            case "5":
                run_experiment("multi_with_friction", g, AIR[0], AIR[1])
            case "6":
                m = get_float_input("Enter new m value (kg): ")
                rho = get_float_input("Enter new rho value (kg/m³): ")
                Cd = get_float_input("Enter new Drag coefficient value (Cd): ")
                r = get_float_input("Enter new radius value (m): ")
                k = calculate_k(rho, Cd, r)
                AIR[0], AIR[1] = m, k
                handle_file_io('write', write_values=[g, m, k])
            case "7":
                print("Goodbye!")
                break
            case _:
                print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
