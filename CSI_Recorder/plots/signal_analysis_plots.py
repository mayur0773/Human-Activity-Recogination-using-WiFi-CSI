import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# IEEE SIGNAL ANALYSIS PLOTS
# ==========================================================

DATASET_FOLDER = "Processed_Dataset"

OUTPUT_FOLDER = os.path.join(
    "Graphs",
    "Signal_Analysis"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

ACTIVITIES = [
    "Empty",
    "Sitting",
    "Standing",
    "Walking"
]

# ----------------------------------------------------------
# IEEE Plot Style
# ----------------------------------------------------------

plt.rcParams.update({

    "font.family": "Times New Roman",

    "font.size": 12,

    "axes.titlesize": 14,

    "axes.labelsize": 12,

    "legend.fontsize": 10,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10

})

COLORS = {

    "Empty": "tab:blue",

    "Sitting": "tab:green",

    "Standing": "tab:orange",

    "Walking": "tab:red"

}

def parse_csi(packet):

    try:

        return np.array(
            ast.literal_eval(packet),
            dtype=float
        )

    except:

        return None


def load_activity(activity):

    folder = os.path.join(
        DATASET_FOLDER,
        activity
    )

    if not os.path.exists(folder):

        print(activity, "folder missing")

        return None

    csv_files = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".csv")
    )

    all_packets = []

    for csv in csv_files:

        path = os.path.join(folder, csv)

        df = pd.read_csv(path)

        for _, row in df.iterrows():

            csi = parse_csi(row["CSI_Data"])

            if csi is None:
                continue

            all_packets.append(csi)

    if len(all_packets) == 0:
        return None

    subcarriers = np.array(all_packets)

    amplitude = np.mean(
        np.abs(subcarriers),
        axis=1
    )

    return {
        "activity": activity,
        "subcarriers": subcarriers,
        "amplitude": amplitude
    }

    csv_files = sorted(

        f for f in os.listdir(folder)

        if f.endswith(".csv")

    )

    all_packets = []

    for csv in csv_files:

        path = os.path.join(folder, csv)

        df = pd.read_csv(path)

        for _, row in df.iterrows():

            csi = parse_csi(row["CSI_Data"])

            if csi is None:

                continue

            all_packets.append(csi)

    if len(all_packets) == 0:

        return None

    subcarriers = np.array(all_packets)

    amplitude = np.mean(
        np.abs(subcarriers),
        axis=1
    )

    return {

        "activity": activity,

        "subcarriers": subcarriers,

        "amplitude": amplitude

    }

data = []

print("\nLoading Processed Dataset...\n")

for activity in ACTIVITIES:

    d = load_activity(activity)

    if d is None:

        continue

    print(

        activity,

        "Packets :", len(d["amplitude"])

    )

    data.append(d)

print("\nDataset Loaded Successfully\n")

def save_figure(filename):

    plt.grid(
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_FOLDER,

            filename

        ),

        dpi=600,

        bbox_inches="tight"

    )

    plt.close()

    # ==========================================================
# FIGURE 1 : MEAN CSI AMPLITUDE
# ==========================================================

plt.figure(figsize=(8,5))

for d in data:

    plt.plot(
        d["amplitude"],
        linewidth=1.8,
        color=COLORS[d["activity"]],
        label=d["activity"]
    )

plt.title(
    "Mean CSI Amplitude Comparison",
    fontweight="bold"
)

plt.xlabel("Packet Number")

plt.ylabel("Mean CSI Amplitude")

plt.legend()

save_figure("Fig_01_Mean_Amplitude.png")

print("✓ Fig_01_Mean_Amplitude.png Saved")

# ==========================================================
# FIGURE 2 : AVERAGE SUBCARRIER AMPLITUDE
# ==========================================================

plt.figure(figsize=(8,5))

for d in data:

    average_subcarrier = np.mean(
        np.abs(d["subcarriers"]),
        axis=0
    )

    plt.plot(
        average_subcarrier,
        linewidth=1.8,
        color=COLORS[d["activity"]],
        label=d["activity"]
    )

plt.title(
    "Average CSI Subcarrier Amplitude",
    fontweight="bold"
)

plt.xlabel("Subcarrier Index")

plt.ylabel("Average Amplitude")

plt.legend()

save_figure("Fig_02_Subcarrier.png")

print("✓ Fig_02_Subcarrier.png Saved")

# ==========================================================
# FIGURE 3 : CSI HEATMAPS
# ==========================================================

for d in data:

    plt.figure(figsize=(8,5))

    plt.imshow(
        np.abs(d["subcarriers"]),
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )

    cbar = plt.colorbar()
    cbar.set_label("Amplitude")

    plt.title(
        f"CSI Heatmap - {d['activity']}",
        fontweight="bold"
    )

    plt.xlabel("Subcarrier Index")
    plt.ylabel("Packet Number")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"Fig_03_Heatmap_{d['activity']}.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

print("✓ CSI Heatmaps Saved")

# ==========================================================
# FIGURE 4 : FFT SPECTRUM
# ==========================================================

plt.figure(figsize=(8,5))

for d in data:

    signal = d["amplitude"]

    # Remove DC component
    signal = signal - np.mean(signal)

    # FFT
    fft = np.abs(np.fft.rfft(signal))

    freq = np.fft.rfftfreq(len(signal), d=1)

    plt.plot(
        freq,
        fft,
        linewidth=1.8,
        color=COLORS[d["activity"]],
        label=d["activity"]
    )

plt.title(
    "FFT Spectrum of CSI Amplitude",
    fontweight="bold"
)

plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")

plt.legend()

save_figure("Fig_04_FFT_Spectrum.png")

print("✓ Fig_04_FFT_Spectrum.png Saved")

# ==========================================================
# FIGURE 5 : HISTOGRAMS
# ==========================================================

for d in data:

    plt.figure(figsize=(8,5))

    plt.hist(
        d["amplitude"],
        bins="auto",
        color=COLORS[d["activity"]],
        edgecolor="black",
        alpha=0.8
    )

    plt.title(
        f"Amplitude Distribution - {d['activity']}",
        fontweight="bold"
    )

    plt.xlabel("Mean CSI Amplitude")
    plt.ylabel("Frequency")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"Fig_05_Histogram_{d['activity']}.png"
        ),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

print("✓ Histograms Saved")

# ==========================================================
# FIGURE 6 : BOXPLOT
# ==========================================================

plt.figure(figsize=(8,5))

values = [d["amplitude"] for d in data]

labels = [d["activity"] for d in data]

plt.boxplot(
    values,
    tick_labels=labels,
    showmeans=True,
    showfliers=False
)

plt.title(
    "CSI Amplitude Distribution",
    fontweight="bold"
)

plt.ylabel("Mean CSI Amplitude")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "Fig_06_Boxplot.png"
    ),
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("✓ Boxplot Saved")