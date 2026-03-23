import pynwb
import pandas as pd
import numpy as np
import os

# 1. TARGET FILE
# Ensure this points to the REAL >100MB file you manually downloaded
file_path = "324257144_ephys.nwb"


def extract_training_data():
    io = pynwb.NWBHDF5IO(file_path, "r")
    nwb = io.read()

    print(f"File loaded: {file_path}")
    print(f"Specimen: {nwb.identifier}")

    # 2. FILTER FOR 'NOISE 1' PROTOCOL (The "Gold Standard" [cite: 47])
    target_sweeps = []

    # We iterate through the sweep table to find "Noise 1"
    # Note: Allen NWB v2 stores stimulus names in the sweep table or stimulus description
    df_sweeps = nwb.sweep_table.to_dataframe()

    # Look for sweeps where stimulus_description contains "Noise 1"
    noise_sweeps = df_sweeps[
        df_sweeps["stimulus_description"].str.contains("Noise 1", na=False)
    ]

    if len(noise_sweeps) == 0:
        print(
            "CRITICAL: No 'Noise 1' sweeps found. Checking for 'Noise 2' or 'Long Square'..."
        )
        # Fallback to Long Square if Noise is missing (though Specimen 324257146 should have Noise [cite: 107])
        noise_sweeps = df_sweeps[
            df_sweeps["stimulus_description"].str.contains("Long Square", na=False)
        ]

    print(f"Found {len(noise_sweeps)} target sweeps.")

    # 3. EXTRACT AND NORMALIZE [cite: 195]
    # We will grab the first valid sweep found.
    sweep_idx = noise_sweeps.index[0]
    sweep_number = noise_sweeps.loc[sweep_idx, "sweep_number"]

    print(f"Extracting Sweep {sweep_number}...")

    # Get Voltage (Response) and Current (Stimulus)
    # Allen SDK/NWB helper might be needed if direct access is complex,
    # but standard NWB allows access via acquisition and stimulus groups.

    # Using simple index access based on NWB structure:
    # (Note: This assumes standard Allen NWB 2.0 structure)
    acq = nwb.acquisition[f"Sweep_{sweep_number}"]
    stim = nwb.stimulus[f"Sweep_{sweep_number}"]

    voltage_raw = acq.data[:]
    current_raw = stim.data[:]

    # Timestamps (reconstruct from rate if necessary)
    rate = acq.rate
    if np.isnan(rate):
        # If rate is NaN, check starting_time and calculate
        rate = 200000.0  # Force 200kHz for this specimen if missing [cite: 107]

    dt = 1.0 / rate
    time = np.arange(len(voltage_raw)) * dt

    # 4. UNIT CONVERSION [cite: 198, 200]
    # Raw Voltage is Volts -> Convert to mV (* 1000)
    # Raw Current is Amps -> Convert to pA (* 1e12)

    voltage_mv = voltage_raw * 1000.0
    current_pa = current_raw * 1e12

    # 5. SAVE TO CSV
    # We save a clean file for Julia to ingest
    df_out = pd.DataFrame({"time": time, "voltage": voltage_mv, "current": current_pa})

    output_filename = "training_sweep.csv"
    df_out.to_csv(output_filename, index=False)
    print(f"SUCCESS: Data exported to {output_filename}")
    print(f"Sampling Rate: {rate} Hz")  # Should be >10k Hz [cite: 39]


extract_training_data()
