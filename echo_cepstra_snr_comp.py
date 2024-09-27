import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.fft import fft, ifft


# Setup
st.set_page_config(page_title='Echo Cepstra SnR', page_icon="🎤", layout="wide")
color_list = px.colors.qualitative.Light24


# Constants for cepstrum calculations
NFFT = 4096
VELOCITY = 343
ECHO_FS = 20161


# Funcs
def read_dat_file(fn: str):
    """
    Read .dat file from FG device containing Echo audio
    Args:
        fn: path to input file
    """
    data = []
    with open(fn, "rb") as binary_file:
        while True:
            couple_bytes = binary_file.read(2)
            if not couple_bytes:
                break
            data.append(int.from_bytes(couple_bytes, byteorder='little', signed=True))

    return np.float32(data)


def calc_spectrum(data: np.ndarray, fs: int):
    """
    Calculate spectrum of audio data
    Args:
        data: audio data as read from
        fs: sample rate of audio data
    """
    # Start and end offsets to account for motor wind up/wind down
    start_samp = int(fs * 20)
    end_samp = int(fs * 10)
    # Window size
    WND_SKIP = 1024
    n_avgs = 0
    X_log_avg = np.zeros(NFFT, dtype=float)
    fv = np.linspace(0, fs, NFFT)

    for i in range(start_samp, len(data) - end_samp, WND_SKIP):
        x = data[i:i + NFFT]
        X = np.abs(fft(x, n=NFFT)) + np.finfo(float).eps
        X_log_avg += np.log(X)
        n_avgs += 1

    X_log_avg /= n_avgs

    return X_log_avg, fv


def get_snr_cep(qf: np.ndarray, cep: np.ndarray):
    """
    Calculate SnR metric of a given cepstrum
    Args:
        qf: quefrency
        cep: cepstrum
    """
    # Expected region of signal (1.5-2.5 metres)
    l_sig = 1.5
    r_sig = 2.5
    idx_l_sig = np.argmin(np.abs(qf - l_sig))
    idx_r_sig = np.argmin(np.abs(qf - r_sig))
    # Expected region of noise (3-4 metres)
    l_noise = 3
    r_noise = 4
    idx_l_noise = np.argmin(np.abs(qf - l_noise))
    idx_r_noise = np.argmin(np.abs(qf - r_noise))

    # Calculate the signal and noise
    sig = np.max(cep[idx_l_sig:idx_r_sig]) - np.min(cep[idx_l_sig:idx_r_sig])
    noise = np.std(cep[idx_l_noise:idx_r_noise])

    # Return the ratio
    return sig / noise


def generate_df(base_folder: str):
    """
    Loop through the provided base_folder, looking for .dat files containing Echo data and process SnR, Spec and Cep
    Args:
        base_folder: Path to initial search directory
    """
    # Search folder for DAT files
    files_dat = glob.glob(os.path.join(base_folder, '**', '*.dat'), recursive=True)

    # Object to store each tests cepstrum data
    r_dict = {}

    for f in files_dat:
        # Figure out the base folder name for a given test
        path = os.path.normpath(f)
        # Basename becomes the unique identifier for each test
        basename = path.split(os.sep)[-2]
        if basename == 'E':
            basename = path.split(os.sep)[-3]

        # Tag for different prototypes, used to group SnR metric box plot
        # TODO adjust this code to filter correctly from the folder name
        #  "tag" can be whatever grouping is desired (prototype design, pressure, mask, tube, device, material etc.)
        tag = basename.split('_')[5]

        # Read audio dat file
        audio = read_dat_file(f)
        audio /= ((2 ** 15) - 1)

        if len(audio) / ECHO_FS < 30:
            continue

        t = np.arange(0, NFFT) * (1/ECHO_FS)
        quefrency = np.array(VELOCITY * t / 2)

        # Calculate spectrum & cepstrum
        spec, fv = calc_spectrum(audio, ECHO_FS)
        cep = ifft(spec).real
        snr = get_snr_cep(quefrency, cep)

        # Combine into dict object
        r_dict[basename] = {
            'tag': tag,
            'fv': fv,
            'spec': spec,
            'que': quefrency,
            'cep': cep,
            'snr_cep': snr
        }

    dataframe = pd.DataFrame.from_dict(r_dict, orient='index')
    return dataframe


#
# Main app
#
st.sidebar.subheader('Setup')

base_dir = st.sidebar.text_input('Folder path:', '')

# Wait until a path is input
if base_dir:
    # Generate data
    df = generate_df(base_dir)

    # Filters
    normalise_cep = st.sidebar.checkbox('Normalise cepstra', 1)

    # Show the dataframe (for debugging)
    # with st.expander('Show df', expanded=False):
    #     st.write(df)

    # Boxplot summaries
    df_sorted = df.sort_values('tag')
    fig_snr = px.box(df_sorted, x='tag', y='snr_cep', points='all')

    fig_snr.update_layout(title='SnR Metric', height=400)
    fig_snr.update_yaxes(title='Snr [arb]')

    # Spectrum plot setup
    fig_spec = go.Figure()
    fig_spec.update_layout(title='Spectrum', height=700, width=1000)
    fig_spec.update_xaxes(range=[0, 10e3], title='Frequency [Hz]')
    fig_spec.update_yaxes(title='log |X|')

    # Cepstrum plot setup
    fig_cep = go.Figure()
    fig_cep.update_layout(title='Cepstrum', height=700, width=1000)
    fig_cep.update_xaxes(range=[0, 5], title='Distance [m]')
    cep_data_range = [-80, 80] if normalise_cep else [-0.04, 0.04]
    fig_cep.update_yaxes(range=cep_data_range, title='cep [arb]')

    # Add the data to the plots
    for ix, row in df.iterrows():
        if normalise_cep:
            # Normalise the cepstra
            ix = np.where((row['que'] > 1.2) & (row['que'] < 1.7))
            row['cep'] /= np.std(row['cep'][ix])

        # Plot spectrum data
        fig_spec.add_trace(go.Scattergl(x=row['fv'], y=row['spec'], name=row.name))
        # Plot cepstrum data
        fig_cep.add_trace(go.Scattergl(x=row['que'], y=row['cep'], name=row.name))

    # Plots
    st.plotly_chart(fig_snr, use_container_width=True)
    st.plotly_chart(fig_spec, use_container_width=True)
    st.plotly_chart(fig_cep, use_container_width=True)
