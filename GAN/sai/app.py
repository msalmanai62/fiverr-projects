import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from generator_disc import build_generator, ModelMonitor
import tensorflow as tf
from trainnn import prepare_data, fashganModel


generator = build_generator()
generator.load_weights('models/myGen.h5')


def generate_synthetic_data(dataset_folder=None):
    imgs = generator.predict(tf.random.normal((16, 128, 1)))
    return imgs


def train_model(ds, modell, epochs=20):
    st.write('Training...')
    hist = modell.fit(ds, epochs=epochs, callbacks=[ModelMonitor()])
    st.write('Training done')
    return hist


# Streamlit UI
def main():
    st.title('Synthetic Data Generation')

    # Sidebar to select dataset folder
    st.sidebar.title('Select Dataset Folder')
    dataset_folder = st.sidebar.selectbox('Folder Path', os.listdir())

    # Button to generate synthetic data
    if st.sidebar.button('Generate Synthetic Data'):
        progress_bar = st.progress(0)
        with st.spinner('Generating synthetic data...'):
            time.sleep(1)  # Simulated delay
            progress_bar.progress(100)
            st.success('Synthetic data generation successful!')

            # Displaying few samples of generated synthetic data
            st.subheader('Sample of Generated Synthetic Data:')
            imgs = generate_synthetic_data()
            fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10,10))
            for r in range(4): 
                for c in range(4): 
                    ax[r][c].imshow(imgs[r*4 + c])  # Indexing the images correctly
                    ax[r][c].axis('off')  # Turn off axis
            plt.tight_layout()
            st.pyplot(fig)

    # Button to trigger training
    if st.sidebar.button('Train Model'):
        st.write('Training in progress...')
        ds, modell = prepare_data(), fashganModel()
        train_model(ds=ds, modell=modell, epochs=20)  # Fill in with appropriate parameters

if __name__ == "__main__":
    main()
