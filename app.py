import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from outliers_detection import detect_and_correct_outliers

def plot_distribution(data_before, data_after, column_name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(data_before, bins=20, color='blue', alpha=0.7, label='Before Outlier Correction')
    ax[1].hist(data_after, bins=20, color='green', alpha=0.7, label='After Outlier Correction')

    ax[0].set_title(f'Distribution of {column_name} - Before Correction')
    ax[1].set_title(f'Distribution of {column_name} - After Correction')

    ax[0].legend()
    ax[1].legend()

    # Use st.pyplot directly to display the Matplotlib figure
    st.pyplot(fig)


def main():
    st.title("Streamlit App with Outlier Detection")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV file:")
        st.write(df)

        # Create a copy of the original DataFrame for visualization
        df_before_correction = df.copy()

        for column in df.columns:
            df = detect_and_correct_outliers(df, column, method='auto')

        st.write("Corrected CSV file:")
        st.write(df)

        # Display the corrected DataFrame
        st.write("Corrected Outliers:")
        st.write(df)

        # Plot distribution before and after correction
        for column in df.columns:
            plot_distribution(df_before_correction[column], df[column], column)

if __name__ == '__main__':
    main()
