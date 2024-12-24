# Data Analysis Streamlit App

This is an interactive data analysis application built with Streamlit. The app allows users to load, clean, visualize, and analyze datasets with various built-in functions, including handling missing values, duplicates, outliers, and more. It also includes the ability to ask questions about the dataset using a generative AI model (Google Gemini).

## Features

- **Dataset Display**: View the first few rows of the dataset and see basic information (number of rows, columns).
- **Data Statistics**: Generate descriptive statistics for the dataset.
- **Missing Value Analysis**: Identify and handle missing values with various methods like mean, median, mode, or dropping.
- **Duplicate Handling**: Identify and remove duplicate rows.
- **Outlier Detection**: Detect outliers using the IQR method and handle them by either clipping or removal.
- **Column Renaming**: Rename dataset columns.
- **Data Type Conversion**: Convert data types of columns (e.g., to datetime, int, float, or string).
- **Visualizations**: Generate histograms, box plots, and correlation matrices for numeric columns.
- **Question Answering**: Use a generative AI model to answer questions about the dataset.
- **Download**: Download the modified dataset as a CSV file.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Google Generative AI SDK (`google.generativeai`)

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

To run the app use:

```bash
pip install -r requirements.txt
```
