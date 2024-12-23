import streamlit as st
import pandas as pd
import hashlib
from io import StringIO
from methods import *

def file_hashing(file):
    hash_func = hashlib.sha256()
    for chunk in iter(lambda: file.read(4096), b""):
        hash_func.update(chunk)
    file.seek(0)
    return hash_func.hexdigest()


def is_same_data(new_data_hash):
    return 'data' in st.session_state and 'file_hash' in st.session_state and st.session_state.file_hash == new_data_hash


def main():
    st.sidebar.title(":blue[Data Quality Analysis]")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset",on_change=func, type=["csv", "xlsx"], key='file_uploader')

    if uploaded_file is not None:
        hashed_file = file_hashing(uploaded_file)
        try:
            if is_same_data(hashed_file):
                global df
                df = st.session_state['data'].copy()
            else:
                if uploaded_file.name.endswith(".csv"):
                    csv_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    data = pd.read_csv(csv_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    data = pd.read_excel(uploaded_file)

                st.session_state['original_data'] = data.copy()
                st.session_state['data'] = data
                st.session_state['file_hash'] = hashed_file
                st.sidebar.success("Dataset uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            
    else:
        st.sidebar.warning("You have to upload a dataset")
        reset_all()


def navigations():
    st.sidebar.subheader(":red[What you want to do with data:]")
    for tab in tabs:
        st.sidebar.button(tab, on_click=callable, args=[tab])
        
    # Render the selected page
    if 'page' in st.session_state and st.session_state['page'] is not None:
        tabs[st.session_state['page']]()


def show_page():
    show_data(df)


def info_page():
    show_info(df)


def describe_page():
    describe_data(df)


def data_types():
    global df
    df = data_types_analysis(df)


def show_uniques():
    show_unique_values(df)


def column_names():
    global df
    df = column_names_analysis(df)


def missing_values():
    missing_value_analysis(df)


def missing_handling():
    method = st.selectbox("Select Method", ["mean", "median", "mode", "drop"], key="missing_method")
    column = st.selectbox("Select Column", df.columns, key="missing_col")
    handle_missing_values(df, method, column)


def dublicates_handling():
    global df
    df = handle_duplicates(df)


def outliers():
    global df
    column_for_outlier = st.selectbox("Select Column for Outlier Analysis", df.select_dtypes(include=['float64', 'int64']).columns)
    lower_bound, upper_bound = outlier_analysis(df, column_for_outlier)
    if lower_bound is not None and upper_bound is not None:
        outlier_method = st.selectbox("Select Outlier Handling Method", ['clip', 'drop'])
        df = handle_outliers(df, column_for_outlier, lower_bound, upper_bound, outlier_method)


def data_visualizing():
    st.header("Data Visualization")
    column_to_visualize = st.selectbox("Select Column for Visualization", df.columns, key="visualize_col")
    fig1, fig2 = visualize_data(df, column_to_visualize)
    st.pyplot(fig1)
    st.pyplot(fig2)


def correlation_matrix():
    st.header("Correlation Matrix")
    fig = correlation_matrix_analysis(df)
    if fig is not None:
        st.pyplot(fig)


def dataset_downloading():
    download_dataset(df)


def question_asking():
    rag_interface(df)


def discard_changes():
    global df
    st.warning("Are you sure you want to discard the changes you made?")
    yes = st.button("yes")
    no = st.button("no")
    if yes: 
        df = st.session_state['data'] = st.session_state['original_data'].copy()
        st.success("Changes discarded successfully!")
    elif no:
        st.success("Keep changes.")


tabs = {
    "Chatbot" : question_asking,
    "Show Data": show_page,
    "Describe Data": describe_page,
    "Show Info" : info_page,
    "Show Uniques" : show_uniques,
    "Data Types Analysis" : data_types,
    "Column Names Analysis" : column_names,
    "Missing Values Analysis" : missing_values,
    "Handling Missing Values" : missing_handling,
    "Handling Duplicates" :  dublicates_handling,
    "Outliers Analysis" : outliers,
    "Discard Changes" : discard_changes,
    "Correlation Matrix" : correlation_matrix,
    "Data Visualize": data_visualizing,
    "Download dataset" : dataset_downloading,
}

#A callback function for the buttons' trigger
def callable(*args):
    st.session_state['page'] = args[0]

def func():
    reset_all()

#entry point
if __name__ == "__main__":
    main()

if st.session_state['data'] is not None:
    df = st.session_state['data'].copy()
    navigations()
