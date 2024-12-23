import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import google.generativeai as genai # type: ignore
from io import StringIO

genai.configure(api_key="AIzaSyCtD2zUEiaawhhPRrPgfiE-95mN42gOcb4")
model = genai.GenerativeModel("gemini-1.5-flash")

def show_data(df):
    """Displays the dataset, number of rows and columns"""
    st.header("Data Show:")
    st.dataframe(df.head(50), width=1000)
    st.subheader(f"Number Of Rows: {len(df)}")
    st.subheader(f"Number Of Columns: {len(df.columns)}")


def describe_data(df):
    """Generates descriptive statistics for the DataFrame."""
    st.header("Data Describe:")
    st.dataframe(df.describe(), width=1000)


def show_info(df):
    """Generates descriptive statistics for the DataFrame."""
    st.header("Data Info:")
    buffer=StringIO()
    df.info(buf=buffer)
    st.table(StringIO(buffer.getvalue()),)


def visualize_data(df, column):
    """Generates visualizations for the selected column."""
    fig, ax = plt.subplots()
    sns.histplot(df[column], ax=ax, kde=True)
    plt.title(f"Histogram of {column} with KDE")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df[column], ax=ax2)
    plt.title(f"Box Plot of {column}")
    return fig, fig2


def correlation_matrix_analysis(df):
    """Generates a correlation matrix for the DataFrame."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    if numeric_cols.empty:
        st.warning("No numeric columns found for correlation analysis.")
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, ax=ax, cmap='coolwarm')
    plt.title("Correlation Matrix (Numeric Columns)")
    return fig


def missing_value_analysis(df):
    """Displays the number of missing values per column."""
    st.header("Missing Value Analysis")
    missing_data = pd.DataFrame({
        "Missing": df.isnull().sum(),
        "Present": df.notnull().sum()
    })

    st.write("Missing and Present Values per Column:")
    st.table(missing_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom_stack = missing_data["Present"]
    ax.bar(
        missing_data.index,
        missing_data["Present"],
        label="Present",
        color="steelblue"
    )
    ax.bar(
        missing_data.index,
        missing_data["Missing"],
        bottom=bottom_stack,
        label="Missing",
        color="orange"
    )

    ax.set_title("Bar Chart for Missing Values", fontsize=14)
    ax.legend(loc="upper right")
    st.pyplot(fig)

def handle_missing_values(df, method, column):
    """Handles missing values based on the selected method and column."""
    change_button = st.button("Make Changes", key='handle_missing_btn')
    before_button = st.button("Before Changes", key='show_handle_missing_btn')
    if before_button:
        try:
            df_temp=df.copy()
            st.write(f"The Number of Missing Values of {column} is {df_temp[column].isna().sum()}")
            if method == "mean":
                df_temp[column].fillna(df_temp[column].mean(), inplace=True)
            elif method == "median":
                df_temp[column].fillna(df_temp[column].median(), inplace=True)
            elif method == "mode":
                df_temp[column].fillna(df_temp[column].mode()[0], inplace=True)
            elif method == "drop":
                st.write(f"The Number of Rows Of  {column} Column Is  {len(df_temp[column])}")
                df_temp.dropna(subset=[column], inplace=True)
                st.write(f"The Number of Rows Of  {column} Column After Dropping Missing Values Is  {len(df_temp[column])}")
            else:
                st.error("Invalid method for handling missing values.")
            st.write(f"The Number of Missing Values of {column} will be {df_temp[column].isna().sum()}")
        except Exception as e:
            st.error(f"Error handling missing values: {e}")

    if change_button:
        try:
            if method == "mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif method == "median":
                df[column].fillna(df[column].median(), inplace=True)
            elif method == "mode":
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif method == "drop":
                df.dropna(subset=[column], inplace=True)
            else:
                st.error("Invalid method for handling missing values.")
            st.success(f"Missing Values for '{column}'handled with '{method}' Successfully!")
        except Exception as e:
            st.error(f"Error handling missing values: {e}")
        st.session_state['data'] = df
    return df


def handle_duplicates(df):
    """Handles duplicate rows in the DataFrame."""
    num_duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {num_duplicates}")
    if num_duplicates > 0:
        show_duplicates = st.checkbox("Show Duplicate Rows", key='show_duplicates')
        if show_duplicates:
            st.write(df[df.duplicated(keep=False)])
        change_button=st.button("Make Changes", key='remove_duplicates')
        before_button=st.button("Before Changes", key='show_remove_duplicates')
        if before_button:
            df_temp=df.copy()
            st.write(f"The Number of Rows Of DataSet Is {len(df_temp)}")
            df_temp.drop_duplicates(inplace=True)
            st.write(f"The Number of Rows Of DataSet  After Removing Duplicates Will be   {len(df_temp)}")
        if change_button:
            df.drop_duplicates(inplace=True)
            st.session_state['data'] = df
            st.success("Duplicate rows removed.")
    return df


def outlier_analysis(df, column):
    """Identifies and displays outliers using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    st.write(f"Number of outliers in {column}: {len(outliers)}")
    if not outliers.empty:
        st.write(outliers)
        show_outliers_vis = st.checkbox("Show outliers visualization", key='show_outliers_vis')
        if show_outliers_vis:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column], ax=ax)
            sns.scatterplot(x=outliers[column], y=[0]*len(outliers), color='red', marker='o', ax=ax)
            plt.title(f"Box Plot of {column} with Outliers highlighted")
            st.pyplot(fig)
    return lower_bound, upper_bound


def handle_outliers(df, column, lower_bound, upper_bound, method='clip'):
    """Handles outliers based on the selected method."""
    before_button = st.button("Before Changes")
    change_button = st.button("Make Changes")
    if before_button:
        df_temp = df.copy()
        if method == 'clip':
            df_temp[column] = df_temp[column].clip(lower=lower_bound, upper=upper_bound)
            st.success(f"Outliers in {column} will be clipped to the defined bounds.")
        elif method == 'drop':
            df_temp = df_temp.drop(df_temp[(df_temp[column] < lower_bound) | (df_temp[column] > upper_bound)].index, axis=0)
            st.success(f"Outliers in {column} will be removed.")
        else:
            st.error("Invalid method for handling outliers.")

    if change_button:
        if method == 'clip':
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            st.success(f"Outliers in {column} have been clipped to the defined bounds.")
        elif method == 'drop':
            df = df.drop(df[(df[column] < lower_bound) | (df[column] > upper_bound)].index, axis=0)
            st.success(f"Outliers in {column} have been removed.")
        else:
            st.error("Invalid method for handling outliers.")
        st.session_state['data'] = df
    return df


def data_types_analysis(df):
    """Displays data type information and allows conversion."""
    st.header("Data Types Analysis")
    st.dataframe(df.dtypes, width=1000)
    st.subheader("Convert Data Types:")
    
    selected_column = st.selectbox("Select a column to convert", df.columns, key="convert_col")
    new_type = st.selectbox("Select the new data type", ["int", "float", "datetime", "str"], key="new_type")
    change_button = st.button("Make Changes", key='convert_btn')
    before_button = st.button("Before Changes")
    if before_button:
        try:
            df_temp = df.copy()
            if new_type == "datetime":
                df_temp[selected_column] = pd.to_datetime(df_temp[selected_column],errors='coerce')
            else:
                if new_type == "str":
                    nans = df_temp[selected_column].isna().sum()
                    if nans > 0:
                        st.warning(f':red[Note:] {nans} NaN values will be converted to a string')
                df_temp[selected_column] = df_temp[selected_column].astype(new_type)
                st.success(f"Column '{selected_column}' can convert to {new_type} successfully!")
        except Exception as e:
            st.error(f"There will be an error converting column '{selected_column}': {e}")
    if change_button:
        try:
            if new_type == "datetime":
                df[selected_column] = pd.to_datetime(df[selected_column],errors='coerce')
            else:
                df[selected_column] = df[selected_column].astype(new_type)
            st.session_state['data'] = df
            st.success(f"Column '{selected_column}' converted to {new_type} successfully!")
            st.table({selected_column: df[selected_column].dtype})
        except Exception as e:
            st.error(f"Error converting column '{selected_column}': {e}")
    return df


def show_unique_values(df):
    st.header("Unique Values for A Column")
    column = st.selectbox("Select a column to view unique values:", df.columns)
    st.subheader(f"Unique Values for Column: {column}")
    unique_values = df[column].unique()
    st.write(f"Number of unique values: {len(unique_values)}")
    if len(unique_values) <= 20:
        st.table(unique_values)
    else: 
        st.write("Too many unique values to display, showing the first 20:")
        st.table(unique_values[:20])


def column_names_analysis(df):
    """Displays column name information and allows renaming of columns."""
    st.header("Column Name Analysis")
    st.subheader("Current Column Names:")
    st.write(df.columns)
    rename_cols = st.checkbox("Rename columns", key='rename_columns_checkbox')
    if rename_cols:
        st.subheader("Rename Columns:")
        new_column_names = {}
        for col in df.columns:
            new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
            new_column_names[col] = new_name
        rename_button = st.button("Apply Column Renaming", key='rename_btn')
        if rename_button:
            try:    
                df = df.rename(columns=new_column_names)
                st.session_state['data'] = df
                st.success("Columns renamed successfully!")
                st.subheader("Updated Column Names:")
                st.write(df.columns)
            except Exception as e:
                st.error(f"Error renaming columns: {e}")
    return df


def rag_query(dataframe, question):
    # Convert the dataframe to a string for context
    context = f"The first few rows of the dataset are:\n{dataframe.head(500).to_string()}\n"

    # Combine the question with the context
    prompt = f"{context}\nQuestion: {question}\nAnswer:"

    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Error: {e}"
    

def rag_interface(df):
    """Creates an interface for users to ask questions about the dataset."""
    st.header("Ask Questions about the Dataset")
    prompt = st.chat_input("Ask Something")
    
    if prompt:
        with st.chat_message("human"):
            st.write(prompt)
        # Query the model and get the answer
        answer = rag_query(df, prompt)
        with st.chat_message("ai"):
            st.write(answer)


def download_dataset(df):
    """Downloads the DataFrame as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="downloaded_data.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)


def reset_all():
    st.session_state['data'] = None
    st.session_state['page'] = None
    st.session_state['file_hash'] = None
    st.cache_data.clear()
    st.cache_resource.clear()