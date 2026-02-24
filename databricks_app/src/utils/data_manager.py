"""
Data Manager for Time Series Storage

Handles saving, loading, and managing time series datasets across pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime


def initialize_data_storage():
    """Initialize session state for data storage if not already present."""
    if 'saved_datasets' not in st.session_state:
        st.session_state.saved_datasets = {}


def save_dataset(
    name: str,
    data: pd.DataFrame,
    dataset_type: str,
    metadata: Optional[Dict] = None
):
    """
    Save a dataset to session state.

    Parameters
    ----------
    name : str
        Name/identifier for the dataset
    data : pd.DataFrame
        Time series data to save
    dataset_type : str
        Type of dataset ('univariate', 'multivariate')
    metadata : dict, optional
        Additional metadata about the dataset
    """
    initialize_data_storage()

    if metadata is None:
        metadata = {}

    st.session_state.saved_datasets[name] = {
        'data': data.copy(),
        'type': dataset_type,
        'metadata': metadata,
        'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def get_dataset(name: str) -> Optional[Dict]:
    """
    Retrieve a saved dataset.

    Parameters
    ----------
    name : str
        Name of the dataset

    Returns
    -------
    dict or None
        Dataset dictionary or None if not found
    """
    initialize_data_storage()
    return st.session_state.saved_datasets.get(name)


def list_datasets(dataset_type: Optional[str] = None) -> List[str]:
    """
    List all saved datasets.

    Parameters
    ----------
    dataset_type : str, optional
        Filter by dataset type ('univariate', 'multivariate')

    Returns
    -------
    list
        List of dataset names
    """
    initialize_data_storage()

    if dataset_type is None:
        return list(st.session_state.saved_datasets.keys())
    else:
        return [
            name for name, info in st.session_state.saved_datasets.items()
            if info['type'] == dataset_type
        ]


def delete_dataset(name: str):
    """Delete a saved dataset."""
    initialize_data_storage()
    if name in st.session_state.saved_datasets:
        del st.session_state.saved_datasets[name]


def add_missing_values(
    data: pd.DataFrame,
    missing_pct: float = 0.1,
    block_size: int = 5,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Add random blocks of missing values to a time series.

    Parameters
    ----------
    data : pd.DataFrame
        Original time series data
    missing_pct : float
        Percentage of data to make missing (0.0 to 1.0)
    block_size : int
        Average size of missing blocks
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Data with missing values added
    """
    if seed is not None:
        np.random.seed(seed)

    data_with_missing = data.copy()
    n_samples = len(data)
    n_missing = int(n_samples * missing_pct)

    if n_missing == 0:
        return data_with_missing

    # Generate random block starts
    n_blocks = max(1, n_missing // block_size)
    block_starts = np.random.choice(n_samples - block_size, size=n_blocks, replace=False)

    # Create blocks of missing values
    for start in block_starts:
        actual_block_size = min(block_size, n_samples - start)
        # Make sure we don't exceed the total missing percentage
        if np.isnan(data_with_missing.values).sum() >= n_missing:
            break
        data_with_missing.iloc[start:start + actual_block_size] = np.nan

    return data_with_missing


def display_dataset_info(dataset_info: Dict):
    """
    Display information about a saved dataset.

    Parameters
    ----------
    dataset_info : dict
        Dataset information dictionary
    """
    st.write(f"**Type**: {dataset_info['type']}")
    st.write(f"**Saved at**: {dataset_info['saved_at']}")
    st.write(f"**Shape**: {dataset_info['data'].shape}")

    if dataset_info['metadata']:
        st.write("**Metadata**:")
        for key, value in dataset_info['metadata'].items():
            st.write(f"  - {key}: {value}")


def create_save_widget(
    data: pd.DataFrame,
    dataset_type: str,
    default_name: str = "",
    metadata: Optional[Dict] = None
):
    """
    Create a widget for saving datasets.

    Parameters
    ----------
    data : pd.DataFrame
        Data to save
    dataset_type : str
        Type of dataset
    default_name : str
        Default dataset name
    metadata : dict, optional
        Metadata to save with dataset
    """
    st.subheader("💾 Save Dataset for Estimation")

    col1, col2 = st.columns([3, 1])

    with col1:
        dataset_name = st.text_input(
            "Dataset Name",
            value=default_name,
            help="Name to identify this dataset"
        )

    with col2:
        if st.button("Save Dataset", type="primary"):
            if dataset_name:
                save_dataset(dataset_name, data, dataset_type, metadata)
                st.success(f"✓ Dataset '{dataset_name}' saved!")
            else:
                st.error("Please enter a dataset name")

    # Download as CSV
    csv = data.to_csv()
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"{dataset_name if dataset_name else 'data'}.csv",
        mime="text/csv"
    )


def create_load_widget(dataset_type: str) -> Optional[pd.DataFrame]:
    """
    Create a widget for loading datasets.

    Parameters
    ----------
    dataset_type : str
        Type of dataset to filter ('univariate', 'multivariate')

    Returns
    -------
    pd.DataFrame or None
        Loaded dataset or None
    """
    st.subheader("📂 Load Dataset")

    available_datasets = list_datasets(dataset_type)

    if not available_datasets:
        st.info(f"No saved {dataset_type} datasets found. Generate data from an exploration page first.")
        return None

    selected_dataset = st.selectbox(
        "Select Dataset",
        available_datasets,
        help="Choose a saved dataset to load"
    )

    if selected_dataset:
        dataset_info = get_dataset(selected_dataset)

        with st.expander("Dataset Info"):
            display_dataset_info(dataset_info)

        if st.button("Load Dataset"):
            st.success(f"✓ Loaded dataset '{selected_dataset}'")
            return dataset_info['data']

    return None


def create_missing_value_widget(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a widget for adding missing values to data.

    Parameters
    ----------
    data : pd.DataFrame
        Original data

    Returns
    -------
    pd.DataFrame
        Data with missing values (if requested)
    """
    st.subheader("🔲 Add Missing Values (Optional)")

    add_missing = st.checkbox("Add random blocks of missing values", value=False)

    if add_missing:
        col1, col2, col3 = st.columns(3)

        with col1:
            missing_pct = st.slider(
                "Missing %",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                help="Percentage of data to make missing"
            ) / 100

        with col2:
            block_size = st.number_input(
                "Block Size",
                min_value=1,
                max_value=20,
                value=5,
                help="Average size of missing blocks"
            )

        with col3:
            seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=10000,
                value=42
            )

        data_with_missing = add_missing_values(data, missing_pct, block_size, seed)

        # Show missing value statistics
        n_missing = data_with_missing.isna().sum().sum()
        total_values = data_with_missing.size
        actual_pct = (n_missing / total_values) * 100

        st.info(f"Added {n_missing} missing values ({actual_pct:.1f}% of data)")

        return data_with_missing

    return data
