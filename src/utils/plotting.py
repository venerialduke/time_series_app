"""
Plotting utilities for time series visualization.

Leverages plotly, matplotlib, and statsmodels plotting functions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st


def plot_time_series(series: pd.Series, title: str = "Time Series", shock_info: dict = None) -> go.Figure:
    """
    Create an interactive time series plot using plotly.

    Parameters
    ----------
    series : pd.Series
        Time series data
    title : str
        Plot title
    shock_info : dict, optional
        Information about shocks to visualize. Should contain 'time', 'magnitude', 'date' keys.

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=series.index if hasattr(series.index, '__iter__') else list(range(len(series))),
        y=series.values,
        mode='lines',
        name='Value',
        line=dict(color='#1f77b4', width=1.5)
    ))

    # Add shock marker if provided
    if shock_info is not None:
        shock_idx = shock_info['time']
        shock_date = series.index[shock_idx]
        shock_value = series.iloc[shock_idx]

        # Add vertical line at shock time
        fig.add_vline(
            x=shock_date,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text=f"Shock: {shock_info['magnitude']:.1f}σ",
            annotation_position="top"
        )

        # Add marker at shock point
        fig.add_trace(go.Scatter(
            x=[shock_date],
            y=[shock_value],
            mode='markers',
            name='Shock',
            marker=dict(
                size=12,
                color='red',
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate=f'<b>Shock Applied</b><br>Observation: {shock_idx}<br>Magnitude: {shock_info["magnitude"]:.2f}σ<br>Value: {shock_value:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 40) -> tuple:
    """
    Create ACF and PACF plots using plotly.

    Parameters
    ----------
    series : pd.Series
        Time series data
    lags : int
        Number of lags to display

    Returns
    -------
    tuple
        (acf_fig, pacf_fig) plotly figure objects
    """
    from statsmodels.tsa.stattools import acf, pacf

    # Calculate ACF values
    acf_values = acf(series, nlags=lags, fft=False)

    # Calculate PACF values
    pacf_values = pacf(series, nlags=lags, method='ywm')

    # Confidence interval (95%)
    conf_int = 1.96 / np.sqrt(len(series))

    # Create ACF plot
    acf_fig = go.Figure()

    # Add bars for ACF
    acf_fig.add_trace(go.Bar(
        x=list(range(len(acf_values))),
        y=acf_values,
        name='ACF',
        marker_color='#1f77b4'
    ))

    # Add confidence interval lines
    acf_fig.add_hline(y=conf_int, line_dash="dash", line_color="red", opacity=0.5)
    acf_fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", opacity=0.5)
    acf_fig.add_hline(y=0, line_color="black", line_width=1)

    acf_fig.update_layout(
        title='Autocorrelation Function (ACF)',
        xaxis_title='Lag',
        yaxis_title='ACF',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    # Create PACF plot
    pacf_fig = go.Figure()

    # Add bars for PACF
    pacf_fig.add_trace(go.Bar(
        x=list(range(len(pacf_values))),
        y=pacf_values,
        name='PACF',
        marker_color='#ff7f0e'
    ))

    # Add confidence interval lines
    pacf_fig.add_hline(y=conf_int, line_dash="dash", line_color="red", opacity=0.5)
    pacf_fig.add_hline(y=-conf_int, line_dash="dash", line_color="red", opacity=0.5)
    pacf_fig.add_hline(y=0, line_color="black", line_width=1)

    pacf_fig.update_layout(
        title='Partial Autocorrelation Function (PACF)',
        xaxis_title='Lag',
        yaxis_title='PACF',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return acf_fig, pacf_fig


def plot_distribution(series: pd.Series) -> go.Figure:
    """
    Create a histogram with KDE overlay for the distribution.

    Parameters
    ----------
    series : pd.Series
        Time series data

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=series.values,
        name='Distribution',
        nbinsx=30,
        marker_color='#1f77b4',
        opacity=0.7
    ))

    fig.update_layout(
        title='Distribution of Values',
        xaxis_title='Value',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def plot_summary_statistics(series: pd.Series):
    """
    Display summary statistics in a formatted way.

    Parameters
    ----------
    series : pd.Series
        Time series data
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{series.mean():.4f}")
    with col2:
        st.metric("Std Dev", f"{series.std():.4f}")
    with col3:
        st.metric("Min", f"{series.min():.4f}")
    with col4:
        st.metric("Max", f"{series.max():.4f}")


def plot_correlogram_comparison(series: pd.Series, lags: int = 40):
    """
    Create side-by-side ACF and PACF plots using Plotly.

    Parameters
    ----------
    series : pd.Series
        Time series data
    lags : int
        Number of lags to display
    """
    acf_fig, pacf_fig = plot_acf_pacf(series, lags=lags)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(acf_fig, width='stretch')
    with col2:
        st.plotly_chart(pacf_fig, width='stretch')
