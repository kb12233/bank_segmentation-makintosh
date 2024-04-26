import numpy as np
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.decomposition import PCA

# Load the model
filename = 'bank_customers_segmenter.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

st.title('Bank Customer Segmentation')

creditcard_df_scaled = pd.read_csv('creditcard_df_scaled.csv')
creditcard_df_cluster = pd.read_csv('creditcard_df_cluster.csv')

clusters = ['The Low Activity Group', 'The Premium, High-Spending Shopper', 'The Consistent Shopper', 'The High Balance, Cash Reliant']
cluster_colors = ['#00427e', '#0083dd', '#80cdfd', '#e4f6ff']

# Create five columns
col1, col2, col3, col4 = st.columns(4)

# Get user input in each column
with col1:
    balance = st.number_input('Balance', min_value=0.0, format="%.4f", step=0.0001)
    balance_frequency = st.number_input('Balance Frequency', min_value=0.0, max_value=1.0, format="%.4f", step=0.0001)
    purchases = st.number_input('Purchases', min_value=0.0, format="%.4f", step=0.0001)
    installments_purchases = st.number_input('Installments Purchases', min_value=0.0, format="%.4f", step=0.0001)

with col2:
    cash_advance = st.number_input('Cash Advance', min_value=0.0, format="%.4f", step=0.0001)
    purchases_frequency = st.number_input('Purchases Frequency', min_value=0.0, max_value=1.0, format="%.4f", step=0.0001)
    one_off_purchase_frequency = st.number_input('One-offPurchaseFrequency', min_value=0.0, max_value=1.0, format="%.4f", step=0.0001)
    cash_advance_frequency = st.number_input('Cash Advance Frequency', min_value=0.0, max_value=1.0, format="%.4f", step=0.0001)

with col3:
    cash_advance_trx = st.number_input('Cash Advance Transactions', min_value=0.0, format="%.4f", step=0.0001)
    purchases_trx = st.number_input('Purchases Transactions', min_value=0.0, format="%.4f", step=0.0001)
    credit_limit = st.number_input('Credit Limit', min_value=0.0, format="%.4f", step=0.0001)
    payments = st.number_input('Payments', min_value=0.0, format="%.4f", step=0.0001)

with col4:
    minimum_payments = st.number_input('Minimum Payments', min_value=0.0, format="%.4f", step=0.0001)
    prc_full_payment = st.number_input('Percent of Full Payment', min_value=0.0, max_value=1.0, format="%.4f", step=0.0001)
    tenure = st.number_input('Tenure', min_value=0.0, format="%.4f", step=0.0001)

# On button click, make prediction
if st.button('Predict'):
    # Use the model to predict the cluster of the new input
    input_data = np.array([[balance, balance_frequency, purchases, installments_purchases,
                            cash_advance, purchases_frequency, one_off_purchase_frequency,
                            cash_advance_frequency, cash_advance_trx, purchases_trx,
                            credit_limit, payments, minimum_payments, prc_full_payment, tenure]])

    # Ensure the input is correctly scaled using the scaler from the loaded model
    input_scaled = loaded_model['scaler'].transform(input_data)

    # Predict the cluster
    cluster_predicted = loaded_model['model'].predict(input_scaled)[0]
    st.write('Predicted customer group:', clusters[cluster_predicted])

    # Prepare new data for plotting by combining scaled data with the predicted cluster label
    new_data_scaled_with_cluster = np.append(input_scaled[0], cluster_predicted)

    # Calculate PCA for both existing and new data
    pca = PCA(n_components=2)
    creditcard_df_scaled_pca = pca.fit_transform(creditcard_df_scaled.values)

    # Map existing clusters to names
    creditcard_df_cluster['Cluster'] = creditcard_df_cluster['cluster'].apply(lambda x: clusters[x])

    # Add PCA components and cluster names to DataFrame
    creditcard_df_scaled_pca_with_cluster = np.hstack(
        [creditcard_df_scaled_pca, creditcard_df_cluster['Cluster'].values[:, None]])

    # Transform the new data point with PCA
    new_pca_data = pca.transform(input_scaled)

    # Create new data point with cluster name
    new_data_pca_with_cluster = np.append(new_pca_data[0], clusters[cluster_predicted])

    # Append new data point to existing data
    creditcard_df_scaled_pca_new = np.vstack([creditcard_df_scaled_pca_with_cluster, new_data_pca_with_cluster])

    # Create a DataFrame for plotting
    columns = ['PC1', 'PC2', 'Cluster']
    creditcard_df_cluster_new = pd.DataFrame(creditcard_df_scaled_pca_new, columns=columns)

    # Plotting using Plotly with a custom blue color theme
    fig = px.scatter(creditcard_df_cluster_new, x='PC1', y='PC2', color='Cluster', title='Bank Customer Segmentation',
                     color_discrete_sequence=cluster_colors)
    fig.add_scatter(x=[new_pca_data[0, 0]], y=[new_pca_data[0, 1]], mode='markers',
                    marker=dict(color='red', size=10, line=dict(color='orange', width=2)), name='New Data')
    st.plotly_chart(fig)