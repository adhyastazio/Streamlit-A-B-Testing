# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:57:55 2024

@author: A.Wardhanawan
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind


df = pd.read_excel('Flag_Pilot_Grouped.xlsx')

df = df[df['FLAG_PILOT'] != 'Inside POS']
df['CODE_RISK'] = df['CODE_RISK'].str.strip()
df_filtered = df[df['CODE_RISK'].isin(['A+', 'B', 'C'])]


st.sidebar.header('Data Filter')

all_flag_pilot = ['All'] + df_filtered['FLAG_PILOT'].unique().tolist()
selected_flag_pilot = st.sidebar.selectbox('Flag Pilot', all_flag_pilot)

all_group_date_pilot = ['All'] + df_filtered['GROUP_DATE_PILOT'].unique().tolist()
selected_group_date_pilot = st.sidebar.selectbox('Pilot Group', all_group_date_pilot)

all_code_risk = ['All'] + df_filtered['CODE_RISK'].unique().tolist()
selected_code_risk = st.sidebar.multiselect('Code Risk', all_code_risk, default=['All'])
start_date = st.sidebar.date_input('Start Date', df_filtered['DATE_0BOD'].min())
end_date = st.sidebar.date_input('End Date', df_filtered['DATE_0BOD'].max())

# Apply date filter
day_df = df_filtered[(df_filtered['DATE_0BOD'].dt.date >= start_date) & (df_filtered['DATE_0BOD'].dt.date <= end_date)]

if selected_flag_pilot == 'All':
    day_df_0 = day_df
else:
    day_df_0 = day_df[day_df['FLAG_PILOT'] == selected_flag_pilot]

if selected_group_date_pilot == 'All':
    day_df_1 = day_df_0
else:
    day_df_1 = day_df_0[day_df_0['GROUP_DATE_PILOT'] == selected_group_date_pilot]

if 'All' in selected_code_risk:
    day_df_2 = day_df_1
else:
    day_df_2 = day_df_1[day_df_1['CODE_RISK'].isin(selected_code_risk)]

day_df_3 = day_df_2.copy()
day_df_3['DATE_0BOD'] = day_df_2['DATE_0BOD'].dt.strftime('%m/%d')

# Streamlit App
def main():
    st.title('Hide Limit Pilot A/B Testing')

    st.subheader('Filtered Data')
    st.dataframe(day_df_3)

    show_labels = st.checkbox('Data Labels', value=True)

    st.subheader('Limit Approved Summary')
    plot_risk_vs_limit(day_df_2, show_labels)

    st.subheader('Limit to Initiate Mitra by Flag Pilot')
    plot_visit_pos_limit_percentage(day_df_2, show_labels)
    
    st.subheader('Sum of Initiate Mitra and Limit Approved')
    mitra_limit(day_df_2, show_labels)
    
    st.subheader('Breakdown')
    initiate_limit(day_df_2, show_labels)
    steps_breakdown(day_df_2, show_labels)
    flag_pilot(day_df_2, show_labels)
    
    st.subheader('Statistics')
    ttest_pilot(day_df_2)

    


def plot_risk_vs_limit(df_filtered, show_labels=True):
    df_pivot = df_filtered.pivot_table(index='DATE_0BOD', columns='CODE_RISK', values='LIMIT_APPROVED', aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_pivot.plot(kind='bar', stacked=True, ax=ax)

    tick1 = range(len(df_filtered['DATE_0BOD'].unique()))
    tick2 = df_filtered['DATE_0BOD'].dt.strftime('%m/%d').unique()

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if show_labels:
            annotation_x = x + width / 2 - 0.05
            annotation_y = y + height / 2 + 3
            ax.annotate('{:.0f}'.format(height), (annotation_x, annotation_y), ha='center', fontsize=8)
    
    plt.xticks(ticks=tick1, labels=tick2, rotation=45, ha='right')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.xlabel('Date')
    plt.ylabel('Limit Approved')
    st.pyplot(fig)

def plot_visit_pos_limit_percentage(df_filtered, show_labels=True):
    grouped_data = df_filtered.groupby('FLAG_PILOT')
    sum_data = grouped_data[['CUSTOMER_INFO_PAGE', 'LIMIT_APPROVED']].sum()
    sum_data['Visit POS/Limit'] = sum_data['CUSTOMER_INFO_PAGE'] / sum_data['LIMIT_APPROVED'] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=sum_data.reset_index(), x='FLAG_PILOT', y='Visit POS/Limit', ax=ax)
    
    for p in ax.patches:
        if show_labels:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points') 
    
    ax.set_ylabel('Limit to Initiate Mitra (%)')
    st.pyplot(fig)

def mitra_limit(df_filtered, show_labels=True):
    grouped_data = df_filtered.groupby('FLAG_PILOT')
    sum_data = grouped_data[['CUSTOMER_INFO_PAGE', 'LIMIT_APPROVED']].sum()
    sum_df = sum_data.reset_index()
    
    fig, ax = plt.subplots()
    index = range(len(sum_df))
    bar_width = 0.35
    
    bar1 = ax.bar(index, sum_df['LIMIT_APPROVED'], bar_width, label='Limit Approved')
    bar2 = ax.bar([i + bar_width for i in index], sum_df['CUSTOMER_INFO_PAGE'], bar_width, label='Initiate Mitra')
    
    for i, (value1, value2) in enumerate(zip(sum_df['LIMIT_APPROVED'], sum_df['CUSTOMER_INFO_PAGE'])):
        if show_labels:
            ax.text(i, value1 + 5, '{:.0f}'.format(value1), ha='center', va='bottom') 
            ax.text(i + bar_width, value2 + 5, '{:.0f}'.format(value2), ha='center', va='bottom') 
        
    ax.set_xlabel('FLAG_PILOT')
    ax.set_ylabel('Number of Events')
    ax.set_title('Sum of Initiate Mitra and Limit Approved')
    ax.set_xticks([i + bar_width/2 for i in index])
    ax.set_xticklabels(sum_df['FLAG_PILOT'])
    ax.legend()
 
    st.pyplot(fig)
    
def initiate_limit(df_filtered, show_labels=True):
    df1 = df_filtered.groupby(['FLAG_PILOT', df_filtered['DATE_0BOD'].dt.date])
    df2 = df1[['CUSTOMER_INFO_PAGE', 'LIMIT_APPROVED']].sum()
    df2['ratio'] = (df2['CUSTOMER_INFO_PAGE'] / df2['LIMIT_APPROVED']) * 100

    df2.reset_index(inplace=True)
    df2['DATE_0BOD'] = pd.to_datetime(df2['DATE_0BOD'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df2, x='DATE_0BOD', y='ratio', hue='FLAG_PILOT', errorbar=None, ax=ax)

    tick_positions = range(len(df2['DATE_0BOD'].unique()))
    tick_labels = df2['DATE_0BOD'].dt.strftime('%m/%d').unique()

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    for p in ax.patches:
        height = p.get_height()
        if height > 0 and show_labels:
            ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=7) 

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.ylabel('Ratio')
    plt.title('Ratio by Flag Pilot and Date')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    st.pyplot(fig)

def steps_breakdown(df_filtered, show_labels=True):
    df1 = df_filtered.groupby(['FLAG_PILOT', df['DATE_0BOD'].dt.date])
    df2 = df1[['CUSTOMER_INFO_PAGE', 'LIMIT_APPROVED']].sum()
    df2['ratio'] = (df2['CUSTOMER_INFO_PAGE'] / df2['LIMIT_APPROVED']) * 100

    df2.reset_index(inplace=True)
    df2['DATE_0BOD'] = pd.to_datetime(df2['DATE_0BOD'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(data=df2, x="DATE_0BOD", y="CUSTOMER_INFO_PAGE", hue="FLAG_PILOT", palette="viridis", alpha=0.7, ax=ax1)

    sns.barplot(data=df2, x="DATE_0BOD", y="LIMIT_APPROVED", hue="FLAG_PILOT", alpha=0.7, ax=ax2)
    
    for ax in [ax1, ax2]:
        for p in ax.patches:
            if show_labels:
                height = p.get_height()
                ax.annotate(f"{height:.0f}", (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="center", xytext=(0, 10), textcoords="offset points", fontsize=7) 
    
    tick_positions = range(len(df2["DATE_0BOD"].unique()))
    tick_labels = df2["DATE_0BOD"].dt.strftime("%m/%d").unique()
    
    for ax in [ax1, ax2]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    
    ax1.legend(title="Initiate Mitra", loc="upper left", bbox_to_anchor=(1, 1))
    ax2.legend(title="Limit Approved", loc="upper left", bbox_to_anchor=(1, 1))
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Number of Events')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Events')
    
    ax1.set_title('Initiate Mitra')
    ax2.set_title('Limit Approved')
    #plt.suptitle("CUSTOMER_INFO_PAGE and LIMIT_APPROVED by Flag Pilot and Date", y=1.05)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    st.pyplot(fig)

def flag_pilot(df_filtered, show_labels=True):
    grouped_data = df_filtered.groupby('FLAG_PILOT')
    
    df_grouped = grouped_data[['CUSTOMER_INFO_PAGE', 'LIMIT_APPROVED', 'SUBMIT_2BOD', 'SIGNED_CONTRACT']].sum()
    df_grouped['Visit POS/Limit'] = (df_grouped['CUSTOMER_INFO_PAGE'] / df_grouped['LIMIT_APPROVED'] * 100).round(2)
    df_grouped['AOL'] = (df_grouped['SUBMIT_2BOD'] / df_grouped['LIMIT_APPROVED'] * 100).round(2)
    df_grouped['Contract/Limit'] = (df_grouped['SIGNED_CONTRACT'] / df_grouped['LIMIT_APPROVED'] * 100).round(2)
    
    # Resetting the index to make 'FLAG_PILOT' a regular column
    df_grouped.reset_index(inplace=True)
    
    # Selecting relevant columns including 'FLAG_PILOT'
    dfx = df_grouped[['FLAG_PILOT', 'Visit POS/Limit', 'AOL', 'Contract/Limit']]
    
    # Using melt with 'FLAG_PILOT' as id_vars
    df_melt = pd.melt(dfx, id_vars='FLAG_PILOT', var_name='Metrics', value_name='Values')
    
    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Metrics', y='Values', hue='FLAG_PILOT', data=df_melt)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0 and show_labels:
            ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=7)

    
    plt.title('Comparison of Metrics between Control and Hide Limit Pilot')
    plt.ylabel('Values')

    st.pyplot(fig)



def ttest_pilot(df_filtered):
    df1 = df_filtered.groupby(['FLAG_PILOT', df_filtered['DATE_0BOD'].dt.date])
    df2 = df1[['CUSTOMER_INFO_PAGE', 'LIMIT_APPROVED']].sum()
    df2['ratio'] = (df2['CUSTOMER_INFO_PAGE'] / df2['LIMIT_APPROVED']) * 100

    df2.reset_index(inplace=True)
    control_group = df2[df2['FLAG_PILOT'] == 'Control']['ratio']
    pilot_group = df2[df2['FLAG_PILOT'] == 'Hide Limit Pilot']['ratio']
    
    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=['Control', 'Hide Limit Pilot'], y=[control_group.mean(), pilot_group.mean()], ax=ax)
    ax.set_ylabel('Average Ratio')
    #   ax.set_title('Average Ratio Comparison between Control and Hide Limit Pilot')
    
    # Adding error bars for standard error of the mean
    plt.errorbar(x=['Control', 'Hide Limit Pilot'], y=[control_group.mean(), pilot_group.mean()],
                 yerr=[control_group.sem(), pilot_group.sem()], fmt='o', color='black', capsize=5)
    
    # Performing t-test
    t_statistic, p_value = ttest_ind(control_group, pilot_group)
    alpha = 0.05
    p_value_r = round(p_value, 3)
    t_statistic_r = round(t_statistic, 3)
    
    if p_value < alpha:
        plt.text(0.5, max(control_group.mean(), pilot_group.mean()) + 5, f'T-statistic: {t_statistic_r}\nP-value: {p_value_r}\nReject the null hypothesis. There is a significant difference between the group.', ha='center', va='bottom')
    else:
        plt.text(0.5, max(control_group.mean(), pilot_group.mean()) + 5, f'T-statistic: {t_statistic_r}\nP-value: {p_value_r}\nFail to reject the null hypothesis. There is no significant difference between the group.', ha='center', va='bottom')
    
    st.pyplot(fig)

if __name__ == '__main__':
    main()
    