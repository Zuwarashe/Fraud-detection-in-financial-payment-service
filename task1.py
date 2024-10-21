import pandas as pd
import matplotlib.pyplot as plt

def exercise_0(file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file)
    return df

def exercise_1(df):
    # Return the column names as a list
    column_names = df.columns.tolist()
    return column_names

def exercise_2(df, k):
    # Return the first k rows from the DataFrame
    first_k_rows = df.head(k)
    return first_k_rows

def exercise_3(df, k):
    # Return a random sample of k rows from the DataFrame
    random_sample = df.sample(n=k)
    return random_sample
def exercise_4(df):
    
    # Return a list of the unique transaction types.
    return df['type'].unique().tolist()

def exercise_5(df):
   # Return a Pandas series of the top 10 transaction destinations with frequencies.
    return df['nameDest'].value_counts().head(10)

def exercise_6(df):
   # Return all the rows from the dataframe for which fraud was detected.
    return df[df['isFraud'] == 1]

def visual_1(df):
    def transaction_counts(df):
        # Count of each transaction type
        return df['type'].value_counts()
    
    def transaction_counts_split_by_fraud(df):
        # Count of each transaction type split by fraud
        return df.groupby(['type', 'isFraud']).size().unstack(fill_value=0)
    
    # Create subplots
    fig, axs = plt.subplots(2, figsize=(10, 12))
    
    # Plot transaction types count
    transaction_counts(df).plot(ax=axs[0], kind='bar')
    axs[0].set_title('Transaction Types Count')
    axs[0].set_xlabel('Transaction Type')
    axs[0].set_ylabel('Count')
    
    # Plot transaction types split by fraud
    transaction_counts_split_by_fraud(df).plot(ax=axs[1], kind='bar', stacked=True)
    axs[1].set_title('Transaction Types Split by Fraud')
    axs[1].set_xlabel('Transaction Type')
    axs[1].set_ylabel('Count')
    
    # Overall title
    fig.suptitle('Transaction Analysis')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add annotations
    for ax in axs:
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), 
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    # Return description
    return (
        'The first chart shows the count of each transaction type. The second chart displays '
        'the counts of each transaction type split by whether fraud was detected. This allows '
        'us to see how different types of transactions are distributed and how they relate to fraud detection.'
    )

def visual_2(df):
    def query(df):
        # Filter for 'CASH_OUT' transactions and select relevant columns
        return df[df['type'] == 'CASH_OUT'][['oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    
    # Calculate balance deltas
    cash_out_df = query(df)
    cash_out_df['orig_balance_delta'] = cash_out_df['newbalanceOrig'] - cash_out_df['oldbalanceOrig']
    cash_out_df['dest_balance_delta'] = cash_out_df['newbalanceDest'] - cash_out_df['oldbalanceDest']
    
    # Create scatter plot
    plot = cash_out_df.plot.scatter(x='orig_balance_delta', y='dest_balance_delta')
    plot.set_title('Origin vs. Destination Balance Delta for Cash Out Transactions')
    plot.set_xlabel('Origin Account Balance Delta')
    plot.set_ylabel('Destination Account Balance Delta')
    plot.set_xlim(left=-1e3, right=1e3)
    plot.set_ylim(bottom=-1e3, top=1e3)
    
    # Return description
    return (
        'This scatter plot shows the relationship between the balance delta of the origin account and '
        'the balance delta of the destination account for cash out transactions. It helps to understand how '
        'the cash out operations affect both the origin and destination accounts.'
    )

def exercise_custom(df):
    
    # Compute the total number of transactions and fraudulent transactions for each type
    fraud_counts = df[df['isFraud'] == 1]['type'].value_counts()
    total_counts = df['type'].value_counts()
    
    # Compute fraud rate for each transaction type
    fraud_rate = fraud_counts / total_counts
    return fraud_rate.sort_values(ascending=False)

def visual_custom(df):
   
    fraud_rate = exercise_custom(df)
    
    # Create the bar plot
    plot = fraud_rate.plot(kind='bar', color='skyblue')
    plot.set_title('Fraud Rate by Transaction Type')
    plot.set_xlabel('Transaction Type')
    plot.set_ylabel('Fraud Rate')
    
    # Set y-axis limit from 0 to 1 since fraud rate is a proportion
    plot.set_ylim(0, 1)
    
    # Add value labels on top of the bars
    for p in plot.patches:
        plot.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()), 
                      ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Return description
    return (
        'This bar chart shows the fraud rate for each transaction type. The fraud rate is calculated as the '
        'proportion of fraudulent transactions relative to the total number of transactions for each type. '
        'A higher fraud rate indicates that a particular transaction type is more frequently associated with fraud.'
    )