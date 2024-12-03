import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import psutil
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd

# Function to read data from a .txt file
def read_data(file_path):
    # Define the columns based on the tcpdump data format
    columns = ['Destination IP', 'Source IP', 'Source Port', 'Destination Port',
               'Protocol', 'Start Timestamp', 'Unknown1', 'Stop Timestamp',
               'Unknown2', 'Unknown3', 'SYN Flag', 'Unknown4', 'Unknown5',
               'Unknown6', 'Unknown7', 'Traffic Direction']

    # Read the data file
    df = pd.read_csv(file_path, sep=" ", header=None, names=columns)
    return df

# Function to filter SYN packets
def filter_syn_packets(df):
    # Filter to include only SYN packets (SYN Flag = 1)
    syn_packets = df[df['SYN Flag'] == 1].copy()  # Using .copy() to avoid SettingWithCopyWarning
    return syn_packets

# Function to convert Timestamps to datetime objects
def convert_to_datetime(df):
    # Convert the 'Start Timestamp' to a datetime format
    df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'], unit='s')
    return df

# Function to detect anomalies using Isolation Forest
def detect_anomalies_with_isolation_forest(df, window_size='80S', min_count=1000):
    # Group by time intervals, destination IP, and destination port, then count SYN packets
    grouped = df.groupby([pd.Grouper(key='Start Timestamp', freq=window_size),
                          'Destination IP', 'Destination Port']).size().reset_index(name='Count')

    # Apply Isolation Forest to detect anomalies
    clf = IsolationForest(contamination=0.0005)
    grouped['Anomaly'] = clf.fit_predict(grouped[['Count']])

    # Filter out the anomalies (SYN flooding)
    anomaly = grouped[grouped['Anomaly'] == -1]

    # Filter anomalies for counts greater than or equal to min_count
    filtered_anomalies = anomaly[anomaly['Count'] >= min_count].copy()

    # Sort the anomalies by destination IP and destination port
    top_bursts = filtered_anomalies.sort_values(by=['Count'], ascending=False).groupby(
        'Destination IP').first().reset_index()

    return top_bursts, grouped  # Return grouped DataFrame as well

# Function to print response times for the top bursts
def print_response_times(top_bursts, syn_packets, window_size):
    for index, row in top_bursts.iterrows():
        destination_ip = row['Destination IP']
        destination_port = row['Destination Port']

        # Find the source IPs associated with each attack
        attack_sources = syn_packets[(syn_packets['Destination IP'] == destination_ip) &
                                     (syn_packets['Destination Port'] == destination_port) &
                                     (syn_packets['Start Timestamp'] >= row['Start Timestamp']) &
                                     (syn_packets['Start Timestamp'] <= row['Start Timestamp'] + pd.Timedelta(
                                         seconds=int(window_size[:-1])))]['Source IP'].unique()

        source_ip_summary = f"{len(attack_sources)} unique source IPs" if len(
            attack_sources) > 1 else f"Source IP: {attack_sources[0]}" if attack_sources else "No source IPs found"

        # Calculate Response Time
        response_time = pd.Timedelta(seconds=int(window_size[:-1]))

        print("--------------------------------------------------\n"
              f"Response Times for SYN Flood Attack:\n"
              f"  Start Time: {row['Start Timestamp']}\n"
              f"  End Time: {row['Start Timestamp'] + response_time}\n"
              f"  {source_ip_summary}\n"
              f"  Destination IP: {destination_ip}\n"
              f"  Destination Port: {destination_port}\n"
              f"  Packet Count: {row['Count']}\n"
              f"  Response Time: {response_time}\n"
              "--------------------------------------------------")

# Function to evaluate the model and calculate performance metrics
def evaluate_model(df, min_count=1000):
    # Define y_true based on SYN Flag
    y_true = df.apply(lambda x: 1 if x['Count'] >= min_count else 0, axis=1)

    # Get predictions from the model and convert from {-1, 1} to {1, 0}
    y_pred = [0 if pred == 1 else 1 for pred in df['Anomaly']]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1

# Function to plot anomaly visualization
def plot_anomaly_visualization(grouped, top_bursts, window_size):
    # Create a colormap for distinctive colors
    colors = plt.cm.jet(np.linspace(0, 1, len(top_bursts)))  # Generate colors for unique IPs

    # Create a single figure for plotting
    fig, ax = plt.subplots(figsize=(15, 10))

    # Print consolidated results with separators and create data for plotting
    for index, row in top_bursts.iterrows():
        destination_ip = row['Destination IP']
        destination_port = row['Destination Port']

        # Calculate Response Time
        response_time = pd.Timedelta(seconds=int(window_size[:-1]))

        # Plot the data with labels, colors, and legends
        color = colors[index]  # Use distinctive colors
        label = f"{destination_ip}:{destination_port} (Packets: {row['Count']})"
        ax.plot([row['Start Timestamp'], row['Start Timestamp'] + response_time], [index, index], marker='o',
                color=color, label=label)

    # Formatting the plot
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.yticks(range(len(top_bursts)),
               [f"{row['Destination IP']}:{row['Destination Port']}" for index, row in top_bursts.iterrows()])
    plt.xlabel('Time')
    plt.ylabel('Destination IP and Port')
    plt.title('SYN Flooding Attack Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Ensure the DataFrame is sorted by 'Start Timestamp' for accurate plotting
    grouped = grouped.sort_values(by='Start Timestamp')

    # Additional Visualizations:

    # Time-Series Plot of SYN Packet Counts
    plt.figure(figsize=(15, 6))
    plt.plot(grouped['Start Timestamp'], grouped['Count'], marker='o', linestyle='-')
    plt.title('Time-Series of SYN Packet Counts')
    plt.xlabel('Time')
    plt.ylabel('SYN Packet Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Histogram of SYN Packet Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(grouped['Count'], bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of SYN Packet Distribution')
    plt.xlabel('SYN Packet Count')
    plt.ylabel('Frequency')
    plt.show()

    # Heatmap of SYN Attacks Over Time and Destination IP
    heatmap_data = grouped.pivot_table(index="Start Timestamp", columns="Destination IP", values="Count", fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='viridis')
    plt.title('Heatmap of SYN Attacks Over Time by Destination IP')
    plt.xlabel('Destination IP')
    plt.ylabel('Time')
    plt.show()

    # Scatter Plot of SYN Packets with Anomaly Indication
    plt.figure(figsize=(15, 6))
    plt.scatter(grouped['Start Timestamp'], grouped['Count'],
                c=grouped['Anomaly'], cmap='RdYlGn', marker='o')
    plt.title('SYN Packets with Anomaly Indication')
    plt.xlabel('Time')
    plt.ylabel('SYN Packet Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to calculate and print False Positive/Negative Rates and Detection Rate
def calculate_and_print_rates(top_bursts, min_count):
    # Define y_true based on SYN Flag
    y_true = top_bursts.apply(lambda x: 1 if x['Count'] >= min_count else 0, axis=1)

    # Calculate False Positive/Negative Rates and Detection Rate
    false_positive_count = sum((y_true == 0) & (top_bursts['Anomaly'] == -1))
    false_negative_count = sum((y_true == 1) & (top_bursts['Anomaly'] == 1))
    true_positive_count = sum((y_true == 1) & (top_bursts['Anomaly'] == -1))

    # Avoid division by zero by checking if the denominators are zero
    if sum(y_true == 0) > 0:
        false_positive_rate = false_positive_count / sum(y_true == 0)
    else:
        false_positive_rate = float('NaN')

    if sum(y_true == 1) > 0:
        false_negative_rate = false_negative_count / sum(y_true == 1)
        detection_rate = true_positive_count / sum(y_true == 1)
    else:
        false_negative_rate = float('NaN')
        detection_rate = float('NaN')

    # Print False Positive/Negative Rates and Detection Rate
    print("--------------------------------------------------\n"  # Upper delimiter
          f"False Positive Rate: {false_positive_rate}\nFalse Negative Rate: {false_negative_rate}\n"
          f"Detection Rate: {detection_rate}\n"
          "--------------------------------------------------\n")

# Main function
def main(file_path, window_size='80S', min_count=1000):
    # Get the current process for memory measurement
    process = psutil.Process(os.getpid())

    # Resource usage before execution
    psutil.cpu_percent()  # Initialize CPU measurement
    memory_before = process.memory_info().rss  # Memory used by the process

    # Start measuring time
    start_time = time.time()

    # Read data from TXT file
    df = read_data(file_path)

    # Filter and preprocess data
    syn_packets = filter_syn_packets(df)
    syn_packets = convert_to_datetime(syn_packets)

    # Detect anomalies using Isolation Forest and get grouped DataFrame
    top_bursts, grouped = detect_anomalies_with_isolation_forest(syn_packets, window_size, min_count)

    # Print response times for the top bursts
    print_response_times(top_bursts, syn_packets, window_size)

    # Plot anomaly visualization using grouped DataFrame
    plot_anomaly_visualization(grouped, top_bursts, window_size)

    # Evaluate the model and get performance metrics
    accuracy, precision, recall, f1 = evaluate_model(top_bursts, min_count)

    # Print the metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Calculate and print False Positive/Negative Rates and Detection Rate
    calculate_and_print_rates(top_bursts, min_count)

    # End measuring time
    end_time = time.time()
    execution_time = end_time - start_time

    # Delay for accurate CPU usage measurement
    time.sleep(1)

    # Calculate CPU and memory usage
    cpu_usage = psutil.cpu_percent()  # Average CPU usage over the period
    memory_after = process.memory_info().rss
    memory_usage = memory_after - memory_before  # Calculate memory usage specifically for this process

    # Output execution time, CPU usage, and memory usage with delimiters
    print("--------------------------------------------------\n"  # Upper delimiter
          f"Execution Time: {execution_time} seconds\n"
          f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage} bytes\n"
          "--------------------------------------------------\n")

if __name__ == "__main__":
    file_path = "FinalDumpFinal.txt"  # Replace with the actual path to your data file
    main(file_path)
