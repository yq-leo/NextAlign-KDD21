import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def write_training_setting():
    pass


def write_training_records_to_csv(num_epochs, stat, file_path, hit_top_ks=(1, 5, 10, 30, 50, 100)):
    """Write the training process data to a CSV file."""
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        col_headers = ['Epoch', 'Loss', 'MRR'] + [f"Hit@{k}" for k in hit_top_ks]
        writer.writerow(col_headers)

        losses, mrrs, hitss = stat['loss'], stat['mrr'], stat['hit']
        for epoch in range(num_epochs):
            loss, mrr, hits = losses[epoch], mrrs[epoch], hitss[epoch]
            # Write the epoch number, loss, and accuracy to the CSV file
            writer.writerow([epoch, f"{loss:.4f}", f"{mrr:.4f}"] + [f"{hits[k]:.4f}" for k in hit_top_ks])

            # Optionally, print the epoch, loss, and accuracy to the console
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}, MRR: {mrr:.4f}, " + ", ".join(
                [f"Hit@{k}: {hits[k]:.4f}" for k in hit_top_ks]))


def read_training_records_from_csv(file_path, hit_top_ks = (1, 5, 10, 30, 50, 100)):
    loss, mrr = [], []
    hits = {hit_top_k: [] for hit_top_k in hit_top_ks}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(row)
            loss.append(row["Loss"])
            mrr.append(row["MRR"])
            for hit_top_k in hit_top_ks:
                hits[hit_top_k].append(row[f"Hit@{hit_top_k}"])
    loss, mrr = np.array(loss), np.array(mrr)
    for hit_top_k in hit_top_ks:
        hits[hit_top_k] = np.array(hits[hit_top_k])
    return loss, mrr, hits


def plot_training_records(data, use_attr):
    # Load your DataFrame
    src_f = f'results/{data}_training_records.csv'
    if use_attr:
        src_f = f'results/{data}_attr_training_records.csv'
    df = pd.read_csv(src_f)  # Update with your CSV file path

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 2 Rows, 2 Column

    # Adjust space between plots
    fig.subplots_adjust(hspace=0.4)

    # Plot 1: Loss vs. Training Epoch
    axs[0].plot(df['Epoch'], df['Loss'], 'r-')  # 'r-' for red solid line
    axs[0].set_title('Loss vs. Training Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    # Plot 2: MRR vs. Epoch
    axs[1].plot(df['Epoch'], df['MRR'], 'b-')  # 'b-' for blue solid line
    axs[1].set_title('MRR vs. Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('MRR')

    # Plot 3: Hits@1 vs. Training Epoch
    # Assuming 'Hits@1' is a column in your DataFrame; adjust if needed
    hits_columns = [col for col in df.columns if 'Hit@' in col]
    for col in hits_columns:
        axs[2].plot(df['Epoch'], df[col], label=col)
    # axs[2].plot(df['Epoch'], df['Hits@1'], 'g-')  # 'g-' for green solid line
    axs[2].set_title('Hits@ vs. Training Epoch')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Hits@')

    # Show the big graph containing all subgraphs
    plt.legend()
    dst_f = f"results/{data}_plot.jpg"
    if use_attr:
        dst_f = f"results/{data}_attr_plot.jpg"
    plt.savefig(dst_f, dpi=720)
    plt.show()
