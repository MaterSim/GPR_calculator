import matplotlib.pyplot as plt
import numpy as np

# Create a function to read data from 'log_gp_neb.txt'
def read_data(filename, N_img=7):
    """
    Read energy data from log_gp_neb.txt.
    Extracts lines starting with 'From ' to get id, energy, and model name (base or surrogate).
    
    Args:
        filename: Path to the log file
        
    Returns:
        data: List of tuples with (id, energy, model_name)
    """
    data = []
    labels = []
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('Calculate'):
                E = float(line.strip().split()[-1])
                data.append([count, E])  # Store energy relative to reference
                labels.append('Base')  # Default label for initialization
                count += 1
                continue

            if count % N_img == 0:  # Only process every N_img-th line
                data.append([count, data[0][1]])  # Initialize with zero energy
                labels.append('GPR')  # Default label for initialization
                count += 1
                print(count, data[0][1], 'GPR')
                #continue

            if line.startswith('From '):
                parts = line.strip().split()    
                if parts[1] == 'Base':
                    model_name = 'Base'
                    col = 4
                else:
                    model_name = 'GPR'
                    col = 3
                eng = float(parts[col].split('/')[2][:-1])
                data.append([count, eng])
                labels.append(model_name)
                count += 1
                if count % N_img == N_img-1:  # Only process every N_img-th line
                    data.append([count, data[0][1]])  # Initialize with zero energy
                    labels.append('GPR')  # Default label for initialization
                    count += 1
                    #print(count, eng, model_name)
    print(len(data), len(labels))
    return np.array(data), labels


def plot_energy_data(data, labels, output_file='energy_scatter.png', N_img=7):
    """
    Create a scatter plot of energy data with points labeled by model type
    
    Args:
        data: List of tuples with (id, energy, model_name)
        labels: List of model names corresponding to the data points
        output_file: Name of the output file
    """
    # Create plot
    data = data[:-N_img]
    labels = labels[:-N_img]

    print(f"Plotting {len(data)} data points with {len(labels)} labels.")
    data[:, 1] -= data[0, 1]  # Normalize energy by subtracting the first energy value
    data[:, 0] /= N_img

    plt.figure(figsize=(12, 4))
    plt.plot(data[:, 0], data[:, 1], '-', color='grey', markersize=1, alpha=0.6)
    base_ids = [i for i in range(len(data)) if labels[i] == 'Base']
    plt.scatter(data[base_ids, 0], data[base_ids, 1], s=5, c='b', label='DFT')

    plt.xlabel('NEB Iteration id', fontsize=14)
    plt.ylabel('Energy', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.xlim(-0.5, len(data)/N_img+0.5)
    
    # Save figure
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved as '{output_file}'")

# Example usage
if __name__ == "__main__":
    data, labels = read_data('log_gp_neb.dat')
    plot_energy_data(data, labels)