import os
import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timezone, timedelta
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

def read_atributos_from_dir(directory):
    all_data = []
    all_utc = []
    all_metadata = []
    all_channel_names = None
    total_normFactor = 0

    hdf5_files = sorted([f for f in os.listdir(directory) if f.endswith('.hdf5')])
    if not hdf5_files:
        raise ValueError("❌ No .hdf5 files found in the directory.")

    for filename in hdf5_files:
        filepath = os.path.join(directory, filename)
        with h5py.File(filepath, "r") as obj:
            var_path = "Data/data_spc"
            channel_names = sorted(obj[var_path].keys())

            if all_channel_names is None:
                all_channel_names = channel_names
            elif channel_names != all_channel_names:
                raise ValueError(f"⚠️ Channel mismatch in {filename}.")

            utc_time = np.array(obj["Data/utctime"])
            data_list = [np.array(obj[f"{var_path}/{channel}"]) for channel in channel_names]
            data_arr = np.stack(data_list, axis=0)
            data_arr = np.moveaxis(data_arr, 1, 0)

            all_data.append(data_arr)
            all_utc.append(utc_time)

            metadata = {key: np.array(obj[f"Metadata/{key}"]) for key in obj["Metadata"].keys()}

            code = metadata.get("code")
            nProfiles = metadata.get("nProfiles", 1)
            nIncohInt = metadata.get("nIncohInt", 1)
            nCohInt = metadata.get("nCohInt", 1)
            windowOfFilter = metadata.get("windowOfFilter", 1)
            pwcode = 1

            if metadata.get("flagDecodeData", False):
                pwcode = np.sum(code[0] ** 2)

            normFactor = nProfiles * nIncohInt * nCohInt * pwcode * windowOfFilter

    final_data = np.concatenate(all_data, axis=0)
    final_utc = np.concatenate(all_utc, axis=0)
    return final_data, metadata, all_channel_names, final_utc, normFactor


def hildebrand_sekhon(data, navg):
    data = data.copy()
    sortdata = np.sort(data, axis=None)
    lenOfData = len(sortdata)
    nums_min = max(lenOfData // 10, 2)

    sump = 0.
    sumq = 0.
    j = 0
    cont = 1

    while cont == 1 and j < lenOfData:
        sump += sortdata[j]
        sumq += sortdata[j] ** 2
        j += 1
        if j > nums_min:
            rtest = float(j) / (j - 1) + 1.0 / navg
            if (sumq * j) > (rtest * sump ** 2):
                j -= 1
                sump -= sortdata[j]
                sumq -= sortdata[j] ** 2
                cont = 0

    lnoise = sump / j
    stdv = np.sqrt((sumq - lnoise ** 2) / (j - 1))
    return lnoise

def getNoisebyHildebrand(data_spc):
    noise = np.zeros(data_spc.shape[0], dtype='f')
    for channel in range(data_spc.shape[0]):
        daux = data_spc[channel, :, :]
        noise[channel] = hildebrand_sekhon(daux, 1)
    return noise

def plot_rti(data_spc, utc, normFactor, metadata, channel_names, 
             power_min=None, power_max=None, height_min=None, height_max=None,
             interactive=False, pause_time=0.5):
    '''
    Plot RTI graph based on spectral data with customizable limits.
    
    Parameters:
      - data_spc: numpy array of shape (n_blocks, n_channels, n_profiles, n_heights)
      - utc: list or array of UTC start times (in seconds since epoch) for each block
      - normFactor: normalization factor
      - metadata: dict containing radar metadata like ippSeconds, freq, nFFTPoints, heightList
      - channel_names: list of channel names
      - power_min: Minimum power value for color scale (dB). If None, uses data minimum.
      - power_max: Maximum power value for color scale (dB). If None, uses data maximum.
      - height_min: Minimum height to display (km). If None, uses minimum from height_list.
      - height_max: Maximum height to display (km). If None, uses maximum from height_list.
      - interactive: Boolean flag to enable interactive visualization (default: False)
      - pause_time: Time in seconds to pause between iterations (default: 0.5)
    '''
    
    n_blocks, n_channels, _, n_heights = data_spc.shape
    
    # Extract metadata with defaults if not provided
    ippSeconds = metadata.get("ippSeconds", 1)
    frequency = metadata.get("freq", 49.92)
    nFFTPoints = metadata.get("nFFTPoints", 256)
    height_list = metadata.get("heightList", np.arange(n_heights))
    
    # Convert UTC timestamps to timezone-aware datetime objects (UTC-5)
    time_axis = [datetime.fromtimestamp(t, tz=timezone.utc) - timedelta(hours=5)
                 for t in utc]
    
    # Calculate average power profiles
    avg_power_profiles = np.zeros((n_channels, n_blocks, n_heights))
    for blk in range(n_blocks):
        for ch in range(n_channels):
            block_data = data_spc[blk, ch, :, :]  # shape: (n_profiles, n_heights)
            avg_profile = np.mean(block_data, axis=0)  # shape: (n_heights,)
            power_db = 10 * np.log10(avg_profile / normFactor)
            avg_power_profiles[ch, blk, :] = power_db
    
    # Set power limits based on parameters or actual min/max without percentiles
    vmin = power_min if power_min is not None else np.nanmin(avg_power_profiles)
    vmax = power_max if power_max is not None else np.nanmax(avg_power_profiles)
    
    # Set height limits
    height_min = height_min if height_min is not None else min(height_list)
    height_max = height_max if height_max is not None else max(height_list)
    
    # Filter height list and data based on height limits
    height_indices = np.where((height_list >= height_min) & (height_list <= height_max))[0]
    filtered_height_list = height_list[height_indices]
    filtered_power_profiles = avg_power_profiles[:, :, height_indices]
    
    # Create custom colormap: similar to jet but ending in vivid purple
    jet = plt.cm.jet
    jet_colors = jet(np.linspace(0, 1, 256))
    custom_colors = jet_colors.copy()
    
    # Define more intense purple endpoint
    target_purple = np.array([0.6, 0.0, 0.8, 1.0])
    
    # Start transition slightly earlier for a more noticeable effect
    transition_point = 210  # Start transition earlier (out of 256 points)
    
    # Apply the transition to purple
    for i in range(transition_point, 256):
        # Calculate transition factor (0 at transition_point, 1 at the end)
        t = (i - transition_point) / (255 - transition_point)
        # Gradually blend from the jet color to our target purple
        custom_colors[i] = (1-t) * custom_colors[i] + t * target_purple
    
    # Create the custom colormap
    jet_to_vivid_purple = LinearSegmentedColormap.from_list("jet_vivid_purple", custom_colors)
    
    # Create figure with extra space at top for the suptitle
    fig = plt.figure(figsize=(12, 3.5 * n_channels + 0.5), dpi=100)
    
    # Create a more complex layout with space for title
    gs = fig.add_gridspec(n_channels, 1, height_ratios=[1] * n_channels, 
                         left=0.1, right=0.9, bottom=0.1, top=0.92,
                         hspace=0.4)
    
    # Create axes from the gridspec
    axes = []
    for i in range(n_channels):
        axes.append(fig.add_subplot(gs[i, 0]))
    
    # Initialize mesh objects for each channel
    mesh_objects = [None] * n_channels
    
    if interactive:
        plt.ion()  # Turn on interactive mode
        
        for block_idx in range(1, n_blocks + 1):
            # Update title to show progress
            plt.suptitle(f"RTI Plot - {frequency} MHz | Power Range: [{vmin:.1f}, {vmax:.1f}] dB | "
                        f"Height: [{height_min}-{height_max}] km | Block: {block_idx}/{n_blocks}", 
                        fontsize=12, fontweight='bold', y=0.99)
            
            # Plot each channel with data up to current block
            for ch in range(n_channels):
                ax = axes[ch]
                
                # Clear previous plot
                if mesh_objects[ch] is not None:
                    mesh_objects[ch].remove()
                
                # Create the pcolormesh with data up to current block
                mesh_objects[ch] = ax.pcolormesh(
                    time_axis[:block_idx],
                    filtered_height_list,
                    filtered_power_profiles[ch, :block_idx, :].T,
                    cmap=jet_to_vivid_purple,
                    shading='gouraud',
                    vmin=vmin,
                    vmax=vmax
                )
                
                # Set styles for better appearance
                ax.set_ylabel("Altitude (km)", fontsize=11, fontweight='bold')
                ax.set_title(f"Channel: {channel_names[ch]}", fontsize=12, fontweight='bold')
                
                # Format x-axis for time
                ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
                ax.grid(True, alpha=0.3, linestyle='--', color='gray')
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.set_ylim(height_min, height_max)
                
                # Only show x-label on bottom subplot
                if ch == n_channels - 1:
                    ax.set_xlabel("Time (UTC-5)", fontsize=11, fontweight='bold')
                
                # Adjust x-axis limits to show only the current data
                if block_idx > 1:
                    ax.set_xlim(time_axis[0], time_axis[block_idx-1])
                
            # Update colorbar for each iteration
            for ch, ax in enumerate(axes):
                # Remove old colorbar if it exists
                if hasattr(ax, 'colorbar') and ax.colorbar is not None:
                    ax.colorbar.remove()
                
                # Add new colorbar
                cbar = fig.colorbar(mesh_objects[ch], ax=ax, pad=0.01, fraction=0.046, aspect=30)
                cbar.set_label("Power (dB)", fontsize=10, fontweight='bold')
                cbar.ax.tick_params(labelsize=9)
                ax.colorbar = cbar
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.draw()
            plt.pause(pause_time)
            
        plt.ioff()  # Turn off interactive mode
        
    else:
        # Normal non-interactive plot (original behavior)
        for ch in range(n_channels):
            ax = axes[ch]
            
            mesh = ax.pcolormesh(
                time_axis,
                filtered_height_list,
                filtered_power_profiles[ch].T,
                cmap=jet_to_vivid_purple,
                shading='gouraud',
                vmin=vmin,
                vmax=vmax
            )
            
            # Set styles for better appearance
            ax.set_ylabel("Altitude (km)", fontsize=11, fontweight='bold')
            ax.set_title(f"Channel: {channel_names[ch]}", fontsize=12, fontweight='bold')
            
            # Format x-axis for time
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_ylim(height_min, height_max)
            
            # Only show x-label on bottom subplot
            if ch == n_channels - 1:
                ax.set_xlabel("Time (UTC-5)", fontsize=11, fontweight='bold')
            
            # Add colorbar with better formatting
            cbar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.046, aspect=30)
            cbar.set_label("Power (dB)", fontsize=10, fontweight='bold')
            cbar.ax.tick_params(labelsize=9)
        
        # Add main title with more information
        plt.suptitle(f"RTI Plot - {frequency} MHz | Power Range: [{vmin:.1f}, {vmax:.1f}] dB | Height: [{height_min}-{height_max}] km", 
                    fontsize=12, fontweight='bold', y=0.99)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()
    
    # Optional: return objects for additional manipulation
    return fig, axes

# Ejemplo de uso:
# directory = "/home/soporte/Downloads/"
# data_spc, metadata, channel_names, utc, normFactor = read_atributos_from_dir(directory)
# 
# # Uso normal (no interactivo)
# plot_rti(data_spc, utc, normFactor, metadata, channel_names, 
#          power_min=15, power_max=30, height_min=60, height_max=200, interactive=False)
# 
# # Uso interactivo (muestra los bloques uno por uno)
# plot_rti(data_spc, utc, normFactor, metadata, channel_names, 
#          power_min=15, power_max=30, height_min=60, height_max=200, 
#          interactive=True, pause_time=0.3)

# Example usage
directory = "/home/soporte/Downloads/"
data_spc, metadata, channel_names, utc, normFactor = read_atributos_from_dir(directory)

#plot_rti(data_spc, utc, normFactor, metadata, channel_names, 
#          power_min=15, power_max=30, height_min=60, height_max=200, interactive=False)

plot_rti(data_spc, utc, normFactor, metadata, channel_names, 
          power_min=15, power_max=30, height_min=60, height_max=200, 
          interactive=True, pause_time=0.3)