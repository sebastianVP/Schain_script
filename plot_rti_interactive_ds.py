import os
import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from datetime import datetime, timezone, timedelta
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Add this import at the top
from pathlib import Path
from matplotlib.gridspec import GridSpec  # 


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


def freq_vel_Range(ipp, freq, nFFTPoints):
    """Calcula el rango de frecuencias y velocidades utilizando los atributos del HDF5."""
    PRF = 1 / ipp  # Frecuencia de repetición de pulso
    fmax = PRF / 2  # Máxima frecuencia
    C = 3.0e8  # Velocidad de la luz en m/s
    _lambda_ = C / (freq * 1e6)  # Longitud de onda
    vmax = fmax * _lambda_ / 2.0  # Velocidad máxima

    deltafreq = fmax / nFFTPoints
    freqrange = 2 * deltafreq * (np.arange(nFFTPoints) - nFFTPoints / 2.0)

    deltavel = vmax / nFFTPoints
    velrange = 2 * deltavel * (np.arange(nFFTPoints) - nFFTPoints / 2.0)

    return freqrange, velrange

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

def integracion_incoherente(data, block_size):
    """
    Realiza la integración incoherente de los datos promediando bloques de la primera dimensión de 'data'.

    :param data: ndarray de forma (20, 4, 1000, 936)
    :param block_size: tamaño de los bloques a promediar
    :return: ndarray con los datos promediados
    """
    # Número de bloques que se pueden formar
    num_blocks = data.shape[0] // block_size

    # Iniciar un arreglo vacío para almacenar los resultados de la integración incoherente
    integrated_data = []

    for i in range(num_blocks):
        # Seleccionar el bloque de datos
        block = data[i * block_size: (i + 1) * block_size]

        # Promediar a lo largo de la primera dimensión (bloques de 5 elementos)
        integrated_block = np.nansum(block, axis=0)

        # Almacenar el bloque promediado
        integrated_data.append(integrated_block)

    # Convertir la lista de bloques promediados a un arreglo numpy
    integrated_data = np.array(integrated_data)

    return integrated_data



def plot_spectra(data, metadata, channel_names, utc_time, normFactor, 
                j=0, int_inc=None, power_min=None, power_max=None,
                MinVel=None, MaxVel=None, min_hei=None, max_hei=None,
                interactive=False, pause_time=0.5, save_plots=False):
    """
    Plot and save spectrograms with customizable height and velocity ranges.
    
    Parameters:
        data (ndarray): Input data array (blocks x channels x profiles x heights)
        metadata (dict): Radar metadata dictionary
        channel_names (list): List of channel names
        utc_time (array): Array of UTC timestamps
        normFactor (float): Normalization factor
        j (int): Initial block index
        int_inc (int): Integration count
        power_min/max (float): Color scale limits
        MinVel/MaxVel (float): Velocity range limits
        min_hei/max_hei (float): Height range limits
        interactive (bool): Enable iterative updates
        pause_time (float): Delay between updates
        save_plots (bool): Save plots to 'spc' subdirectory
    """
    # Input validation
    if (data is None or metadata is None or channel_names is None 
        or utc_time is None or normFactor is None):
        raise ValueError("Critical parameters cannot be None")

    # Create output directory if saving
    if save_plots:
        output_dir = Path(__file__).parent / "spc"
        output_dir.mkdir(exist_ok=True)

    # Process data with integration if specified
    processed_data = integracion_incoherente(data, int_inc) if int_inc else data
    n_blocks = len(processed_data)

    # Initialize figure
    fig = plt.figure(figsize=(14, 8), num="Spectrograms Analysis")
    gs = GridSpec(2, 6, figure=fig, width_ratios=[4, 1, 0.3, 4, 1, 0.3], 
                 hspace=0.3, wspace=0.3)
    
    # Store plot objects for updates
    plot_objects = {
        'heatmaps': [],
        'profiles': [],
        'cbar': None,
        'axes': []
    }

    def calculate_power_data(j):
        """Calculate power data for given block index with height filtering"""
        data_spc = processed_data[j]
        current_utc = utc_time[j] - 18000  # UTC-5 conversion
        readable_date = datetime.utcfromtimestamp(current_utc).strftime("%Y-%m-%d %H:%M:%S")
        
        # Get radar parameters
        ippSeconds = metadata.get("ippSeconds", 1)
        frequency = metadata.get("freq", 49.92)
        nFFTPoints = metadata.get("nFFTPoints", 256)
        height_list = metadata.get("heightList", np.arange(data_spc.shape[2]))
        
        # Apply height filtering
        if min_hei is not None or max_hei is not None:
            height_mask = ((height_list >= (min_hei or -np.inf)) & 
                          (height_list <= (max_hei or np.inf)))
            height_list = height_list[height_mask]
        else:
            height_mask = slice(None)
        
        # Calculate velocity range
        _, vel_range = freq_vel_Range(ippSeconds, frequency, nFFTPoints)
        
        # Velocity filtering
        vel_mask = ((vel_range >= (MinVel or -np.inf)) & 
                   (vel_range <= (MaxVel or np.inf))) if (MinVel is not None or MaxVel is not None) else slice(None)
        vel_range_filtered = vel_range[vel_mask]
        
        # Power calculation with filtering
        power = data_spc / (normFactor * (int_inc if int_inc else 1))
        power_db = 10 * np.log10(power.transpose(0, 2, 1))  # Shape: (channels, heights, velocities)
        power_db = np.nan_to_num(power_db, nan=np.nanmin(power_db))
        power_db_filtered = power_db[:, height_mask, :][:, :, vel_mask]
        power_profiles = np.mean(power_db_filtered, axis=2)
        
        return {
            'power_db': power_db_filtered,
            'profiles': power_profiles,
            'vel_range': vel_range_filtered,
            'heights': height_list,
            'date': readable_date,
            'zmin': power_min if power_min is not None else np.min(power_db_filtered),
            'zmax': power_max if power_max is not None else np.max(power_db_filtered)
        }

    def save_current_plot(j):
        """Save current plot to spc subdirectory"""
        if not save_plots:
            return
            
        data_dict = calculate_power_data(j)
        timestamp = datetime.utcfromtimestamp(utc_time[j]).strftime("%Y%m%d_%H%M%S")
        filename = f"spectrogram_{timestamp}.png"
        save_path = output_dir / filename
        
        try:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved spectrogram to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {str(e)}")

    def init_plots():
        """Initialize all plot elements"""
        nonlocal plot_objects
        
        plt.clf()  # Clear existing figure
        data_dict = calculate_power_data(j)
        
        # Create subplots for each channel
        for idx, channel in enumerate(channel_names):
            row = 0 if idx < 2 else 1
            col_base = 0 if idx % 2 == 0 else 3

            ax_heatmap = fig.add_subplot(gs[row, col_base])
            ax_profile = fig.add_subplot(gs[row, col_base + 1])

            # Create heatmap with filtered heights
            hm = ax_heatmap.imshow(
                data_dict['power_db'][idx],
                aspect='auto',
                origin='lower',
                extent=[data_dict['vel_range'][0], data_dict['vel_range'][-1], 
                       data_dict['heights'][0], data_dict['heights'][-1]],
                cmap='jet',
                vmin=data_dict['zmin'],
                vmax=data_dict['zmax']
            )
            ax_heatmap.set_title(channel)
            ax_heatmap.set_xlabel("Velocity (m/s)")
            ax_heatmap.set_ylabel("Altitude (km)")

            # Create power profile plot
            profile, = ax_profile.plot(data_dict['profiles'][idx], data_dict['heights'], 'k-')
            ax_profile.set_xlabel("Power (dB)")
            ax_profile.set_yticklabels([])
            ax_profile.grid(True, alpha=0.3)

            plot_objects['heatmaps'].append(hm)
            plot_objects['profiles'].append(profile)
            plot_objects['axes'].extend([ax_heatmap, ax_profile])

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        plot_objects['cbar'] = fig.colorbar(plot_objects['heatmaps'][0], cax=cbar_ax, label="Power (dB)")

        fig.suptitle(f"Spectrograms | Local Time: {data_dict['date']}\n"
                    f"Heights: {data_dict['heights'][0]:.1f}-{data_dict['heights'][-1]:.1f} km | "
                    f"Velocities: {data_dict['vel_range'][0]:.1f}-{data_dict['vel_range'][-1]:.1f} m/s", 
                    fontsize=12, weight='bold')
        plt.tight_layout(rect=[0, 0, 0.91, 0.95])

        # Save initial plot if requested
        if save_plots and not interactive:
            save_current_plot(j)

    def update_plots(j):
        """Update plots with data from block j"""
        data_dict = calculate_power_data(j)
        
        # Update heatmaps and profiles
        for idx in range(len(channel_names)):
            plot_objects['heatmaps'][idx].set_data(data_dict['power_db'][idx])
            plot_objects['heatmaps'][idx].set_clim(vmin=data_dict['zmin'], vmax=data_dict['zmax'])
            plot_objects['profiles'][idx].set_data(data_dict['profiles'][idx], data_dict['heights'])
            
            # Update axes limits
            plot_objects['heatmaps'][idx].axes.set_xlim(data_dict['vel_range'][0], data_dict['vel_range'][-1])
            plot_objects['heatmaps'][idx].axes.set_ylim(data_dict['heights'][0], data_dict['heights'][-1])
            
            # Update profile plot limits
            ax_profile = plot_objects['profiles'][idx].axes
            ax_profile.set_xlim(np.min(data_dict['profiles'][idx])-5, 
                              np.max(data_dict['profiles'][idx])+5)
            ax_profile.set_ylim(data_dict['heights'][0], data_dict['heights'][-1])

        # Update title
        fig.suptitle(f"Spectrograms | Local Time: {data_dict['date']}\n"
                    f"Heights: {data_dict['heights'][0]:.1f}-{data_dict['heights'][-1]:.1f} km | "
                    f"Velocities: {data_dict['vel_range'][0]:.1f}-{data_dict['vel_range'][-1]:.1f} m/s", 
                    fontsize=12, weight='bold')

        # Save plot if in interactive mode
        if save_plots and interactive:
            save_current_plot(j)

    # Initial plot setup
    init_plots()
    
    # Interactive mode handling
    if interactive:
        plt.ion()
        try:
            for block_idx in range(n_blocks):
                update_plots(block_idx)
                plt.draw()
                plt.pause(pause_time)
                
                if not plt.fignum_exists(fig.number):
                    break
        finally:
            plt.ioff()
    else:
        plt.show()

    return fig



def plot_rti(data_spc, utc, normFactor, metadata, channel_names, 
             power_min=None, power_max=None, height_min=None, height_max=None,
             interactive=False, pause_time=0.5,save_plot=False):
    '''
    Plot RTI graph based on spectral data with customizable limits.
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
            block_data = data_spc[blk, ch, :, :]
            avg_profile = np.mean(block_data, axis=0)
            power_db = 10 * np.log10(avg_profile / normFactor)
            avg_power_profiles[ch, blk, :] = power_db
    
    # Set power and height limits
    vmin = power_min if power_min is not None else np.nanmin(avg_power_profiles)
    vmax = power_max if power_max is not None else np.nanmax(avg_power_profiles)
    height_min = height_min if height_min is not None else min(height_list)
    height_max = height_max if height_max is not None else max(height_list)
    
    # Filter height list and data
    height_indices = np.where((height_list >= height_min) & (height_list <= height_max))[0]
    filtered_height_list = height_list[height_indices]
    filtered_power_profiles = avg_power_profiles[:, :, height_indices]
    
    # Create custom colormap
    jet = plt.cm.jet
    jet_colors = jet(np.linspace(0, 1, 256))
    custom_colors = jet_colors.copy()
    target_purple = np.array([0.6, 0.0, 0.8, 1.0])
    transition_point = 210
    for i in range(transition_point, 256):
        t = (i - transition_point) / (255 - transition_point)
        custom_colors[i] = (1-t) * custom_colors[i] + t * target_purple
    jet_to_vivid_purple = LinearSegmentedColormap.from_list("jet_vivid_purple", custom_colors)
    
    # Create figure and gridspec
    fig = plt.figure(figsize=(12, 3.5 * n_channels + 0.5), dpi=100)
    gs = fig.add_gridspec(n_channels, 1, height_ratios=[1] * n_channels, 
                         left=0.1, right=0.9, bottom=0.1, top=0.92,
                         hspace=0.4)
    
    # Create axes and store them
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n_channels)]
    
    # Initialize plot objects
    mesh_objects = [None] * n_channels
    cbar_objects = [None] * n_channels
    
    if interactive:
        plt.ion()
        
        for block_idx in range(1, n_blocks + 1):
            plt.suptitle(f"RTI Plot - {frequency} MHz | Power Range: [{vmin:.1f}, {vmax:.1f}] dB | "
                        f"Height: [{height_min}-{height_max}] km | Block: {block_idx}/{n_blocks}", 
                        fontsize=12, fontweight='bold', y=0.99)
            
            for ch in range(n_channels):
                ax = axes[ch]
                
                # Clear the axis completely (this removes everything including colorbars)
                ax.clear()
                
                # Create new mesh
                mesh_objects[ch] = ax.pcolormesh(
                    time_axis[:block_idx],
                    filtered_height_list,
                    filtered_power_profiles[ch, :block_idx, :].T,
                    cmap=jet_to_vivid_purple,
                    shading='gouraud',
                    vmin=vmin,
                    vmax=vmax
                )
                
                # Create new colorbar if it doesn't exist
                if cbar_objects[ch] is None:
                    cbar_objects[ch] = fig.colorbar(mesh_objects[ch], ax=ax)
                    cbar_objects[ch].set_label("Power (dB)", fontsize=10, fontweight='bold')
                    cbar_objects[ch].ax.tick_params(labelsize=9)
                else:
                    # Update existing colorbar
                    cbar_objects[ch].update_normal(mesh_objects[ch])
                
                # Set axis properties
                ax.set_ylabel("Altitude (km)", fontsize=11, fontweight='bold')
                ax.set_title(f"Channel: {channel_names[ch]}", fontsize=12, fontweight='bold')
                ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
                ax.grid(True, alpha=0.3, linestyle='--', color='gray')
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.set_ylim(height_min, height_max)
                
                if ch == n_channels - 1:
                    ax.set_xlabel("Time (UTC-5)", fontsize=11, fontweight='bold')
                
                if block_idx > 1:
                    ax.set_xlim(time_axis[0], time_axis[block_idx-1])
            
            plt.draw()
            plt.pause(pause_time)
        
        plt.ioff()
        
    else:
        # Non-interactive plot
        for ch in range(n_channels):
            ax = axes[ch]
            
            mesh_objects[ch] = ax.pcolormesh(
                time_axis,
                filtered_height_list,
                filtered_power_profiles[ch].T,
                cmap=jet_to_vivid_purple,
                shading='gouraud',
                vmin=vmin,
                vmax=vmax
            )
            
            cbar_objects[ch] = fig.colorbar(mesh_objects[ch], ax=ax)
            cbar_objects[ch].set_label("Power (dB)", fontsize=10, fontweight='bold')
            cbar_objects[ch].ax.tick_params(labelsize=9)
            
            ax.set_ylabel("Altitude (km)", fontsize=11, fontweight='bold')
            ax.set_title(f"Channel: {channel_names[ch]}", fontsize=12, fontweight='bold')
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.grid(True, alpha=0.3, linestyle='--', color='gray')
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.set_ylim(height_min, height_max)
            
            if ch == n_channels - 1:
                ax.set_xlabel("Local Time ", fontsize=11, fontweight='bold')
        
        plt.suptitle(f"RTI Plot - {frequency} MHz | Power Range: [{vmin:.1f}, {vmax:.1f}] dB | Height: [{height_min}-{height_max}] km", 
                    fontsize=12, fontweight='bold', y=0.99)
    
    plt.show()
    # Save plot functionality
    if save_plot:
        try:
            # Get the directory of the current script
            script_dir = Path(__file__).parent.resolve()
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"RTI_Plot_{timestamp}.png"
            save_path = script_dir / filename
            
            # Save with high quality
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"RTI successfully saved in:: {save_path}")
        except Exception as e:
            print(f"Error saving graph: {e}")
    
    
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
# directory = "/media/soporte/DATA/150kM/"
data_spc, metadata, channel_names, utc, normFactor = read_atributos_from_dir(directory)
"""
plot_rti(data_spc, utc,
         normFactor, 
         metadata, 
         channel_names, 
         power_min=15, 
         power_max=30, 
         height_min=60, 
         height_max=200, 
         interactive=False,
         save_plot=True)
"""
plot_rti(data_spc, utc, 
         normFactor,
         metadata, 
         channel_names, 
         power_min=15, 
         power_max=30, 
         height_min=60, 
         height_max=200, 
         interactive=True,
         pause_time=0.3,
         save_plot=True)

plot_spectra(data_spc, metadata, channel_names, utc, normFactor,
            MinVel=-100, MaxVel=100,power_min=15,power_max=30, min_hei=100, max_hei=180, interactive=True)