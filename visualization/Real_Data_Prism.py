import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Plot configuration
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'mathtext.rm': 'serif',
    'mathtext.fontset': 'custom'
})

# Load and clean data
file_path = 'PRISM_ppt_tmean_tdmean.csv'
df = pd.read_csv(file_path, skiprows=10)
df.columns = ['Date', 'Precipitation (inches)', 'Mean Temp (F)', 'Mean Dewpoint Temp (F)']
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# --- Begin Plotting ---
fig, ax1 = plt.subplots(figsize=(6, 6), dpi=300)

# Move spines (axes lines) behind content
ax1.set_axisbelow(True)

# Plot precipitation as bars
bars = ax1.bar(df.index, df['Precipitation (inches)'],
               color='blue', width=20, label='Precipitation (inches)')
ax1.set_ylabel('Precipitation (inches)', fontsize=9, color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
ax1.spines['left'].set_color('black')
ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

# Twin axis for temperature
ax2 = ax1.twinx()
ax2.plot(df.index, df['Mean Temp (F)'], color='red', linewidth=1.5, label='Mean Temp (F)')
ax2.plot(df.index, df['Mean Dewpoint Temp (F)'], color='green', linewidth=1.5, label='Mean Dewpoint Temp (F)')
ax2.set_ylabel('Temperature (°F)', fontsize=9, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=8)
ax2.spines['right'].set_color('black')

# X-axis formatting
ax1.set_xlabel('Date', fontsize=9)
ax1.tick_params(axis='x', labelrotation=45, labelsize=8)
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # every month
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # custom format

# Combine legends and move to the top (outside)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
combined_lines = lines1 + lines2
combined_labels = labels1 + labels2

legend = fig.legend(combined_lines, combined_labels,
                    loc='upper center', bbox_to_anchor=(0.5, 1),
                    ncol=3, fontsize=8, frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')

# Title and layout
#plt.title('PRISM Climate Data: Precipitation and Temperature Trends', fontsize=10, pad=25)
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for top legend

# Show plot
plt.show()
