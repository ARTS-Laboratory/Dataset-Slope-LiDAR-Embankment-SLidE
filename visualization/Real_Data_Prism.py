import pandas as pd
import matplotlib.pyplot as plt

# Update plot settings for LaTeX-style formatting (no real LaTeX engine required)
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'mathtext.rm': 'serif',
    'mathtext.fontset': 'custom'
})

# Load and clean PRISM CSV data
file_path = 'PRISM_ppt_tmean_tdmean_stable_800m_202108_202309_32.2799_-90.2117.csv'
df = pd.read_csv(file_path, skiprows=10)
df.columns = ['Date', 'Precipitation (inches)', 'Mean Temp (F)', 'Mean Dewpoint Temp (F)']
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Create plot
fig, ax1 = plt.subplots(figsize=(12, 4))

# Precipitation as bars (primary y-axis)
ax1.bar(df.index, df['Precipitation (inches)'], color='blue', label='Precipitation (inches)', width=20)
ax1.set_ylabel('Inches', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.spines['left'].set_color('black')

# Temperature and dewpoint on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(df.index, df['Mean Temp (F)'], color='red', linewidth=2, label='Mean Temp (F)')
ax2.plot(df.index, df['Mean Dewpoint Temp (F)'], color='green', linewidth=2, label='Mean Dewpoint Temp (F)')
ax2.set_ylabel('Temperature (°F)', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.spines['right'].set_color('black')

# X-axis formatting
ax1.set_xlabel('Date')
ax1.tick_params(axis='x', rotation=45)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')

# Layout and display
fig.tight_layout()
plt.title('PRISM Climate Data: Precipitation and Temperature Trends')
plt.show()

