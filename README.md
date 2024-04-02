# SLidE
## Slope LiDAR Embankment Dataset

![Project Image](<media/LiDAR_Point_Cloud_Surface.jpg>)

Highway cross-slope embankment failures are common in central Mississippi due to the existence of high-plasticity clay. Monitoring these terrain slopes is crucial to ensure road safety, prevent infrastructure damage, and mitigate potential economic losses. This repository presents an open-source Terrestrial LiDAR dataset that tracks the movement in embankments with high-plasticity clays through time. Among the Slope LiDAR embankment (SLidE) data, a highway embankment along the Terry Road Exit from I-20 westbound in Jackson, MS has been selected to monitor the slope movement. The embankment slope surfaces were scanned at different time intervals between Summer 2021 and Fall/Winter 2022, capturing the slopes' changing conditions over time.

The LiDAR data acquisition and analysis involved the use of Terrestrial LiDAR equipment (Trimble X7) to perform 3D laser scanning on embankment slope surfaces. These stations were carefully chosen, allowing for optimal data registration, and up to 10 spherical targets were positioned around the slopes to assist in the post-processing phase. Data collection was facilitated by the "Trimble Perspective" software loaded onto a handheld tablet (Trimble T10X).  For each investigation, the scanner was systematically moved between stations to ensure overlapping scans and reliable data registration. A total of five to six scanning stations were employed, resulting in the collection of multiple-point clouds with over 20 million points. 

The raw point clouds were refined through segmentation, eliminating undesirable points at the scan edges, as well as through the application of a ground extraction algorithm to remove vegetation and extraneous above-surface data points. The LiDAR focus was placed solely on the bare ground points. For georeferencing purposes, ground control points, aerial targets placed during UAV missions, and Global Navigation Satellite System (GNSS) measurements were employed. The comprehensive data acquisition approach enabled the precise alignment and superimposition of point clouds collected during different seasons. To optimize computational efficiency, these dense point clouds were sampled, increasing the minimum distance between data points. 

## Current Scans
- Summer 2021
- Fall 2021

## Usage
The datasets are currently provided in ```.laz``` format, a lightweight and easily interfaceable filetype containing the LiDAR pointcloud and scan metadata 

```bash
# To run the visualization script
git clone https://github.com/ARTS-Laboratory/Dataset-Slope-LiDAR-Embankment-SLidE.git
cd Dataset-Slope-LiDAR-Embankment-SLidE
pip install laszip laspy
cd visualization
python3 vis.py
```


## Licensing and Citation

This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License [cc-by-sa 4.0].

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)


Cite this data as: 

Sadik Khan, AQM Zohuruzzaman, David Wambi, Austin R.J. Downey. “Dataset-Slope-LiDAR-embankment-(SLidE),” April 2024. [Online]. Available: URL: https://github.com/ARTS-Laboratory/Dataset-Slope-LiDAR-Embankment-SLidE

#### Bibtex

@Article{Khan2024DatasetSlopeLidar,  
  author = {Sadik Khan and AQM Zohuruzzaman and David Wambi and Austin {R.J.} Downey},  
  title  = {Dataset-Slope-LiDAR-embankment-(SLidE)},  
  year   = {2024},  
  month  = apr,  
  url    = {https://github.com/ARTS-Laboratory/Dataset-Slope-LiDAR-Embankment-SLidE},  
}
