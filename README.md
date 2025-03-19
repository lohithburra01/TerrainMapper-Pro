### Project Title:
**TerrainMapper Pro: DEM Analysis & PBR Map Generator**

---

### GitHub README:

# TerrainMapper Pro: DEM Analysis & PBR Map Generator

![Banner Image](https://via.placeholder.com/1200x400)  
*(Replace with a banner image showcasing the app in action, e.g., a 3D terrain visualization or exported maps)*

---

## Overview

**TerrainMapper Pro** is a powerful web-based tool for analyzing Digital Elevation Models (DEMs) and generating high-quality Physically Based Rendering (PBR) maps for 3D modeling and visualization. Built with Python, Streamlit, and Rasterio, this application allows users to:

- Download DEM data for any region using the OpenTopography API.
- Generate **contour maps** and **hillshade visualizations**.
- Create **3D terrain models** for interactive exploration.
- Export **PBR maps** (Normal, Roughness, Height, and Ambient Occlusion) for use in Blender, Unity, Unreal Engine, or other 3D software.

Whether you're a GIS professional, 3D artist, or hobbyist, TerrainMapper Pro simplifies the process of terrain analysis and map generation.

---

## Features

### 1. **DEM Download**
   - Fetch DEM data for any region using latitude/longitude coordinates.
   - Supports multiple DEM types (SRTMGL3, SRTMGL1, AW3D30, GEDI).
   - Preview downloaded DEMs with elevation statistics.

### 2. **Contour Generation**
   - Generate customizable contour maps with adjustable intervals.
   - Overlay hillshade effects for enhanced visualization.
   - Export contour maps as high-resolution PNG images.

### 3. **3D Terrain Visualization**
   - Interactive 3D surface plots with adjustable vertical exaggeration.
   - Multiple color schemes for terrain representation.
   - Option to overlay contour lines on the 3D model.

### 4. **PBR Map Generation**
   - Generate **Normal Maps**, **Roughness Maps**, **Height Maps**, and **Ambient Occlusion Maps**.
   - Customize map parameters (e.g., smoothing, kernel size, intensity).
   - Export maps as PNG files for use in 3D software.

### 5. **Blender Integration**
   - Includes ready-to-use Python code for importing generated maps into Blender.
   - Set up PBR materials with a single click.

---

## Screenshots

![image](https://github.com/user-attachments/assets/fb7028e5-3156-47a3-9fc6-328588c41177) ![image](https://github.com/user-attachments/assets/6e7e4884-468e-49bd-bb72-06a1d1868245) ![image](https://github.com/user-attachments/assets/092ac1e6-5bab-4e30-a062-da63bd146e17)




1. **DEM Preview**  

![b62c21d95204e50eb5232f9c397719497039562fef5fd3bafbee675e](https://github.com/user-attachments/assets/95651bcb-c917-4324-a4f2-f5a4187d1230)    
   *Visualize elevation data with terrain colormaps.*

2. **Contour Map with Hillshade**  
![WhatsApp Image 2025-03-19 at 2 12 02 AM (2)](https://github.com/user-attachments/assets/e4efb015-73d3-4e0d-bc39-8b3a9e19b5d5)
   *Generate contour lines with customizable intervals and hillshade effects.*

3. **3D Terrain Model**  
![WhatsApp Image 2025-03-19 at 2 12 02 AM](https://github.com/user-attachments/assets/0a90aacc-0808-4835-a14e-f3d40aadd2f0)

   *Interactive 3D terrain visualization with adjustable vertical exaggeration.*

4. **PBR Maps**  
![WhatsApp Image 2025-03-19 at 2 12 02 AM (1)](https://github.com/user-attachments/assets/aa68ee25-dea3-430f-b867-a9d2fb133e7e)

   *Export Normal, Roughness, Height, and Ambient Occlusion maps for 3D rendering.*

---

## How It Works

1. **Download DEM Data**  
   - Enter the latitude/longitude bounds of your area of interest.
   - Provide an OpenTopography API key to fetch DEM data.

2. **Generate Maps**  
   - Use the app's tabs to create contour maps, 3D visualizations, and PBR maps.
   - Customize parameters like contour intervals, hillshade angles, and map smoothing.

3. **Export and Use**  
   - Export maps as PNG files.
   - Use the provided Blender Python script to import maps and set up PBR materials.

---

## Installation

To run TerrainMapper Pro locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/lohithburra01/terrainmapper-pro.git
   cd terrainmapper-pro
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`.

---

## Requirements

- Python 3.8+
- Streamlit
- Rasterio
- NumPy
- Matplotlib
- Plotly
- SciPy
- Pillow

---

## Usage

1. **DEM Download Tab**  
   - Enter the coordinates of your area of interest.
   - Provide an OpenTopography API key.
   - Download and preview the DEM.

2. **Contours Tab**  
   - Adjust contour intervals and hillshade settings.
   - Export the contour map as a PNG.

3. **3D Visualization Tab**  
   - Explore the terrain in 3D with adjustable settings.
   - Overlay contour lines for added detail.

4. **PBR Maps Tab**  
   - Generate and export Normal, Roughness, Height, and AO maps.
   - Use the provided Blender script to import maps.

---

## Contributing

Contributions are welcome! If you'd like to improve TerrainMapper Pro, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **OpenTopography** for providing DEM data via their API.
- **Streamlit** for enabling the creation of interactive web apps.
- **Rasterio** for handling geospatial data.

---

## Contact

For questions or feedback, feel free to reach out:

- **Email**: your-email@example.com
- **GitHub**: https://github.com/lohithburra01

---

**TerrainMapper Pro** is your one-stop solution for terrain analysis and PBR map generation. Start exploring today! 
