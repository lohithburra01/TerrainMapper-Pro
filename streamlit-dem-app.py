import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
import os
import io
import base64
from PIL import Image
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, uniform_filter, sobel
import rasterio
import rasterio.features
import tempfile

# Set page configuration
st.set_page_config(layout="wide", page_title="Terrain Analysis Tool")

# App title and description
st.title("Terrain Analysis and PBR Map Generator")
st.markdown("""
This application allows you to download and process Digital Elevation Models (DEM).
You can generate contours, hillshades, and PBR maps for use in Blender or other 3D software.
""")

# Create tabs
tabs = st.tabs(["DEM Download", "Contours", "3D Visualization", "PBR Maps"])

# Global session state to store DEM data
if 'dem_data' not in st.session_state:
    st.session_state.dem_data = None
if 'transform' not in st.session_state:
    st.session_state.transform = None
if 'crs' not in st.session_state:
    st.session_state.crs = None
if 'dem_file' not in st.session_state:
    st.session_state.dem_file = None

# Functions for DEM processing
def download_dem(south, north, west, east, api_key, dem_type="SRTMGL3", output_format="GTiff"):
    """
    Download DEM data from OpenTopography API
    """
    url = f"https://portal.opentopography.org/API/globaldem?demtype={dem_type}&south={south}&north={north}&west={west}&east={east}&outputFormat={output_format}&API_Key={api_key}"
    
    with st.spinner("Downloading DEM data..."):
        response = requests.get(url)
        
        if response.status_code == 200:
            # Save to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
            with open(temp_file.name, "wb") as f:
                f.write(response.content)
            
            st.success("DEM data downloaded successfully!")
            return temp_file.name
        else:
            st.error(f"Failed to download DEM data. Status code: {response.status_code}")
            if response.status_code == 401:
                st.error("API key error. Please check your API key.")
            return None

def load_dem(dem_path):
    """
    Load DEM data using rasterio
    """
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        transform = src.transform
        crs = src.crs
    return dem_data, transform, crs

def create_hillshade(dem_data, azimuth=315, altitude=45):
    """
    Create hillshade from DEM data
    """
    # Convert angles to radians
    azimuth_rad = np.radians(360.0 - azimuth)
    altitude_rad = np.radians(altitude)
    
    # Calculate slope and aspect
    x, y = np.gradient(dem_data)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    
    # Calculate hillshade
    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    
    # Scale to 0-255 and convert to uint8
    shaded = 255 * (shaded + 1) / 2
    return shaded.astype(np.uint8)

def create_contours(dem_data, transform, interval=100):
    """
    Generate contours from DEM data
    """
    min_height = np.floor(np.min(dem_data) / interval) * interval
    max_height = np.ceil(np.max(dem_data) / interval) * interval
    levels = np.arange(min_height, max_height + interval, interval)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(dem_data, cmap='terrain', alpha=0.5)
    contour = ax.contour(dem_data, levels=levels, colors='black', alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%d')
    ax.set_title(f"Contours (Interval: {interval}m)")
    ax.axis('off')
    
    return fig

def create_normal_map(dem_data, sigma=1.0):
    """
    Create normal map from DEM data for PBR
    """
    # Apply Gaussian filter to smooth the DEM
    smoothed_dem = gaussian_filter(dem_data, sigma=sigma)
    
    # Calculate gradients
    dy, dx = np.gradient(smoothed_dem)
    
    # Create normal map
    z = np.ones_like(smoothed_dem)
    scale = 10.0  # Scale factor for height
    norm = np.sqrt(dx**2 + dy**2 + (z/scale)**2)
    
    nx = dx / norm
    ny = dy / norm
    nz = z / (scale * norm)
    
    # Remap from [-1, 1] to [0, 1] for RGB
    nx_map = (nx + 1) / 2
    ny_map = (ny + 1) / 2
    nz_map = (nz + 1) / 2
    
    # Create RGB normal map
    normal_map = np.zeros((dem_data.shape[0], dem_data.shape[1], 3))
    normal_map[:, :, 0] = nx_map
    normal_map[:, :, 1] = ny_map
    normal_map[:, :, 2] = nz_map
    
    return normal_map

def create_roughness_map(dem_data, kernel_size=5):
    """
    Create roughness map from DEM data for PBR
    """
    # Calculate local mean
    mean = uniform_filter(dem_data, size=kernel_size)
    
    # Calculate roughness as deviation from local mean
    roughness = np.abs(dem_data - mean)
    
    # Normalize to 0-1
    roughness = (roughness - np.min(roughness)) / (np.max(roughness) - np.min(roughness))
    
    return roughness

def create_ambient_occlusion(dem_data, intensity=1.0):
    """
    Create simple ambient occlusion map
    """
    # Calculate gradient magnitude
    sobel_x = sobel(dem_data, axis=0)
    sobel_y = sobel(dem_data, axis=1)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize and invert (higher gradients = darker AO)
    ao = 1.0 - (gradient_magnitude / np.max(gradient_magnitude)) * intensity
    
    # Smooth the result
    ao = gaussian_filter(ao, sigma=1.0)
    
    return ao

def create_3d_surface(dem_data, z_scale=50):
    """
    Create 3D surface visualization from DEM data
    """
    # Create meshgrid for coordinates
    rows, cols = dem_data.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    x_mesh, y_mesh = np.meshgrid(x, y)
    
    # Create Plotly surface
    fig = go.Figure(data=[go.Surface(
        z=dem_data,
        colorscale='earth',
        contours={
            "z": {"show": True, "start": np.min(dem_data), 
                  "end": np.max(dem_data), "size": 100}
        }
    )])
    
    fig.update_layout(
        scene={"aspectratio": {"x": 1, "y": 1, "z": z_scale/100}},
        scene_camera_eye=dict(x=1.5, y=-1.5, z=1),
        width=800, height=600,
        title="3D Terrain Model"
    )
    
    return fig

def get_download_link(img, filename, text):
    """
    Generate a download link for an image
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Tab 1: DEM Download
with tabs[0]:
    st.header("Download Digital Elevation Model (DEM)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Area of Interest")
        south = st.number_input("South Latitude", value=-34.0, min_value=-90.0, max_value=90.0)
        north = st.number_input("North Latitude", value=-33.5, min_value=-90.0, max_value=90.0)
        west = st.number_input("West Longitude", value=18.0, min_value=-180.0, max_value=180.0)
        east = st.number_input("East Longitude", value=18.5, min_value=-180.0, max_value=180.0)
        
        st.subheader("API Key")
        api_key = st.text_input("OpenTopography API Key", type="password", 
                               help="Get your API key from https://opentopography.org/developers")
        
        dem_type = st.selectbox("DEM Type", ["SRTMGL3", "SRTMGL1", "AW3D30", "GEDI"])
        
        if st.button("Download DEM"):
            if api_key:
                dem_file = download_dem(south, north, west, east, api_key, dem_type)
                if dem_file:
                    st.session_state.dem_file = dem_file
                    st.session_state.dem_data, st.session_state.transform, st.session_state.crs = load_dem(dem_file)
                    st.success(f"DEM downloaded successfully! Shape: {st.session_state.dem_data.shape}")
            else:
                st.error("Please enter an API key")
    
    with col2:
        st.subheader("Preview")
        if st.session_state.dem_data is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(st.session_state.dem_data, cmap='terrain')
            plt.colorbar(im, ax=ax, label='Elevation (m)')
            ax.set_title("DEM Elevation")
            ax.axis('off')
            st.pyplot(fig)
            
            st.markdown("### DEM Statistics")
            stats = {
                "Minimum Elevation": float(np.min(st.session_state.dem_data)),
                "Maximum Elevation": float(np.max(st.session_state.dem_data)),
                "Mean Elevation": float(np.mean(st.session_state.dem_data)),
                "Standard Deviation": float(np.std(st.session_state.dem_data))
            }
            st.json(stats)
        else:
            st.info("Download a DEM to see preview")

# Tab 2: Contours
with tabs[1]:
    st.header("Contour Generation")
    
    if st.session_state.dem_data is not None:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            contour_interval = st.slider("Contour Interval (m)", 10, 500, 100, 10)
            
            # Hillshade options
            st.subheader("Hillshade Options")
            show_hillshade = st.checkbox("Show Hillshade", value=True)
            if show_hillshade:
                hillshade_azimuth = st.slider("Light Azimuth (°)", 0, 360, 315, 5)
                hillshade_altitude = st.slider("Light Altitude (°)", 0, 90, 45, 5)
        
        with col2:
            if show_hillshade:
                hillshade = create_hillshade(st.session_state.dem_data, hillshade_azimuth, hillshade_altitude)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(hillshade, cmap='gray')
                contour = ax.contour(st.session_state.dem_data, 
                                    levels=np.arange(np.floor(np.min(st.session_state.dem_data) / contour_interval) * contour_interval,
                                                    np.ceil(np.max(st.session_state.dem_data) / contour_interval) * contour_interval,
                                                    contour_interval),
                                    colors='red', alpha=0.7)
                ax.clabel(contour, inline=True, fontsize=8, fmt='%d')
                ax.set_title(f"Contours with Hillshade (Interval: {contour_interval}m)")
                ax.axis('off')
            else:
                fig = create_contours(st.session_state.dem_data, st.session_state.transform, contour_interval)

            # Save figure to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            
            # Display plot
            st.pyplot(fig)
            
            # Export option
            st.download_button(
                label="Export Contour Map",
                data=buf,
                file_name=f"contour_map_{contour_interval}m.png",
                mime="image/png"
            )
            plt.close(fig)
    else:
        st.info("Please download a DEM first in the 'DEM Download' tab")

# Tab 3: 3D Visualization
with tabs[2]:
    st.header("3D Visualization")
    
    if st.session_state.dem_data is not None:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            z_exaggeration = st.slider("Vertical Exaggeration", 0.1, 5.0, 1.5, 0.1)
            colormap = st.selectbox("Color Scheme", ['earth', 'terrain', 'viridis', 'plasma', 'turbo', 'cividis'])
            show_contours_3d = st.checkbox("Show Contours on 3D", value=True)
            contour_interval_3d = st.slider("3D Contour Interval (m)", 10, 500, 100, 10)
        
        with col2:
            # Create meshgrid for coordinates
            rows, cols = st.session_state.dem_data.shape
            x = np.arange(0, cols)
            y = np.arange(0, rows)
            x_mesh, y_mesh = np.meshgrid(x, y)
            
            # Create Plotly surface
            fig = go.Figure(data=[go.Surface(
                z=st.session_state.dem_data,
                colorscale=colormap
            )])
            
            if show_contours_3d:
                fig.update_traces(
                    contours={
                        "z": {"show": True, "start": np.min(st.session_state.dem_data), 
                              "end": np.max(st.session_state.dem_data), 
                              "size": contour_interval_3d}
                    }
                )
            
            fig.update_layout(
                scene={"aspectratio": {"x": 1, "y": 1, "z": z_exaggeration}},
                scene_camera_eye=dict(x=1.5, y=-1.5, z=1),
                width=800, height=600,
                title="3D Terrain Model"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please download a DEM first in the 'DEM Download' tab")

# Tab 4: PBR Maps
with tabs[3]:
    st.header("PBR Map Generation")
    
    if st.session_state.dem_data is not None:
        st.markdown("""
        Generate Physically Based Rendering (PBR) maps for use in Blender or other 3D software.
        These maps can be used to create realistic terrain materials.
        """)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Settings")
            normal_smoothing = st.slider("Normal Map Smoothing", 0.0, 5.0, 1.0, 0.1)
            roughness_kernel = st.slider("Roughness Kernel Size", 3, 15, 5, 2)
            ao_intensity = st.slider("AO Intensity", 0.1, 3.0, 1.0, 0.1)
            
            generate_button = st.button("Generate PBR Maps")
        
        with col2:
            if generate_button:
                with st.spinner("Generating PBR maps..."):
                    # Create PBR maps
                    normal_map = create_normal_map(st.session_state.dem_data, sigma=normal_smoothing)
                    roughness_map = create_roughness_map(st.session_state.dem_data, kernel_size=roughness_kernel)
                    height_map = (st.session_state.dem_data - np.min(st.session_state.dem_data)) / (np.max(st.session_state.dem_data) - np.min(st.session_state.dem_data))
                    ao_map = create_ambient_occlusion(st.session_state.dem_data, intensity=ao_intensity)
                    
                    # Display the maps
                    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    
                    axs[0, 0].imshow(normal_map)
                    axs[0, 0].set_title("Normal Map")
                    axs[0, 0].axis('off')
                    
                    axs[0, 1].imshow(roughness_map, cmap='gray')
                    axs[0, 1].set_title("Roughness Map")
                    axs[0, 1].axis('off')
                    
                    axs[1, 0].imshow(height_map, cmap='gray')
                    axs[1, 0].set_title("Height Map")
                    axs[1, 0].axis('off')
                    
                    axs[1, 1].imshow(ao_map, cmap='gray')
                    axs[1, 1].set_title("Ambient Occlusion Map")
                    axs[1, 1].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Create download buttons for each map
                    st.subheader("Download Maps")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # Convert numpy array to PIL Image
                        normal_img = Image.fromarray((normal_map * 255).astype(np.uint8))
                        normal_bytes = io.BytesIO()
                        normal_img.save(normal_bytes, format='PNG')
                        normal_bytes = normal_bytes.getvalue()
                        
                        st.download_button(
                            label="Normal Map",
                            data=normal_bytes,
                            file_name="normal_map.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        roughness_img = Image.fromarray((roughness_map * 255).astype(np.uint8))
                        roughness_bytes = io.BytesIO()
                        roughness_img.save(roughness_bytes, format='PNG')
                        roughness_bytes = roughness_bytes.getvalue()
                        
                        st.download_button(
                            label="Roughness Map",
                            data=roughness_bytes,
                            file_name="roughness_map.png",
                            mime="image/png"
                        )
                    
                    with col3:
                        height_img = Image.fromarray((height_map * 255).astype(np.uint8))
                        height_bytes = io.BytesIO()
                        height_img.save(height_bytes, format='PNG')
                        height_bytes = height_bytes.getvalue()
                        
                        st.download_button(
                            label="Height Map",
                            data=height_bytes,
                            file_name="height_map.png",
                            mime="image/png"
                        )
                    
                    with col4:
                        ao_img = Image.fromarray((ao_map * 255).astype(np.uint8))
                        ao_bytes = io.BytesIO()
                        ao_img.save(ao_bytes, format='PNG')
                        ao_bytes = ao_bytes.getvalue()
                        
                        st.download_button(
                            label="AO Map",
                            data=ao_bytes,
                            file_name="ao_map.png",
                            mime="image/png"
                        )
                        
                    # Provide code for Blender
                    st.subheader("Blender Import Code")
                    blender_code = """
                    # Blender Python code to import these maps
                    import bpy
                    
                    # Create a new material
                    mat = bpy.data.materials.new(name="Terrain_Material")
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    
                    # Clear default nodes
                    for node in nodes:
                        nodes.remove(node)
                    
                    # Create PBR shader
                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (0, 0)
                    
                    # Create output node
                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (300, 0)
                    
                    # Link shader to output
                    links = mat.node_tree.links
                    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
                    
                    # Create texture nodes
                    normal_tex = nodes.new(type='ShaderNodeTexImage')
                    normal_tex.name = 'Normal Map'
                    normal_tex.location = (-600, 0)
                    
                    # Create normal map node
                    normal_map = nodes.new(type='ShaderNodeNormalMap')
                    normal_map.location = (-300, 0)
                    
                    # Link normal map
                    links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
                    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                    
                    # Load the normal map image
                    normal_tex.image = bpy.data.images.load("path/to/normal_map.png")
                    
                    # Create roughness texture
                    rough_tex = nodes.new(type='ShaderNodeTexImage')
                    rough_tex.name = 'Roughness Map'
                    rough_tex.location = (-600, -300)
                    
                    # Link roughness map
                    links.new(rough_tex.outputs['Color'], principled.inputs['Roughness'])
                    
                    # Load the roughness map image
                    rough_tex.image = bpy.data.images.load("path/to/roughness_map.png")
                    
                    # Create height texture for displacement
                    height_tex = nodes.new(type='ShaderNodeTexImage')
                    height_tex.name = 'Height Map'
                    height_tex.location = (-600, -600)
                    
                    # Create displacement node
                    disp = nodes.new(type='ShaderNodeDisplacement')
                    disp.location = (0, -300)
                    
                    # Link height map to displacement
                    links.new(height_tex.outputs['Color'], disp.inputs['Height'])
                    links.new(disp.outputs['Displacement'], output.inputs['Displacement'])
                    
                    # Load the height map image
                    height_tex.image = bpy.data.images.load("path/to/height_map.png")
                    
                    # Create AO texture
                    ao_tex = nodes.new(type='ShaderNodeTexImage')
                    ao_tex.name = 'AO Map'
                    ao_tex.location = (-600, 300)
                    
                    # Create mix node for AO
                    mix = nodes.new(type='ShaderNodeMixRGB')
                    mix.blend_type = 'MULTIPLY'
                    mix.inputs[0].default_value = 0.8  # Factor
                    mix.location = (-300, 300)
                    
                    # Link AO map
                    links.new(ao_tex.outputs['Color'], mix.inputs[1])
                    
                    # Load the AO map image
                    ao_tex.image = bpy.data.images.load("path/to/ao_map.png")
                    """
                    
                    st.code(blender_code, language="python")
    else:
        st.info("Please download a DEM first in the 'DEM Download' tab")

# Footer
st.markdown("---")
st.markdown("### About this application")
st.markdown("""
This Terrain Analysis Tool allows you to download and process Digital Elevation Models (DEMs).
You can generate contours, create 3D visualizations, and export PBR maps for use in 3D software.

To use this app:
1. Obtain an API key from [OpenTopography](https://opentopography.org/developers)
2. Enter geographic coordinates for your area of interest
3. Download DEM data
4. Explore different visualizations in the tabs
5. Export maps for use in Blender or other 3D software
""")
