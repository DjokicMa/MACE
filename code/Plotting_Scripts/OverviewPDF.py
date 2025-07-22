#!/usr/bin/python3
"""
Structure Visualization and Analysis Report Generator

Overview:
This script generates a consolidated PDF report containing visualizations of 
crystal structures, their band diagrams (BAND plots), and density of states (DOS)
plots. It organizes these elements into a structured document with multiple views.

Features:
- Automatically discovers and groups related files from specified directories
- Creates a multi-page PDF document with organized layouts
- Includes crystal structure visualizations from different angles
- Embeds band diagram and DOS plots alongside structural information

Author: Marcus Djokic
Institution: Michigan State University, Mendoza Group

Directories Setup:
views_dir: Path to directory containing crystal view PNG images (e.g., "view0.png")
bands_dir: Directory for BAND plot files ending with ".BANDS.png"
dos_dir: Directory for DOSS plot files ending with ".DOSS.png"
output_pdf: Output PDF file name and path

Usage:
1. Update the directories in views_dir, bands_dir, dos_dir
2. Run the script to generate output PDF at specified location
3. The generated PDF will include one page per structure containing all available data

Structure of Generated Pages:
- Each page is titled with the base name of the structure
- Contains up to three different crystal structure visualizations (views)
- Includes band diagram and DOS plots if they exist for the structure

Requirements:
Python 3.8+
fpdf, Pillow, os libraries installed
Input files organized by their naming conventions as follows:
Structure Name_view0.png, StructureName.BANDS.png etc.

Notes:
The script automatically arranges images based on file names.
Files are matched by base name after removing view numbers and plot types.
Ensure consistent naming of input files for proper grouping."
"""


from fpdf import FPDF
import os
from PIL import Image

# Define directories for your image files
views_dir = "/home/marcus/Documents/Comp/CAllotropes/Sacada/Crystal Outputs/Opt1/PNGs/"
bands_dir = "BAND/Plots/"
dos_dir = "DOSS/Plots/"
output_pdf = "combined_structures.pdf"

class PDF(FPDF):
    def add_structure_page(self, structure_name, files):
        # Add a page
        self.add_page()

        # Title
        self.set_font("Helvetica", style='B', size=18)
        self.cell(0, 10, structure_name, ln=True, align='C')

        # Title spacing
        y_offset_title = 60

        # Band and DOS plots
        y_offset_bands_dos = 80
        if files["band"]:
            band_path = os.path.join(bands_dir, files["band"])
            if os.path.exists(band_path):
                print(f"Processing band file: {band_path}")
                img = Image.open(band_path)
                band_aspect_ratio = img.width / img.height
                img_width = 95  # Width for each image
                img_height = img_width / band_aspect_ratio
                img = img.resize((img_width, int(img_height)))
                self.image(band_path, x=10, y=y_offset_bands_dos, w=95, h=img_height)
            else:
                # Draw empty white space where band image would be
                self.set_fill_color(255, 255, 255)  # White
                self.rect(10, y_offset_bands_dos, 95, 100, style='F')

        if files["dos"]:
            dos_path = os.path.join(dos_dir, files["dos"])
            if os.path.exists(dos_path):
                print(f"Processing DOS file: {dos_path}")
                img = Image.open(dos_path)
                dos_aspect_ratio = img.width / img.height
                img_width = 95  # Width for each image
                img_height = img_width / dos_aspect_ratio
                img = img.resize((img_width, int(img_height)))
                self.image(dos_path, x=105, y=y_offset_bands_dos, w=95, h=img_height)
            else:
                # Draw empty white space where DOS image would be
                self.set_fill_color(255, 255, 255)  # White
                self.rect(105, y_offset_bands_dos, 95, 100, style='F')

        # Draw the first three views directly underneath the title
        y_offset_views = y_offset_title - 50
        if files["band"] or files["dos"]:
            view_files = sorted(files["views"])
            if len(view_files) >= 3:  # We need exactly three views
                for i, view_file in enumerate(view_files[:3]):
                    view_path = os.path.join(views_dir, view_file)
                    if os.path.exists(view_path):
                        print(f"Processing view file: {view_path}")
                        img = Image.open(view_path)
                        aspect_ratio = img.width / img.height
                        img_width = 75  # Width for each image
                        img_height = img_width / aspect_ratio
                        img = img.resize((img_width, int(img_height)))
                        self.image(view_path, x=0 + i * (img_width-5), y=y_offset_views, w=img_width, h=img_height)
                    else:
                        # Draw empty white space
                        self.set_fill_color(255, 255, 255)  # White
                        self.rect(0 + i * (img_width-5), y_offset_views, img_width, img_height, style='F')

def find_matching_files():
    """Find and group files belonging to the same structure."""
    all_files = set(os.listdir(views_dir)) | set(os.listdir(bands_dir)) | set(os.listdir(dos_dir))
    structures = {}

    for file in all_files:
        if file.endswith(".png"):
            try:
                base = file.split("_", 1)[0] if "_" in file else file
                base = base.replace("view0", "").replace("view1", "").replace("view2", "")
                if base not in structures:
                    structures[base] = {"views": [], "band": None, "dos": None}

                if "view" in file:
                    structures[base]["views"].append(file)
                elif file.endswith(".BANDS.png"):
                    structures[base]["band"] = file
                elif file.endswith(".DOSS.png"):
                    structures[base]["dos"] = file
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    return structures

def create_pdf():
    """Generate the PDF document."""
    structures = find_matching_files()
    pdf = PDF()

    for structure_name, files in structures.items():
        if files["band"] or files["dos"]:  # Only include if BAND or DOS exists
            print(f"Processing structure: {structure_name}")
            pdf.add_structure_page(structure_name, files)

    pdf.output(output_pdf)

if __name__ == "__main__":
    try:
        create_pdf()
        print(f"PDF created: {output_pdf}")
    except Exception as e:
        print(f"Error creating PDF: {e}")
