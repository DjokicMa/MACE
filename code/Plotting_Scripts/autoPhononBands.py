import os, sys, glob, re
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
from matplotlib import rc

=======
from matplotlib import rc 
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)

class PhononBandPlotter:
    def __init__(self):
        self.conversion_factor = 0.029979  # conversion to THz
<<<<<<< HEAD

    def parse_d12_file(self, material):
        """Parse the .d12 file to extract k-point path information"""
        d12_file = material + ".d12"
        if not os.path.exists(d12_file):
            print(f"Warning: {d12_file} not found.")
            return self.prompt_for_kpoint_path()

        kpoint_labels = []
        kpoint_coords = []

        try:
            with open(d12_file, "r") as f:
                lines = f.readlines()

=======
        
    def parse_d12_file(self, material):
        """Parse the .d12 file to extract k-point path information"""
        d12_file = material + '.d12'
        if not os.path.exists(d12_file):
            print(f"Warning: {d12_file} not found.")
            return self.prompt_for_kpoint_path()
            
        kpoint_labels = []
        kpoint_coords = []
        
        try:
            with open(d12_file, 'r') as f:
                lines = f.readlines()
                
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
            # Look for BAND keyword and subsequent k-point definitions
            in_band_section = False
            for i, line in enumerate(lines):
                line = line.strip().upper()
<<<<<<< HEAD

                if "BAND" in line:
                    in_band_section = True
                    continue

=======
                
                if 'BAND' in line:
                    in_band_section = True
                    continue
                    
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
                if in_band_section:
                    # Look for k-point definitions
                    # Format can be: label x y z or just coordinates
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            # Try to parse as coordinates
                            coords = [float(x) for x in parts[-3:]]
                            kpoint_coords.append(coords)
<<<<<<< HEAD

                            # Check if there's a label
                            if (
                                len(parts) > 3
                                and not parts[0]
                                .replace(".", "")
                                .replace("-", "")
                                .isdigit()
                            ):
                                kpoint_labels.append(parts[0])
                            else:
                                kpoint_labels.append(f"K{len(kpoint_labels)}")

=======
                            
                            # Check if there's a label
                            if len(parts) > 3 and not parts[0].replace('.','').replace('-','').isdigit():
                                kpoint_labels.append(parts[0])
                            else:
                                kpoint_labels.append(f"K{len(kpoint_labels)}")
                                
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
                        except ValueError:
                            # If we can't parse coordinates, we might be done with k-points
                            if len(kpoint_coords) > 0:
                                break
<<<<<<< HEAD
                    elif line == "" or "END" in line:
                        break

        except Exception as e:
            print(f"Error parsing {d12_file}: {e}")
            return self.prompt_for_kpoint_path()

=======
                    elif line == '' or 'END' in line:
                        break
                        
        except Exception as e:
            print(f"Error parsing {d12_file}: {e}")
            return self.prompt_for_kpoint_path()
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
        # Check if we successfully parsed k-point labels
        if not kpoint_labels or len(kpoint_labels) < 2:
            print(f"Could not extract k-point path from {d12_file}")
            return self.prompt_for_kpoint_path()
<<<<<<< HEAD

=======
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
        return kpoint_labels, kpoint_coords

    def prompt_for_kpoint_path(self):
        """Prompt user to enter k-point path manually"""
        print("\nPlease enter your k-point path.")
        print("Enter k-point labels separated by spaces (e.g., 'Gamma X S Y Gamma')")
        print("Common labels: Gamma (or G), X, Y, Z, L, M, K, H, A, S, R, T, U, W")
        print("Or press Enter to use default labels K0, K1, K2, etc.")
<<<<<<< HEAD

        try:
            user_input = input("K-point path: ").strip()

            if not user_input:
                return None, None

=======
        
        try:
            user_input = input("K-point path: ").strip()
            
            if not user_input:
                return None, None
                
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
            # Parse user input
            labels = user_input.split()
            if len(labels) < 2:
                print("Warning: Need at least 2 k-points. Using default labels.")
                return None, None
<<<<<<< HEAD

            print(f"Using k-point path: {' → '.join(labels)}")
            return labels, None

=======
                
            print(f"Using k-point path: {' → '.join(labels)}")
            return labels, None
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
        except (KeyboardInterrupt, EOFError):
            print("\nUsing default labels.")
            return None, None

    def get_bands(self, material):
        """Parse the .f25 file to extract eigenvalues and k-point information"""
<<<<<<< HEAD
        file = material + ".f25"

        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found")

        with open(file) as f:
            numpattern = r"[-+]?\d\.\d+E[-+]?\d+"
            header = f.readline()

            # Parse header information
            header_parts = header.split()
            N = int(header_parts[2])  # number of points along path
            M = int(header_parts[1])  # number of bands

=======
        file = material + '.f25'
        
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found")
            
        with open(file) as f:
            numpattern = r'[-+]?\d\.\d+E[-+]?\d+'
            header = f.readline()
            
            # Parse header information
            header_parts = header.split()
            N = int(header_parts[2])  # number of points along path
            M = int(header_parts[1])  # number of bands 
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
            eigenvalues = []
            k_values = [0]
            num_kvalues = []
            n = 0
            startline = 0

<<<<<<< HEAD
            f.seek(0)
=======
            f.seek(0) 
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
            for i, line in enumerate(f):
                if line.startswith("-%-"):
                    startline = i
                    eigenvalues.append([])
                    kvalue = float(line.split()[-2])
                    k_values.append(kvalue + k_values[n])
                    num_kvalues.append(int(line.split()[2]))
                    n += 1
                elif i > startline + 2:
                    numbers = re.findall(numpattern, line)
                    for x in numbers:
<<<<<<< HEAD
                        eigenvalues[n - 1].append(float(x) * self.conversion_factor)

=======
                        eigenvalues[n-1].append(float(x) * self.conversion_factor)
                        
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
        return eigenvalues, k_values, num_kvalues, N, M

    def group_list(self, lst, n):
        """Group eigenvalues by k-point"""
<<<<<<< HEAD
        return [
            [sublist[i : i + n] for i in range(0, len(sublist), n)] for sublist in lst
        ]
=======
        return [[sublist[i:i + n] for i in range(0, len(sublist), n)] for sublist in lst]
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)

    def organize_bands(self, eigenvalue_list, num_bands):
        """Organize eigenvalues into bands"""
        bands = [[] for _ in range(num_bands)]
        for i in eigenvalue_list:
            for j in i:
                for k in range(len(j)):
                    bands[k].append(j[k])
        return bands

    def format_label(self, label):
        """Format k-point labels for matplotlib"""
        # Common high-symmetry point formatting
        label_map = {
<<<<<<< HEAD
            "GAMMA": r"$\Gamma$",
            "G": r"$\Gamma$",
            "X": "X",
            "Y": "Y",
            "Z": "Z",
            "L": "L",
            "M": "M",
            "K": "K",
            "H": "H",
            "A": "A",
            "S": "S",
            "R": "R",
            "T": "T",
            "U": "U",
            "W": "W",
        }

        upper_label = label.upper()
        return label_map.get(upper_label, label)

    def plot_bands(
        self,
        material,
        bands,
        k_values,
        num_kvalues,
        kpoint_labels=None,
        figsize=(8, 6),
        save_format="pdf",
    ):
        """Plot the phonon band structure"""

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
        rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

        # Create x-axis coordinates
        x_axis = []
        for i in range(len(num_kvalues)):
            x_axis.append(np.linspace(k_values[i], k_values[i + 1], num_kvalues[i]))
        combined_x_axis = np.concatenate(x_axis)

        # Plot vertical lines at high-symmetry points
        for k_val in k_values:
            plt.axvline(k_val, color="k", linewidth=0.5)

        # Plot bands
        for i, band in enumerate(bands):
            plt.plot(combined_x_axis, band, color="r", linewidth=0.7, alpha=0.6)

        # Add horizontal line at zero frequency
        plt.axhline(0, color="k", linestyle="--", alpha=0.7)

        # Set labels and formatting
        ax1.set_xlabel("Wave vector", fontsize=12)
        ax1.set_ylabel("Frequency (THz)", fontsize=12)
        ax1.set_xticks(k_values)

        # Set k-point labels
        if kpoint_labels and len(kpoint_labels) == len(k_values):
            # Check if labels are just default K0, K1, K2... format
            has_meaningful_labels = any(
                not label.startswith("K") or not label[1:].isdigit()
                for label in kpoint_labels
                if len(label) > 1
            )

=======
            'GAMMA': r'$\Gamma$',
            'G': r'$\Gamma$',
            'X': 'X',
            'Y': 'Y', 
            'Z': 'Z',
            'L': 'L',
            'M': 'M',
            'K': 'K',
            'H': 'H',
            'A': 'A',
            'S': 'S',
            'R': 'R',
            'T': 'T',
            'U': 'U',
            'W': 'W'
        }
        
        upper_label = label.upper()
        return label_map.get(upper_label, label)

    def plot_bands(self, material, bands, k_values, num_kvalues, 
                   kpoint_labels=None, figsize=(8, 6), save_format='pdf'):
        """Plot the phonon band structure"""
        
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
        
        # Create x-axis coordinates
        x_axis = []
        for i in range(len(num_kvalues)):
            x_axis.append(np.linspace(k_values[i], k_values[i+1], num_kvalues[i]))
        combined_x_axis = np.concatenate(x_axis)
        
        # Plot vertical lines at high-symmetry points
        for k_val in k_values:
            plt.axvline(k_val, color='k', linewidth=0.5)
        
        # Plot bands
        for i, band in enumerate(bands):
            plt.plot(combined_x_axis, band, color='r', linewidth=0.7, alpha=0.6)
        
        # Add horizontal line at zero frequency
        plt.axhline(0, color='k', linestyle='--', alpha=0.7)
        
        # Set labels and formatting
        ax1.set_xlabel('Wave vector', fontsize=12)
        ax1.set_ylabel('Frequency (THz)', fontsize=12)
        ax1.set_xticks(k_values)
        
        # Set k-point labels
        if kpoint_labels and len(kpoint_labels) == len(k_values):
            # Check if labels are just default K0, K1, K2... format
            has_meaningful_labels = any(not label.startswith('K') or not label[1:].isdigit() 
                                      for label in kpoint_labels if len(label) > 1)
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
            if has_meaningful_labels:
                formatted_labels = [self.format_label(label) for label in kpoint_labels]
                ax1.set_xticklabels(formatted_labels)
            else:
                # Labels are just K0, K1, K2... so treat as no meaningful labels
                kpoint_labels = None
<<<<<<< HEAD

        if not kpoint_labels or len(kpoint_labels) != len(k_values):
            if kpoint_labels and len(kpoint_labels) != len(k_values):
                print(
                    f"Warning: Number of provided labels ({len(kpoint_labels)}) doesn't match number of k-points ({len(k_values)})"
                )

            # No meaningful labels provided, use defaults for now
            # This will be caught by the calling function to prompt user
            default_labels = [f"K{i}" for i in range(len(k_values))]
            ax1.set_xticklabels(default_labels)

        ax1.tick_params(labelsize=10)
        ax1.set_xlim(min(k_values), max(k_values))
        ax1.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        # Save the figure
        output_file = f"{material}_phonon_bands.{save_format}"
        fig.savefig(output_file, format=save_format, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_file}")

        # Return whether meaningful labels were used
        has_meaningful_labels = kpoint_labels is not None
        if kpoint_labels:
            has_meaningful_labels = any(
                not label.startswith("K") or not label[1:].isdigit()
                for label in kpoint_labels
                if len(label) > 1
            )

        return fig, ax1, has_meaningful_labels

    def process_material(
        self, material, figsize=(8, 6), save_format="pdf", interactive=True
    ):
        """Process a single material and create its band structure plot"""
        print(f"Processing material: {material}")

=======
                
        if not kpoint_labels or len(kpoint_labels) != len(k_values):
            if kpoint_labels and len(kpoint_labels) != len(k_values):
                print(f"Warning: Number of provided labels ({len(kpoint_labels)}) doesn't match number of k-points ({len(k_values)})")
            
            # No meaningful labels provided, use defaults for now
            # This will be caught by the calling function to prompt user
            default_labels = [f'K{i}' for i in range(len(k_values))]
            ax1.set_xticklabels(default_labels)
        
        ax1.tick_params(labelsize=10)
        ax1.set_xlim(min(k_values), max(k_values))
        ax1.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_file = f"{material}_phonon_bands.{save_format}"
        fig.savefig(output_file, format=save_format, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        
        # Return whether meaningful labels were used
        has_meaningful_labels = kpoint_labels is not None
        if kpoint_labels:
            has_meaningful_labels = any(not label.startswith('K') or not label[1:].isdigit() 
                                      for label in kpoint_labels if len(label) > 1)
        
        return fig, ax1, has_meaningful_labels

    def process_material(self, material, figsize=(8, 6), save_format='pdf', interactive=True):
        """Process a single material and create its band structure plot"""
        print(f"Processing material: {material}")
        
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
        try:
            # Parse input file for k-point information
            if interactive:
                kpoint_labels, kpoint_coords = self.parse_d12_file(material)
            else:
                # Non-interactive mode - just try to parse d12, don't prompt
<<<<<<< HEAD
                d12_file = material + ".d12"
=======
                d12_file = material + '.d12'
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
                if os.path.exists(d12_file):
                    kpoint_labels, kpoint_coords = self.parse_d12_file(material)
                else:
                    kpoint_labels, kpoint_coords = None, None
<<<<<<< HEAD

            # Get band structure data
            eigenvalues, k_values, num_kvalues, N, M = self.get_bands(material)

            # Organize the data
            grouped_eigenvalues = self.group_list(eigenvalues, M)
            bands = self.organize_bands(grouped_eigenvalues, M)

            # Create initial plot to check if we have meaningful labels
            fig, ax, has_meaningful_labels = self.plot_bands(
                material,
                bands,
                k_values,
                num_kvalues,
                kpoint_labels,
                figsize,
                save_format,
            )

=======
            
            # Get band structure data
            eigenvalues, k_values, num_kvalues, N, M = self.get_bands(material)
            
            # Organize the data
            grouped_eigenvalues = self.group_list(eigenvalues, M)
            bands = self.organize_bands(grouped_eigenvalues, M)
            
            # Create initial plot to check if we have meaningful labels
            fig, ax, has_meaningful_labels = self.plot_bands(material, bands, k_values, num_kvalues, 
                                                           kpoint_labels, figsize, save_format)
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
            # If we don't have meaningful labels and we're in interactive mode, prompt user
            if not has_meaningful_labels and interactive:
                plt.close(fig)  # Close the plot with default labels
                print(f"\nNo meaningful k-point labels found for {material}.")
                user_labels, _ = self.prompt_for_kpoint_path()
<<<<<<< HEAD

                # Recreate plot with user-provided labels
                fig, ax, _ = self.plot_bands(
                    material,
                    bands,
                    k_values,
                    num_kvalues,
                    user_labels,
                    figsize,
                    save_format,
                )
                kpoint_labels = user_labels

=======
                
                # Recreate plot with user-provided labels
                fig, ax, _ = self.plot_bands(material, bands, k_values, num_kvalues, 
                                           user_labels, figsize, save_format)
                kpoint_labels = user_labels
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
            print(f"Successfully processed {material}")
            print(f"  Number of bands: {M}")
            print(f"  Number of k-point segments: {len(num_kvalues)}")
            print(f"  Total k-points: {sum(num_kvalues)}")
            if kpoint_labels:
                print(f"  K-point path: {' → '.join(kpoint_labels)}")
            else:
<<<<<<< HEAD
                print(f"  Using default labels: K0 → K1 → ... → K{len(k_values) - 1}")

            return fig, ax

=======
                print(f"  Using default labels: K0 → K1 → ... → K{len(k_values)-1}")
            
            return fig, ax
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
        except Exception as e:
            print(f"Error processing {material}: {e}")
            return None, None

<<<<<<< HEAD
    def process_all_materials(
        self, directory=None, figsize=(8, 6), save_format="pdf", interactive=True
    ):
        """Process all .f25 files in the specified directory"""
        if directory is None:
            directory = os.getcwd()

        search_pattern = os.path.join(directory, "*.f25")
        pathlist = glob.glob(search_pattern)

        if not pathlist:
            print(f"No .f25 files found in {directory}")
            return

=======
    def process_all_materials(self, directory=None, figsize=(8, 6), save_format='pdf', interactive=True):
        """Process all .f25 files in the specified directory"""
        if directory is None:
            directory = os.getcwd()
            
        search_pattern = os.path.join(directory, '*.f25')
        pathlist = glob.glob(search_pattern)
        
        if not pathlist:
            print(f"No .f25 files found in {directory}")
            return
            
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
        processed_count = 0
        for i, path in enumerate(pathlist):
            path_str = str(path)
            material = os.path.splitext(os.path.basename(path_str))[0]
<<<<<<< HEAD

            if material:
                print(f"\n--- Processing {i + 1}/{len(pathlist)}: {material} ---")
                fig, ax = self.process_material(
                    material, figsize, save_format, interactive
                )
                if fig is not None:
                    processed_count += 1

=======
            
            if material:
                print(f"\n--- Processing {i+1}/{len(pathlist)}: {material} ---")
                fig, ax = self.process_material(material, figsize, save_format, interactive)
                if fig is not None:
                    processed_count += 1
                    
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
        print(f"\nProcessed {processed_count}/{len(pathlist)} materials successfully")


def main():
    """Main function to run the phonon band structure plotter"""
    plotter = PhononBandPlotter()
<<<<<<< HEAD

    # Process all materials in current directory
    plotter.process_all_materials(figsize=(8, 6), save_format="pdf")

=======
    
    # Process all materials in current directory
    plotter.process_all_materials(figsize=(8, 6), save_format='pdf')
    
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
    # Alternative: process a specific material
    # plotter.process_material('your_material_name', figsize=(10, 8), save_format='png')


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> bf47fddb (Automatic phonon plotting based on f25 and d12)
