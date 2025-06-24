def _extract_functional(self, lines):
    """Extract DFT functional including 3C methods - IMPROVED VERSION"""
    
    # Mapping from CRYSTAL output format to common functional names
    FUNCTIONAL_MAPPING = {
        # Exchange-Correlation pairs to functional names
        ("WU-COHEN GGA", "PERDEW-WANG GGA"): "B1WC",
        ("BECKE 88", "PERDEW-WANG GGA"): "B3PW",
        ("B97-3c", "B97-3c"): "B97-3C",
        ("B97-D GGA + GRIMME D2", "B97-D"): "B97-D3",
        ("BECKE 97", "BECKE 97"): "B97H",
        ("BECKE 88", "LEE-YANG-PARR"): "BLYP",
        ("CAM-B3LYP", "LEE-YANG-PARR"): "CAM-B3LYP",
        ("HISS MR/GGA", "PERDEW-BURKE-ERNZERHOF"): "HISS",
        ("HSE-3c", "PBEh-3c"): "HSE-3C",
        ("HSEsol GGA", "PBEsol"): "HSEsol",
        ("LC-BLYP_X", "LEE-YANG-PARR"): "LC-BLYP",
        ("LC-PBE_X", "PERDEW-BURKE-ERNZERHOF"): "LC-PBE",
        ("HJS B88 SR/GGA", "LEE-YANG-PARR"): "LC-wBLYP",
        ("HJS PBE SR/GGA", "PERDEW-BURKE-ERNZERHOF"): "LC-wPBE",
        ("HJS PBEsol SR/GGA", "PBEsol"): "LC-wPBEsol",
        ("M05-2X", "M05-2X"): "M052X",
        ("M05", "M05"): "M05",
        ("M06-2X", "M06-2X"): "M062X",
        ("M06", "M06"): "M06",
        ("M06-HF", "M06-HF"): "M06HF",
        ("M06-L", "M06-L"): "M06L",
        ("MODIFIED PERDEW-WANG 91", "PERDEW-WANG GGA"): "mPW1K",
        ("PERDEW-WANG GGA", "PERDEW-WANG GGA"): "mPW1PW91",
        ("PERDEW-BURKE-ERNZERHOF", "PERDEW-BURKE-ERNZERHOF"): "PBE",
        ("PBEh-3c", "PBEh-3c"): "PBEh-3C",
        ("PBEsol GGA", "PBEsol"): "PBEsol",
        ("r2SCAN", "r2SCAN"): "r2SCAN",
        ("revM06-L", "revM06-L"): "revM06L",
        ("revM06", "revM06"): "revM06",
        ("RSHXLDA (SR-LDA/LR-HF)", "VOSKO-WILK-NUSAIR"): "RSHXLDA",
        ("SCAN", "SCAN"): "SCAN",
        ("SC-BLYP_X", "LEE-YANG-PARR"): "SC-BLYP",
        ("SOGGA11X", "SOGGA11X"): "SOGGA11X",
        ("DIRAC-SLATER LDA", "VOSKO-WILK-NUSAIR"): "SVWN",
        ("wB97 (SR-B97/LR-HF)", "wB97"): "wB97",
        ("wB97-X (SR-B97/LR+SR HF)", "wB97-X"): "wB97X",
        ("WU-COHEN GGA", "LEE-YANG-PARR"): "WC1LYP",
    }
    
    # First check for Hartree-Fock methods
    for i, line in enumerate(lines):
        if "TYPE OF CALCULATION" in line:
            # Check current line and next few lines for KOHN-SHAM
            is_dft = False
            for j in range(i, min(i + 5, len(lines))):
                if "KOHN-SHAM" in lines[j]:
                    is_dft = True
                    break
            
            # Only assign HF if it's definitely not DFT
            if not is_dft:
                if "RESTRICTED CLOSED SHELL" in line:
                    self.data["functional"] = "RHF"
                    return
                elif "UNRESTRICTED OPEN SHELL" in line:
                    self.data["functional"] = "UHF"
                    return

    # Check for 3C methods early
    for line in lines:
        if "HF-3C" in line or "HF3C" in line:
            self.data["functional"] = "HF-3C"
            self.data["is_3c_method"] = True
            return
        elif "PBEH-3C" in line or "PBEH3C" in line:
            self.data["functional"] = "PBEh-3C"
            self.data["is_3c_method"] = True
            return
        elif "HSE-3C" in line or "HSE3C" in line:
            self.data["functional"] = "HSE-3C"
            self.data["is_3c_method"] = True
            return
        elif "B97-3C" in line or "B973C" in line:
            self.data["functional"] = "B97-3C"
            self.data["is_3c_method"] = True
            return
        elif "HFSOL-3C" in line or "HFSOL3C" in line:
            self.data["functional"] = "HFsol-3C"
            self.data["is_3c_method"] = True
            return
        elif "PBESOL0-3C" in line or "PBESOL03C" in line:
            self.data["functional"] = "PBEsol0-3C"
            self.data["is_3c_method"] = True
            return
        elif "HSESOL-3C" in line or "HSESOL3C" in line:
            self.data["functional"] = "HSEsol-3C"
            self.data["is_3c_method"] = True
            return

    # Look for the standard (EXCHANGE)[CORRELATION] FUNCTIONAL: pattern
    functional_found = False
    for i, line in enumerate(lines):
        if "(EXCHANGE)[CORRELATION] FUNCTIONAL:" in line:
            # Extract the exchange and correlation parts
            match = re.search(r'\(([^)]+)\)\[([^\]]+)\]\s*FUNCTIONAL:\s*\(([^)]+)\)\[([^\]]+)\]', line)
            if match:
                exchange_func = match.group(3).strip()
                correlation_func = match.group(4).strip()
                
                # Look up in mapping
                func_pair = (exchange_func, correlation_func)
                if func_pair in FUNCTIONAL_MAPPING:
                    self.data["functional"] = FUNCTIONAL_MAPPING[func_pair]
                    functional_found = True
                else:
                    # Handle special cases with exact percentage checking
                    # Check if it's a hybrid functional
                    for j in range(max(0, i-5), min(len(lines), i+10)):
                        if "HYBRID EXCHANGE" in lines[j]:
                            # Check the percentage
                            percent_match = re.search(r'(\d+\.?\d*)\s*%', lines[j])
                            if percent_match:
                                percentage = float(percent_match.group(1))
                                if exchange_func == "PERDEW-BURKE-ERNZERHOF" and percentage == 25.0:
                                    self.data["functional"] = "PBE0"
                                    functional_found = True
                                elif exchange_func == "PERDEW-BURKE-ERNZERHOF" and percentage == 13.0:
                                    self.data["functional"] = "PBE0-13"
                                    functional_found = True
                                elif exchange_func == "PBEsol GGA" and percentage == 25.0:
                                    self.data["functional"] = "PBEsol0"
                                    functional_found = True
                                elif exchange_func == "r2SCAN":
                                    if percentage == 0.0:
                                        self.data["functional"] = "r2SCAN0"
                                    elif percentage == 50.0:
                                        self.data["functional"] = "r2SCAN50"
                                    elif percentage == 10.0:
                                        self.data["functional"] = "r2SCANh"
                                    functional_found = True
                                elif exchange_func == "SCAN" and percentage == 0.0:
                                    self.data["functional"] = "SCAN0"
                                    functional_found = True
                                break
                    
                    # If still not found, try simpler mapping
                    if not functional_found:
                        # Handle B3LYP special case
                        if exchange_func == "BECKE 88" and correlation_func == "LEE-YANG-PARR":
                            # Check if it's B3LYP (with hybrid exchange) or BLYP
                            for j in range(max(0, i-5), min(len(lines), i+10)):
                                if "HYBRID EXCHANGE" in lines[j] and "20" in lines[j]:
                                    self.data["functional"] = "B3LYP"
                                    functional_found = True
                                    break
                            if not functional_found:
                                self.data["functional"] = "BLYP"
                                functional_found = True
                
                if functional_found:
                    break

    # Check for functional in DFT PARAMETERS section (as backup)
    if not functional_found:
        for i, line in enumerate(lines):
            if "KOHN-SHAM HAMILTONIAN" in line:
                # Look at the next few lines for functional info
                for j in range(i + 1, min(i + 10, len(lines))):
                    if "(EXCHANGE)" in lines[j] and "[CORRELATION]" in lines[j]:
                        # Parse functional info
                        if "BECKE 88" in lines[j] and "LEE-YANG-PARR" in lines[j]:
                            self.data["functional"] = "B3LYP"
                            functional_found = True
                            break
                        elif "PBE" in lines[j]:
                            # Check if it's a hybrid
                            for k in range(j, min(j + 5, len(lines))):
                                if "HYBRID EXCHANGE" in lines[k]:
                                    self.data["functional"] = "PBE0"
                                    functional_found = True
                                    break
                            if not functional_found:
                                self.data["functional"] = "PBE"
                                functional_found = True
                            break
                
                if functional_found:
                    break

    # Check for dispersion correction after determining the functional
    if functional_found and self.data.get("functional"):
        # Look for GRIMME D3 or other dispersion indicators
        for line in lines:
            if "GRIMME D3" in line or "DFT-D3" in line or "DISPERSION" in line:
                # Don't add -D3 if it's already in the functional name
                if "-D3" not in self.data["functional"] and "-3C" not in self.data["functional"]:
                    self.data["functional"] += "-D3"
                break