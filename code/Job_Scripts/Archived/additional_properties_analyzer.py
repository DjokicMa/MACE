#!/usr/bin/env python3
"""
Additional Properties Analyzer for CRYSTAL Calculations
=======================================================
Analyze CRYSTAL OPT and SP output files to identify additional properties
that could be extracted beyond the current coverage.

Author: Generated for materials database project
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict


class AdditionalPropertiesAnalyzer:
    """Analyze CRYSTAL outputs to find additional extractable properties."""
    
    def __init__(self):
        self.patterns_to_find = {
            # Electronic structure properties
            'electronic': [
                r'HOMO\s*[-=]\s*([\d.-]+)',
                r'LUMO\s*[-=]\s*([\d.-]+)', 
                r'FERMI ENERGY\s*[=:]\s*([\d.-]+)',
                r'VACUUM LEVEL\s*[=:]\s*([\d.-]+)',
                r'WORK FUNCTION\s*[=:]\s*([\d.-]+)',
                r'IONIZATION POTENTIAL\s*[=:]\s*([\d.-]+)',
                r'ELECTRON AFFINITY\s*[=:]\s*([\d.-]+)',
                r'BAND WIDTH\s*[=:]\s*([\d.-]+)',
                r'DOS AT FERMI LEVEL\s*[=:]\s*([\d.-]+)',
                r'EFFECTIVE MASS\s*[=:]\s*([\d.-]+)',
            ],
            
            # Thermodynamic properties
            'thermodynamic': [
                r'ZERO POINT ENERGY\s*[=:]\s*([\d.-]+)',
                r'THERMAL ENERGY\s*[=:]\s*([\d.-]+)',
                r'ENTROPY\s*[=:]\s*([\d.-]+)',
                r'FREE ENERGY\s*[=:]\s*([\d.-]+)',
                r'HEAT CAPACITY\s*[=:]\s*([\d.-]+)',
                r'FORMATION ENERGY\s*[=:]\s*([\d.-]+)',
                r'COHESIVE ENERGY\s*[=:]\s*([\d.-]+)',
            ],
            
            # Vibrational properties
            'vibrational': [
                r'FREQUENCY\s*[=:]\s*([\d.-]+)',
                r'PHONON\s*[=:]\s*([\d.-]+)',
                r'VIBRATION\s*[=:]\s*([\d.-]+)',
                r'NORMAL MODE\s*[=:]\s*([\d.-]+)',
                r'INFRARED INTENSITY\s*[=:]\s*([\d.-]+)',
                r'RAMAN INTENSITY\s*[=:]\s*([\d.-]+)',
            ],
            
            # Mechanical properties  
            'mechanical': [
                r'BULK MODULUS\s*[=:]\s*([\d.-]+)',
                r'SHEAR MODULUS\s*[=:]\s*([\d.-]+)',
                r'YOUNG.?S MODULUS\s*[=:]\s*([\d.-]+)',
                r'POISSON RATIO\s*[=:]\s*([\d.-]+)',
                r'ELASTIC CONSTANT\s*[=:]\s*([\d.-]+)',
                r'COMPRESSIBILITY\s*[=:]\s*([\d.-]+)',
                r'HARDNESS\s*[=:]\s*([\d.-]+)',
            ],
            
            # Optical properties
            'optical': [
                r'REFRACTIVE INDEX\s*[=:]\s*([\d.-]+)',
                r'ABSORPTION\s*[=:]\s*([\d.-]+)',
                r'REFLECTANCE\s*[=:]\s*([\d.-]+)',
                r'TRANSMITTANCE\s*[=:]\s*([\d.-]+)',
                r'DIELECTRIC CONSTANT\s*[=:]\s*([\d.-]+)',
                r'OPTICAL GAP\s*[=:]\s*([\d.-]+)',
                r'PLASMA FREQUENCY\s*[=:]\s*([\d.-]+)',
            ],
            
            # Charge and electric properties
            'electric': [
                r'DIPOLE MOMENT\s*[=:]\s*([\d.-]+)',
                r'QUADRUPOLE MOMENT\s*[=:]\s*([\d.-]+)',
                r'POLARIZABILITY\s*[=:]\s*([\d.-]+)',
                r'ELECTRIC FIELD\s*[=:]\s*([\d.-]+)',
                r'ELECTROSTATIC POTENTIAL\s*[=:]\s*([\d.-]+)',
                r'CHARGE DENSITY\s*[=:]\s*([\d.-]+)',
            ],
            
            # Convergence and computational properties
            'computational': [
                r'SCF CYCLES\s*[=:]\s*(\d+)',
                r'CONVERGENCE REACHED\s*[=:]\s*([\d.-]+)',
                r'CPU TIME\s*[=:]\s*([\d.-]+)',
                r'WALL TIME\s*[=:]\s*([\d.-]+)',
                r'MEMORY USAGE\s*[=:]\s*([\d.-]+)',
                r'DISK USAGE\s*[=:]\s*([\d.-]+)',
                r'ITERATIONS\s*[=:]\s*(\d+)',
            ],
            
            # Symmetry and structural
            'symmetry': [
                r'POINT GROUP\s*[=:]\s*([A-Za-z0-9]+)',
                r'SPACE GROUP SYMBOL\s*[=:]\s*([A-Za-z0-9]+)',
                r'BRAVAIS LATTICE\s*[=:]\s*([A-Za-z0-9]+)',
                r'CRYSTAL CLASS\s*[=:]\s*([A-Za-z0-9]+)',
                r'SYMMETRY OPERATIONS\s*[=:]\s*(\d+)',
                r'EQUIVALENT ATOMS\s*[=:]\s*(\d+)',
            ]
        }
        
        # Properties we already extract (to avoid duplicates)
        self.existing_properties = {
            'total_energy_au', 'total_energy_ev', 'band_gap', 'direct_band_gap', 'indirect_band_gap',
            'primitive_a', 'primitive_b', 'primitive_c', 'primitive_alpha', 'primitive_beta', 'primitive_gamma',
            'primitive_cell_volume', 'density', 'optimization_cycles', 'optimization_converged',
            'mulliken_population', 'overlap_population', 'space_group_number', 'atoms_in_unit_cell'
        }
    
    def analyze_output_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single output file for additional properties."""
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Could not read file: {e}"}
        
        found_properties = defaultdict(list)
        potential_sections = []
        
        # Search for each pattern category
        for category, patterns in self.patterns_to_find.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    property_info = {
                        'pattern': pattern,
                        'value': match.group(1) if match.groups() else match.group(0),
                        'context': self._get_context(content, match.start(), match.end())
                    }
                    found_properties[category].append(property_info)
        
        # Look for interesting sections that might contain properties
        potential_sections = self._find_potential_sections(content)
        
        return {
            'file_path': str(file_path),
            'found_properties': dict(found_properties),
            'potential_sections': potential_sections,
            'analysis_summary': self._summarize_findings(found_properties)
        }
    
    def _get_context(self, content: str, start: int, end: int, context_size: int = 100) -> str:
        """Get context around a match."""
        context_start = max(0, start - context_size)
        context_end = min(len(content), end + context_size)
        context = content[context_start:context_end]
        
        # Clean up and truncate
        lines = context.split('\n')
        if len(lines) > 5:
            lines = lines[:2] + ['...'] + lines[-2:]
        
        return '\n'.join(lines)
    
    def _find_potential_sections(self, content: str) -> List[Dict[str, str]]:
        """Find sections that might contain interesting properties."""
        section_patterns = [
            r'(PROPERTIES|ANALYSIS|SUMMARY|RESULTS)\s*\n([^\n]*\n){1,20}',
            r'(EIGENVALUES|EIGENVECTORS)\s*\n([^\n]*\n){1,10}',
            r'(TOTAL|FINAL|CONVERGED)\s+[A-Z\s]+\s*\n([^\n]*\n){1,5}',
            r'(\*{10,})\s*\n([^\n]*\n){1,10}',
            r'(={10,})\s*\n([^\n]*\n){1,10}'
        ]
        
        potential_sections = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                section = {
                    'type': 'potential_section',
                    'content': match.group(0)[:500],  # First 500 chars
                    'start_pos': match.start()
                }
                potential_sections.append(section)
        
        return potential_sections[:10]  # Limit to first 10
    
    def _summarize_findings(self, found_properties: Dict) -> Dict[str, Any]:
        """Summarize the analysis findings."""
        summary = {
            'total_categories': len(found_properties),
            'total_matches': sum(len(props) for props in found_properties.values()),
            'categories_found': list(found_properties.keys()),
            'most_common_category': None,
            'unique_patterns': set()
        }
        
        if found_properties:
            # Find most common category
            category_counts = {cat: len(props) for cat, props in found_properties.items()}
            summary['most_common_category'] = max(category_counts, key=category_counts.get)
            
            # Collect unique patterns
            for props in found_properties.values():
                for prop in props:
                    summary['unique_patterns'].add(prop['pattern'])
        
        summary['unique_patterns'] = len(summary['unique_patterns'])
        
        return summary
    
    def analyze_multiple_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Analyze multiple output files to find common additional properties."""
        all_results = {}
        combined_findings = defaultdict(list)
        
        print(f"🔍 Analyzing {len(file_paths)} output files for additional properties...")
        
        for file_path in file_paths:
            result = self.analyze_output_file(file_path)
            all_results[str(file_path)] = result
            
            # Combine findings
            if 'found_properties' in result:
                for category, properties in result['found_properties'].items():
                    combined_findings[category].extend(properties)
        
        # Analyze patterns across files
        pattern_frequency = defaultdict(int)
        value_examples = defaultdict(list)
        
        for category, properties in combined_findings.items():
            for prop in properties:
                pattern = prop['pattern']
                pattern_frequency[pattern] += 1
                value_examples[pattern].append(prop['value'])
        
        # Identify most promising additional properties
        promising_properties = []
        for pattern, frequency in pattern_frequency.items():
            if frequency >= 2:  # Found in at least 2 files
                promising_properties.append({
                    'pattern': pattern,
                    'frequency': frequency,
                    'example_values': value_examples[pattern][:3],  # First 3 examples
                    'potential_property_name': self._suggest_property_name(pattern)
                })
        
        # Sort by frequency
        promising_properties.sort(key=lambda x: x['frequency'], reverse=True)
        
        return {
            'files_analyzed': len(file_paths),
            'total_matches': sum(len(props) for props in combined_findings.values()),
            'categories_found': list(combined_findings.keys()),
            'pattern_frequency': dict(pattern_frequency),
            'promising_properties': promising_properties,
            'detailed_results': all_results
        }
    
    def _suggest_property_name(self, pattern: str) -> str:
        """Suggest a property name based on the pattern."""
        # Extract key words from pattern
        pattern_clean = re.sub(r'[^A-Za-z\s]', ' ', pattern).lower()
        words = [w for w in pattern_clean.split() if w and len(w) > 2]
        
        # Remove common regex words
        exclude = {'s', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        words = [w for w in words if w not in exclude]
        
        if words:
            return '_'.join(words[:3])  # Use first 3 meaningful words
        else:
            return 'unknown_property'


def analyze_workflow_outputs():
    """Analyze all workflow output files for additional properties."""
    analyzer = AdditionalPropertiesAnalyzer()
    
    # Find all output files
    workflow_dir = Path("workflow_outputs")
    output_files = list(workflow_dir.rglob("*.out"))
    
    if not output_files:
        print("❌ No output files found in workflow_outputs/")
        return
    
    # Separate by calculation type
    opt_files = [f for f in output_files if 'opt' in f.name.lower()]
    sp_files = [f for f in output_files if 'sp' in f.name.lower()]
    band_files = [f for f in output_files if 'band' in f.name.lower()]
    doss_files = [f for f in output_files if 'doss' in f.name.lower()]
    
    print(f"📁 Found {len(output_files)} total output files:")
    print(f"   OPT files: {len(opt_files)}")
    print(f"   SP files: {len(sp_files)}")
    print(f"   BAND files: {len(band_files)}")
    print(f"   DOSS files: {len(doss_files)}")
    
    # Analyze each type
    results = {}
    
    if opt_files:
        print(f"\n🔍 Analyzing OPT calculations...")
        results['OPT'] = analyzer.analyze_multiple_files(opt_files)
    
    if sp_files:
        print(f"\n🔍 Analyzing SP calculations...")
        results['SP'] = analyzer.analyze_multiple_files(sp_files)
    
    return results


if __name__ == "__main__":
    print("🔍 Additional Properties Analysis for CRYSTAL Calculations")
    print("=" * 60)
    
    # Analyze workflow outputs
    results = analyze_workflow_outputs()
    
    if not results:
        print("❌ No analysis results obtained")
        exit(1)
    
    # Report findings
    for calc_type, analysis in results.items():
        print(f"\n📊 {calc_type} CALCULATION ANALYSIS:")
        print(f"   Files analyzed: {analysis['files_analyzed']}")
        print(f"   Total matches: {analysis['total_matches']}")
        print(f"   Categories found: {', '.join(analysis['categories_found'])}")
        
        promising = analysis['promising_properties']
        if promising:
            print(f"\n🎯 Most Promising Additional Properties for {calc_type}:")
            for i, prop in enumerate(promising[:10], 1):  # Top 10
                print(f"   {i}. {prop['potential_property_name']}")
                print(f"      Pattern: {prop['pattern']}")
                print(f"      Found in: {prop['frequency']} files")
                print(f"      Examples: {', '.join(str(v) for v in prop['example_values'])}")
                print()
        else:
            print(f"   No additional properties found with high confidence")
    
    # Overall summary
    all_promising = []
    for analysis in results.values():
        all_promising.extend(analysis['promising_properties'])
    
    if all_promising:
        print(f"\n🎯 OVERALL RECOMMENDATIONS:")
        print(f"   Found {len(all_promising)} potential additional properties")
        
        # Group by potential impact
        high_impact = [p for p in all_promising if p['frequency'] >= 5]
        medium_impact = [p for p in all_promising if 2 <= p['frequency'] < 5]
        
        if high_impact:
            print(f"\n⭐ HIGH IMPACT (found in 5+ files):")
            for prop in high_impact[:5]:
                print(f"   • {prop['potential_property_name']} ({prop['frequency']} files)")
        
        if medium_impact:
            print(f"\n🔸 MEDIUM IMPACT (found in 2-4 files):")
            for prop in medium_impact[:5]:
                print(f"   • {prop['potential_property_name']} ({prop['frequency']} files)")
    
    print(f"\n✅ Analysis complete! Check results for implementation opportunities.")