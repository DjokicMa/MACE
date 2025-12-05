#!/usr/bin/env python3
"""
Node Exclusion Utility for MACE SLURM Scripts

This module provides functionality to query available SLURM nodes and generate
compact exclusion strings for SLURM submission scripts.

Author: Marcus Djokic
"""

import subprocess
import re
from typing import List, Tuple, Optional, Set
import sys


class NodeExclusionManager:
    """Manages SLURM node exclusion for job submissions."""

    # Common node types in the HPC cluster
    KNOWN_NODE_TYPES = ['amr', 'nvf', 'nvl', 'agg', 'agx', 'skl', 'nfh', 'neh', 'nel', 'acm', 'nal', 'nif', 'dev']

    # Node type descriptions with cluster generation
    NODE_TYPE_INFO = {
        'amr': 'AMD EPYC 7452 (amd20)',
        'nvf': 'AMD EPYC 7452 + V100 (amd20-v100)',
        'agg': 'Intel Aggregator (amd24)',
        'agx': 'Intel AGX (amd24)',
        'nfh': 'NVIDIA H-series (amd24)',
        'neh': 'Enhanced H-series (amd24)',
        'nel': 'Enhanced L-series (amd24)',
        'acm': 'ACM compute (amd22)',
        'nal': 'NAL nodes (amd21)',
        'nif': 'NIF nodes (intel21)',
        'skl': 'Intel Skylake (intel18)',
        'nvl': 'NVIDIA L-series (intel18-v100)',
        'dev': 'Development/test nodes'
    }

    # Cluster generation groupings
    CLUSTER_TYPES = {
        'amd24': ['agg', 'agx', 'nfh', 'neh', 'nel'],
        'amd22': ['acm'],
        'amd21': ['nal'],
        'amd20': ['amr'],
        'amd20-v100': ['nvf'],
        'intel21': ['nif'],
        'intel18': ['skl'],
        'intel18-v100': ['nvl']
    }

    # Predefined exclusion sets
    MENDOZA_NODES = ['agg-011', 'agg-012', 'amr-163', 'amr-178', 'amr-179']

    # AMD20 node types (amr + nvf) - kept for backward compatibility
    AMD20_NODE_TYPES = ['amr', 'nvf']

    def __init__(self):
        """Initialize the node exclusion manager."""
        self.available_nodes = {}

    def get_amd20_nodes(self) -> List[str]:
        """
        Query and return all AMD20 nodes (amr + nvf types).

        Returns:
            Combined list of all amr and nvf nodes
        """
        all_amd20_nodes = []
        for node_type in self.AMD20_NODE_TYPES:
            nodes = self.query_nodes_by_type(node_type)
            all_amd20_nodes.extend(nodes)
        return sorted(all_amd20_nodes, key=self._extract_node_number)

    def get_nodes_by_cluster(self, cluster_type: str) -> List[str]:
        """
        Query and return all nodes for a specific cluster generation.

        Args:
            cluster_type: Cluster generation (e.g., 'amd24', 'amd22', 'intel18')

        Returns:
            Combined list of all nodes in that cluster generation
        """
        if cluster_type not in self.CLUSTER_TYPES:
            print(f"Warning: Unknown cluster type '{cluster_type}'")
            return []

        all_nodes = []
        node_types = self.CLUSTER_TYPES[cluster_type]

        for node_type in node_types:
            nodes = self.query_nodes_by_type(node_type)
            all_nodes.extend(nodes)

        return sorted(all_nodes, key=self._extract_node_number)

    def query_nodes_by_type(self, node_type: str) -> List[str]:
        """
        Query SLURM for all available nodes of a specific type.

        Args:
            node_type: Node prefix (e.g., 'amr', 'nvf', 'agg')

        Returns:
            List of node names matching the type
        """
        try:
            # Query SLURM for nodes
            cmd = f"scontrol show nodes | grep 'NodeName={node_type}'"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"Warning: Could not query nodes of type '{node_type}'")
                return []

            # Parse node names
            nodes = []
            pattern = rf'NodeName=({node_type}-\d+)'
            for line in result.stdout.split('\n'):
                match = re.search(pattern, line)
                if match:
                    nodes.append(match.group(1))

            return sorted(nodes, key=self._extract_node_number)

        except subprocess.TimeoutExpired:
            print(f"Error: Timeout querying nodes of type '{node_type}'")
            return []
        except Exception as e:
            print(f"Error querying nodes: {e}")
            return []

    def _extract_node_number(self, node_name: str) -> int:
        """
        Extract the numeric part from a node name.

        Args:
            node_name: Node name like 'amr-042' or 'nvf-123'

        Returns:
            The numeric part as an integer
        """
        match = re.search(r'-(\d+)$', node_name)
        if match:
            return int(match.group(1))
        return 0

    def _parse_node_list(self, nodes: List[str]) -> Tuple[str, List[int]]:
        """
        Parse a list of node names to extract prefix and numbers.

        Args:
            nodes: List of node names like ['amr-000', 'amr-001', ...]

        Returns:
            Tuple of (prefix, sorted list of numbers)
        """
        if not nodes:
            return ('', [])

        # Extract prefix from first node
        match = re.match(r'^([a-z]+)-', nodes[0])
        prefix = match.group(1) if match else ''

        # Extract all numbers
        numbers = []
        for node in nodes:
            num = self._extract_node_number(node)
            numbers.append(num)

        return (prefix, sorted(set(numbers)))

    def _group_into_ranges(self, numbers: List[int]) -> List[Tuple[int, int]]:
        """
        Group consecutive numbers into ranges.

        Args:
            numbers: Sorted list of integers

        Returns:
            List of (start, end) tuples representing ranges
        """
        if not numbers:
            return []

        ranges = []
        start = numbers[0]
        end = numbers[0]

        for i in range(1, len(numbers)):
            if numbers[i] == end + 1:
                # Consecutive number, extend range
                end = numbers[i]
            else:
                # Gap found, save current range and start new one
                ranges.append((start, end))
                start = numbers[i]
                end = numbers[i]

        # Add final range
        ranges.append((start, end))

        return ranges

    def _format_range(self, start: int, end: int, width: int = 3) -> str:
        """
        Format a range of numbers with proper zero-padding.

        Args:
            start: Start of range
            end: End of range
            width: Number of digits for zero-padding (default: 3)

        Returns:
            Formatted range string (e.g., '000-042' or '042' if single)
        """
        if start == end:
            return f"{start:0{width}d}"
        else:
            return f"{start:0{width}d}-{end:0{width}d}"

    def create_exclude_string(self, nodes: List[str]) -> str:
        """
        Create a compact SLURM exclude string from a list of nodes.

        Args:
            nodes: List of node names to exclude

        Returns:
            Formatted exclude string (e.g., 'amr-[000-042,050-100]')
            For mixed prefixes: 'agg-[011-012],amr-[163,178-179]'
        """
        if not nodes:
            return ""

        # Group nodes by prefix
        prefix_groups = {}
        for node in nodes:
            match = re.match(r'^([a-z]+)-(\d+)$', node)
            if match:
                prefix = match.group(1)
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(node)

        # If only one prefix, use simple format
        if len(prefix_groups) == 1:
            prefix, nodes_list = list(prefix_groups.items())[0]
            _, numbers = self._parse_node_list(nodes_list)
            if not numbers:
                return ""

            ranges = self._group_into_ranges(numbers)

            # Format ranges into compact notation
            range_strs = [self._format_range(start, end) for start, end in ranges]

            if len(range_strs) == 1 and '-' not in range_strs[0]:
                # Single node, no brackets needed
                return f"{prefix}-{range_strs[0]}"
            else:
                # Multiple ranges or single range, use bracket notation
                return f"{prefix}-[{','.join(range_strs)}]"
        else:
            # Multiple prefixes, use multi-type format
            return self.create_multi_type_exclude_string(prefix_groups)

    def create_multi_type_exclude_string(self, exclude_dict: dict) -> str:
        """
        Create exclude string for multiple node types.

        Args:
            exclude_dict: Dictionary mapping node types to lists of nodes
                         e.g., {'amr': ['amr-000', 'amr-001'], 'agg': ['agg-011']}

        Returns:
            Combined exclude string (e.g., 'amr-[000-001],agg-011')
        """
        exclude_parts = []

        for node_type, nodes in exclude_dict.items():
            if nodes:
                exclude_str = self.create_exclude_string(nodes)
                if exclude_str:
                    exclude_parts.append(exclude_str)

        return ','.join(exclude_parts)

    def interactive_node_exclusion(self) -> Optional[str]:
        """
        Interactive prompt for node exclusion configuration.

        Returns:
            SLURM exclude string or None if no exclusion requested
        """
        print("\n" + "="*70)
        print("SLURM Node Exclusion Configuration")
        print("="*70)

        print("\nSelect node exclusion option:")
        print("1) No exclusions (use all available nodes)")
        print("2) Exclude all AMD20 nodes (amr + nvf) [RECOMMENDED for CRYSTAL23]")
        print("3) Exclude by cluster generation (amd24, amd22, amd21, intel18, etc.)")
        print("4) Exclude Mendoza group nodes (save CPU hours)")
        print("5) Exclude specific node type(s) - allows multiple")
        print("6) Custom node exclusion list")

        choice = input("\nEnter choice [1-6] (default: 2): ").strip() or "2"

        if choice == "1":
            print("\nNo node exclusions will be applied.")
            return None

        elif choice == "2":
            return self._exclude_amd20_nodes()

        elif choice == "3":
            return self._exclude_by_cluster()

        elif choice == "4":
            exclude_str = self.create_exclude_string(self.MENDOZA_NODES)
            print(f"\nExcluding Mendoza nodes: {exclude_str}")
            return exclude_str

        elif choice == "5":
            return self._exclude_by_type()

        elif choice == "6":
            return self._custom_exclusion()

        else:
            print("Invalid choice. No exclusions will be applied.")
            return None

    def _exclude_amd20_nodes(self) -> Optional[str]:
        """Handle exclusion of all AMD20 nodes (amr + nvf)"""
        print("\nQuerying SLURM for all AMD20 nodes (amr + nvf)...")
        amd20_nodes = self.get_amd20_nodes()

        if not amd20_nodes:
            print("No AMD20 nodes found")
            return None

        # Count by type
        amr_count = sum(1 for n in amd20_nodes if n.startswith('amr-'))
        nvf_count = sum(1 for n in amd20_nodes if n.startswith('nvf-'))

        print(f"Found {len(amd20_nodes)} AMD20 nodes:")
        print(f"  - amr (AMD EPYC 7452): {amr_count} nodes")
        print(f"  - nvf (AMD EPYC 7452 + V100): {nvf_count} nodes")

        confirm = input(f"\nExclude all {len(amd20_nodes)} AMD20 nodes? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            exclude_str = self.create_exclude_string(amd20_nodes)
            print(f"\nExclude string: {exclude_str}")
            return exclude_str
        else:
            print("Exclusion cancelled.")
            return None

    def _exclude_by_cluster(self) -> Optional[str]:
        """Handle exclusion by cluster generation."""
        print("\nAvailable cluster generations:")
        print("  1) amd24      - agg, agx, nfh, neh, nel (newest)")
        print("  2) amd22      - acm")
        print("  3) amd21      - nal")
        print("  4) amd20      - amr (memory issues with CRYSTAL23)")
        print("  5) amd20-v100 - nvf (memory issues with CRYSTAL23)")
        print("  6) intel21    - nif")
        print("  7) intel18    - skl")
        print("  8) intel18-v100 - nvl")
        print("  9) Multiple clusters (comma/space separated)")

        choice = input("\nSelect cluster generation [1-9]: ").strip()

        cluster_map = {
            '1': 'amd24',
            '2': 'amd22',
            '3': 'amd21',
            '4': 'amd20',
            '5': 'amd20-v100',
            '6': 'intel21',
            '7': 'intel18',
            '8': 'intel18-v100'
        }

        clusters = []
        if choice in cluster_map:
            clusters = [cluster_map[choice]]
        elif choice == '9':
            cluster_input = input("\nEnter cluster types (comma or space separated): ").strip()
            if ',' in cluster_input:
                clusters = [c.strip() for c in cluster_input.split(',') if c.strip()]
            else:
                clusters = [c.strip() for c in cluster_input.split() if c.strip()]
        else:
            print("Invalid choice.")
            return None

        if not clusters:
            print("No clusters specified.")
            return None

        # Query nodes for each cluster
        all_nodes = []
        cluster_counts = {}

        print(f"\nQuerying SLURM for cluster(s): {', '.join(clusters)}...")

        for cluster in clusters:
            if cluster not in self.CLUSTER_TYPES:
                print(f"  Warning: Unknown cluster type '{cluster}', skipping")
                continue

            nodes = self.get_nodes_by_cluster(cluster)
            if nodes:
                all_nodes.extend(nodes)
                node_types = self.CLUSTER_TYPES[cluster]
                cluster_counts[cluster] = {
                    'total': len(nodes),
                    'types': ', '.join(node_types)
                }
                print(f"  Found {len(nodes):3d} {cluster} nodes ({', '.join(node_types)})")
            else:
                print(f"  Found   0 {cluster} nodes")

        if not all_nodes:
            print(f"\nNo nodes found for specified cluster(s)")
            return None

        # Summary
        print(f"\nTotal: {len(all_nodes)} nodes across {len(cluster_counts)} cluster(s)")
        for cluster, info in cluster_counts.items():
            print(f"  - {cluster}: {info['total']} nodes ({info['types']})")

        confirm = input(f"\nExclude all {len(all_nodes)} nodes? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            exclude_str = self.create_exclude_string(all_nodes)
            print(f"\nExclude string: {exclude_str}")
            return exclude_str
        else:
            print("Exclusion cancelled.")
            return None

    def _exclude_by_type(self) -> Optional[str]:
        """Handle exclusion of all nodes by type(s) - supports multiple types."""
        print("\nAvailable node types:")
        for i, node_type in enumerate(self.KNOWN_NODE_TYPES, 1):
            info = self.NODE_TYPE_INFO.get(node_type, '')
            print(f"{i}) {node_type:6s} - {info}")
        print(f"{len(self.KNOWN_NODE_TYPES) + 1}) Enter custom type(s)")

        choice = input(f"\nSelect node type [1-{len(self.KNOWN_NODE_TYPES) + 1}]: ").strip()

        # Parse the choice
        node_types = []
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(self.KNOWN_NODE_TYPES):
                node_types = [self.KNOWN_NODE_TYPES[choice_num - 1]]
            elif choice_num == len(self.KNOWN_NODE_TYPES) + 1:
                # Allow custom types - comma or space separated
                custom_input = input("\nEnter node type(s) (comma or space separated): ").strip()
                # Parse both comma and space separated
                if ',' in custom_input:
                    node_types = [t.strip() for t in custom_input.split(',') if t.strip()]
                else:
                    node_types = [t.strip() for t in custom_input.split() if t.strip()]
            else:
                print("Invalid choice.")
                return None
        except ValueError:
            print("Invalid input.")
            return None

        if not node_types:
            print("No node types specified.")
            return None

        # Query SLURM for each node type
        all_nodes = []
        node_counts = {}

        print(f"\nQuerying SLURM for node types: {', '.join(node_types)}...")

        for node_type in node_types:
            nodes = self.query_nodes_by_type(node_type)
            if nodes:
                all_nodes.extend(nodes)
                node_counts[node_type] = len(nodes)
                print(f"  Found {len(nodes):3d} {node_type} nodes")
            else:
                print(f"  Found   0 {node_type} nodes")

        if not all_nodes:
            print(f"\nNo nodes found for specified type(s)")
            return None

        # Summary
        print(f"\nTotal: {len(all_nodes)} nodes across {len(node_counts)} type(s)")
        for node_type, count in node_counts.items():
            print(f"  - {node_type}: {count} nodes")

        confirm = input(f"\nExclude all {len(all_nodes)} nodes? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            exclude_str = self.create_exclude_string(all_nodes)
            print(f"\nExclude string: {exclude_str}")
            return exclude_str
        else:
            print("Exclusion cancelled.")
            return None

    def _custom_exclusion(self) -> Optional[str]:
        """Handle custom node exclusion list."""
        print("\nEnter nodes to exclude (comma-separated).")
        print("Examples:")
        print("  - Single nodes: amr-042,amr-050,nvf-123")
        print("  - With ranges: amr-[042-050],nvf-[100-110]")

        custom_input = input("\nNodes to exclude: ").strip()

        if not custom_input:
            print("No exclusions specified.")
            return None

        # Check if already in SLURM format
        if '[' in custom_input and ']' in custom_input:
            print(f"\nUsing provided exclude string: {custom_input}")
            return custom_input

        # Parse comma-separated node names
        nodes = [n.strip() for n in custom_input.split(',')]

        # Group by prefix
        node_groups = {}
        for node in nodes:
            match = re.match(r'^([a-z]+)-(\d+)$', node)
            if match:
                prefix = match.group(1)
                if prefix not in node_groups:
                    node_groups[prefix] = []
                node_groups[prefix].append(node)
            else:
                print(f"Warning: Invalid node format '{node}', skipping")

        # Create exclude strings for each type
        exclude_dict = {}
        for prefix, prefix_nodes in node_groups.items():
            exclude_dict[prefix] = prefix_nodes

        exclude_str = self.create_multi_type_exclude_string(exclude_dict)
        print(f"\nCompact exclude string: {exclude_str}")

        return exclude_str


def main():
    """Main function for testing node exclusion functionality."""
    manager = NodeExclusionManager()

    if len(sys.argv) > 1:
        # Command-line mode
        if sys.argv[1] == '--query':
            if len(sys.argv) < 3:
                print("Usage: node_exclusion.py --query <node_type>")
                sys.exit(1)
            node_type = sys.argv[2]
            nodes = manager.query_nodes_by_type(node_type)
            print(f"Found {len(nodes)} nodes of type '{node_type}':")
            for node in nodes:
                print(f"  {node}")

        elif sys.argv[1] == '--exclude-type':
            if len(sys.argv) < 3:
                print("Usage: node_exclusion.py --exclude-type <node_type>")
                sys.exit(1)
            node_type = sys.argv[2]
            nodes = manager.query_nodes_by_type(node_type)
            exclude_str = manager.create_exclude_string(nodes)
            print(exclude_str)

        elif sys.argv[1] == '--test':
            # Test with example nodes
            test_nodes = [
                'amr-000', 'amr-001', 'amr-002', 'amr-005', 'amr-006',
                'amr-010', 'amr-011', 'amr-012', 'amr-015',
                'amr-100', 'amr-101', 'amr-102'
            ]
            exclude_str = manager.create_exclude_string(test_nodes)
            print(f"Test nodes: {test_nodes}")
            print(f"Exclude string: {exclude_str}")

        else:
            print("Unknown command")
            print("Usage: node_exclusion.py [--query <type> | --exclude-type <type> | --test]")
    else:
        # Interactive mode
        exclude_str = manager.interactive_node_exclusion()
        if exclude_str:
            print(f"\n{'='*70}")
            print("SLURM exclude line to add to your submission script:")
            print(f"#SBATCH --exclude={exclude_str}")
            print('='*70)


if __name__ == '__main__':
    main()
