# Markdown Files Reorganization Plan

## Overview
Reorganizing markdown documentation to keep main directory focused on essential docs while preserving development history.

## Actions to Take:

### 1. ARCHIVE These Files (Move to Archived/):
- [x] archive_summary_20250627.md - Today's archiving activities
- [ ] duplicate_functions_analysis.md - Code duplication analysis
- [ ] script_usage_verification.md - Script usage analysis
- [ ] workflow_consolidation_guide.md - Future consolidation guide
- [ ] workflow_refactoring_plan.md - Comprehensive refactoring plan
- [ ] workflow_robustness_report.md - Robustness analysis report
- [ ] workflow_dependencies_analysis.md - After extracting key info
- [ ] yaml_files_analysis.md - After extracting key info

### 2. MERGE Key Information into README.md:

#### From workflow_dependencies_analysis.md:
Add new section "External Dependencies and Integration":
- List of external scripts (NewCifToD12.py, CRYSTALOptToD12.py, etc.)
- Key integration points
- Script location resolution details

#### From yaml_files_analysis.md:
Add note in configuration section:
- recovery_config.yaml is actively used by error_recovery.py
- workflows.yaml exists but is not currently loaded (workflow definitions are hardcoded)

### 3. KEEP in Main Directory:
- README.md - Main comprehensive documentation

## Benefits:
1. Cleaner main directory focused on active documentation
2. Preserved development history in Archived/
3. More comprehensive README.md with all essential information
4. Clear separation between operational docs and development artifacts