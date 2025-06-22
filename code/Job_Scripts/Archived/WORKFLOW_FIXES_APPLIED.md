# Workflow Fixes Applied

## Summary of Issues Fixed

### 1. Workflow Sequence Execution Fix
**Problem**: Workflow engine was ignoring the planned sequence and using hardcoded dependency rules
- After OPT, it would trigger SP instead of OPT2
- Material names were getting corrupted (test2_sp2, test2_sp3 instead of test3_sp, test4_sp)

**Solution**: Updated `_get_next_steps_from_sequence` method to:
- Follow the exact sequence order as planned
- Only apply parallel execution for specific known cases:
  - OPT2 → SP + OPT3 (parallel)
  - OPT3 → SP2 + FREQ (parallel)
  - SP2 → BAND + DOSS (parallel)

### 2. Material Name Extraction Fix
**Problem**: Materials with numbers (test3, test4, etc.) were incorrectly parsed
- Complex parsing logic was failing for simple numbered materials

**Solution**: Updated `extract_core_material_name` method to:
- First try simple suffix removal (_opt, _sp, etc.)
- Only use complex parsing for names with technical keywords
- Properly handle materials like test1, test2, test3-RCSR-ums, test4-CA

## Updated Workflow Logic

For sequence: `["OPT", "OPT2", "SP", "OPT3", "SP2", "BAND", "DOSS", "FREQ"]`

The execution flow is now:
```
1. OPT completes → Generate OPT2
2. OPT2 completes → Generate SP and OPT3 (parallel)
3. SP completes → Nothing (wait for other calculations)
4. OPT3 completes → Generate SP2 and FREQ (parallel)
5. SP2 completes → Generate BAND and DOSS (parallel)
6. BAND/DOSS complete → Nothing
7. FREQ completes → Workflow complete
```

## Files Modified

1. **workflow_engine.py**:
   - `_get_next_steps_from_sequence()`: Simplified to follow sequence order
   - `extract_core_material_name()`: Fixed for numbered materials

## Testing the Fix

To verify the fixes work correctly:

1. Check workflow progression:
   ```bash
   python check_workflows.py
   ```

2. Monitor workflow status:
   ```bash
   python workflow_status.py
   ```

3. Check material names in database:
   ```bash
   sqlite3 materials.db "SELECT material_id, calc_type FROM calculations ORDER BY created_at DESC LIMIT 20"
   ```

## Expected Behavior

After these fixes:
- OPT completion will trigger OPT2 (not SP)
- Material names will be preserved correctly (test3_sp, not test2_sp2)
- Workflow will follow the planned sequence exactly
- Parallel execution will occur only where specified

## Next Steps

1. The currently running calculations will complete normally
2. When they complete, the workflow engine will now follow the correct sequence
3. Monitor with `python workflow_status.py` to ensure proper progression

## Manual Intervention (if needed)

If calculations have already been generated with wrong names, you can:
1. Let them complete (they'll still work)
2. Use `check_workflows.py` to trigger the correct next steps
3. Or cancel and restart with the fixed workflow engine