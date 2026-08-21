#!/bin/bash --login
# Empirically map element coverage of CRYSTAL23 internal basis sets.
# Usage: scan_basis.sh "<basis list>" "<z list>" <outfile>
module purge >/dev/null 2>&1
module load CRYSTAL/23-intel-2023a >/dev/null 2>&1
BASES="$1"; ZLIST="$2"; OUT="$3"
SCR=/mnt/gs21/scratch/djokicma/basisscan
mkdir -p "$SCR"
echo "basis,Z,status,detail" > "$OUT"
for B in $BASES; do
  for Z in $ZLIST; do
    d="$SCR/${B//[^A-Za-z0-9]/_}"; rm -rf "$d"; mkdir -p "$d"; cd "$d" || continue
    cat > INPUT <<EOF
scan
CRYSTAL
0 0 0
1
8.0 8.0 8.0 90.0 90.0 90.0
1
$Z 0.0 0.0 0.0
BASISSET
$B
MAXCYCLE
1
END
EOF
    timeout 30 "$EBROOTCRYSTAL/bin/crystal" < INPUT > out.txt 2>&1
    err=$(grep -m1 "ERROR" out.txt 2>/dev/null | tr -s ' ' | sed 's/^ *//;s/,/;/g')
    [ -z "$err" ] && err=$(head -1 fort.87 2>/dev/null | tr -s ' ' | sed 's/^ *//;s/,/;/g')
    if grep -q "LOCAL ATOMIC FUNCTIONS BASIS SET" out.txt 2>/dev/null; then
      st=PRESENT; err=""
    elif echo "$err" | grep -qi "LoadBa\|NOT NEUTRAL\|basis"; then
      st=MISSING
    elif [ -n "$err" ]; then
      st=OTHER
    else
      st=UNKNOWN
    fi
    echo "$B,$Z,$st,$err" >> "$OUT"
  done
done
