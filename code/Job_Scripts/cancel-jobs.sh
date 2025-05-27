#!/bin/sh

if [ -z "$1" ] ; then
    echo "Minimum Job Number argument is required.  Run as '$0 jobnum'"
    exit 1
fi

minjobnum="$1"

myself="$(id -u -n)"

for j in $(squeue --user="$myself" --noheader --format='%i') ; do
  if [ "$j" -gt "$minjobnum" ] ; then
    scancel "$j"
  fi
done
