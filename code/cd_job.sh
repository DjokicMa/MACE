#!/bin/bash

if [ "$1" != "" ]; then
    MyJobOutFile=$1
else
    JobOutFileCount=`ls *.o | wc -l`
    if [ $JobOutFileCount == 1 ]; then
	MyJobOutFile=`find *.o`
    else
	echo "Cannot determine job output file"
	return 1
    fi
fi

ScratchJobDir=$(grep "scratch job directory:" $MyJobOutFile -A 1 | tail -1)
cd "$ScratchJobDir"
