#!/bin/bash

JobFilePattern="job*.o"

if [ "$1" != "" ]; then
    MyJobOutFile=$1
else
    if ls $JobFilePattern &> /dev/null ; then
	    JobOutFileCount=`ls $JobFilePattern | wc -l`
	    if [ $JobOutFileCount == 1 ]; then
		MyJobOutFile=`find $JobFilePattern`
	    else
		echo "Could not determine which job output files to use. Please specify one:"
		ls -tr $JobFilePattern
		return 2
	    fi
    else
	    echo "Could not find job output file."
	    return 1

    fi
fi

ScratchJobDir=$(grep "scratch job directory:" $MyJobOutFile -A 1 | tail -1)
cd "$ScratchJobDir"
