#!/bin/bash

PYTHON35_CONDA=/home/micou/.conda/envs/python35/bin/python3.5

while :
do
	if ps ax | grep -v grep | grep "$PYTHON35_CONDA main.py" > /dev/null
	then
		sleep 5
	else
		$PYTHON35_CONDA main.py
		sleep 5
	fi
done
