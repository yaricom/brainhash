#!/bin/sh
#
# The experiment runner
#


help () {
    echo
    echo "The runner for specific experiment"
    echo "Usage:"
    echo "      experiment.sh config_script"
    echo "          config_script - the experiment configuration script (.py)"
    echo
    
}

if [[ "$#" -lt 1 ]]; then
    help
    exit 0
fi

cd src

/usr/bin/env python3 $1


