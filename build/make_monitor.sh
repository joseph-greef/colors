#!/bin/bash

if [[ ! -f Makefile ]]; then
    cmake ..
    make
fi

while true; do
    inotifywait -e modify,delete,move --exclude "..\/build|..\/\.git" -r ../ && \
        clear && \
        make -j 12
    if [ "$1" == "autostart" ] && [ $? -eq 0 ]; then
        ./bin/colors "${@:2}"
    fi
done
