#!/bin/bash

#cd GUI && python RoboConGui.py &
#P1=$!
#cd RoboConVision && ./RoboConVision && fg
#P2=$!
#wait $P1 $P2


#gnome-terminal -e 'bash -c "cd RoboConVision && ./RoboConVision";exec bash"'
#gnome-terminal -e 'bash -c "second-script.sh; exec bash"'
#cd GUI && python RoboConGui.py & 
#cd RoboConVision && ./RoboConVision

#trap "exit" INT TERM ERR
#trap "kill 0" EXIT

#gnome-terminal -x sh -c "./RoboCon; exit" &

#cd GUI && python RoboConGui.py &
#P1=$!
#gnome-terminal -x sh -c "./RoboCon; exec bash" &
#P2=$!
#gnome-terminal -- /bin/sh -c "python3.6 client.py; exec bash" &
#P3=$!

cd GUI && python RoboConGui.py &
P1=$!
gnome-terminal -- /bin/sh -c "../../build/RoboCon; exec bash" &
P2=$!
gnome-terminal -- /bin/sh -c "python3.6 main.py; exec bash" &
P3=$!

wait $P1 $P2 $P3

