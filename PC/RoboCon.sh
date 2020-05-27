#!/bin/bash
echo ""
echo "|--        |"
echo "Please enter 'y' if build files need generating"
read buildReq
if [[ $buildReq == y ]]
then
	echo ""	
	echo "Generating build files"
	if [ -d "build" ]
	then
		rm -r build
	fi
	echo "|----      |"
	mkdir build
	cd build && cmake ../src/scripts && cd ../
	echo ""	
	echo "Build files generated"
	echo "|------    |"
	echo ""
fi
if [ -d "build" ]
then
cd build
echo "|--------  |"
else
	echo ""
	echo "Build files not found"
	echo "Generating build files"
	mkdir build
	cd build && cmake ../src/scripts
	echo ""	
	echo "Build files generated"
	echo "|--------  |"
	echo ""
fi
make
echo "|----------|"
echo "---->>> now running..."
cd ../src/scripts && chmod +x roboCon.sh && ./roboCon.sh

