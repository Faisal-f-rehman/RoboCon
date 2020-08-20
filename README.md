# **RoboCon** - ***Gaming Robot***

[![RoboCon](https://img.youtube.com/vi/zAyr8a5UKs0/0.jpg)](https://www.youtube.com/watch?v=zAyr8a5UKs0?t=516)

<br>

------------------------------
## Section 1 - Dependencies
------------------------------
The software was developed on Linux Ubuntu 18.04 LTS (PC) and Raspbian (RPI) using the following:

<br>

|Application / Library|	Version|
| :---:               |  :---: |
|    **PC**                        |
|------------------------------|
|Cmake|	3.5|
|Python|	3.6.9|
|Tensorflow|	1.15.0|
|Keras 	|2.3.1|
|Pygame	|1.9.6|
|Tqdm	|4.45.0|
|NumPy	|1.18.2|
|OpenCV (python)	|4.2.0.34|
|Matplotlib	|3.2.1|
|Boost (C++)	|1.65.1|
|------------------------------|
|**Raspberry Pi**|
|------------------------------|
|Python	|3.7.3|
|adafruit-circuitpython-servokit^	|1.1.1|

^ May require additional libraries, please follow the link for details:  
https://learn.adafruit.com/adafruit-16-channel-pwm-servo-hat-for-raspberry-pi/

Please check following link for vision system dependencies:  
https://github.com/Faisal-f-rehman/10538828_RoboConVision


<br>

------------------------------
## Section 2 - Setup
------------------------------
The following instructions assume that all dependencies listed in section 1 have been satisfied.

### 2.1	Get Repository
First get a copy of the source code from Github:
  - ssh:
```shell
$ git clone git@github.com:Faisal-f-rehman/RoboCon.git
```
  - https:
```shell
$ git clone https://github.com/Faisal-f-rehman/RoboCon.git
```

<br>

### 2.2	Raspberry Pi
* In the RoboCon folder there are two sub-folders, PC and RPI. First Copy the RPI folder onto the Raspberry Pi. Then open terminal (ctrl + alt + t) and change directory into RPI folder (this may vary depending on where you have placed the RPI folder on your Raspberry Pi):

```shell
$ cd RPI
```

* Execute python scripts:
```shell
$ python3 main.py

      Or

$ sudo python3 main.py
```

<br>

### 2.3	PC
#### 2.3.1	Setup with bash script
* Open terminal (ctrl + alt + t) and change directory to PC in the RoboCon folder:
```shell
$ cd RoboCon/PC
```

*	Make bash (shell) script executable:
```shell
$ chmod +x RoboCon.sh
```

*	Execute bash (shell) script^:
```shell
$ ./RoboCon.sh
```

*	Enter y on the terminal and press enter (to skip generating build files, enter any other key):
```shell
$ y
```

^ This executes a bash script that generates build files using cmake, builds the files using make, makes another bash script executable located in the scripts (RoboCon/PC/src/scripts/roboCon.sh) folder and executes it. This saves the hassle of changing directories between build and src folders for any subsequent builds and runs.

<br>

#### 2.3.2	Setup without bash script
For manual setup instead of using the bash script provided, use the following instructions:

*	Open terminal (ctrl + alt + t) and change directory to PC in the RoboCon folder:
```shell
$ cd RoboCon/PC
```

*	Create build directory:
```shell
$ mkdir build
```

*	Generate build files:
```shell
$ cmake ../src/scripts
```

*	Build generated files:
```shell
$ make
```

*	Change directory to scripts directory:
```shell
$ cd ../src/scripts
```

*	Make roboCon.sh bash script executable:
```shell
$ chmod +x roboCon.sh
```

*	Execute program:
```shell
$ ./roboCon.sh
```
