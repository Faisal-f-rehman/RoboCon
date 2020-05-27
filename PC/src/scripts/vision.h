#ifndef VISION_H
#define VISION_H

#include "guicomms.h"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>



#define WIN_NAME_ORG_IMG "Original Image"
#define WIN_NAME_BLUE_IMG "Blue filter Image"
#define WIN_NAME_EDGE_IMG "Edge filter Image"

#define WIN_NAME_YELLOW_BLOB "Yellow_Blobs_Found"
#define WIN_NAME_RED_BLOB "Red_Blobs_Found"
#define WIN_NAME_UNKNOWN_BLOB "Colour_Unknown_Blobs_Found"





class Vision
{
private:
    const std::string _redCalFile = "./cal_files/red_cal_file.txt";
    const std::string _yellowCalFile = "./cal_files/yellow_cal_file.txt";
    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    //___Holds yellow and red disk locations
    typedef struct{
        std::vector<float> yellow_x;
        std::vector<float> yellow_y;
        std::vector<float> red_x;
        std::vector<float> red_y;
    } StructDiskLoc;
    StructDiskLoc DiskLoc;

public:
    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    //___Constructor
    Vision();
    //___Destructor
    ~Vision();

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    typedef cv::Mat Mat;    //___Define Mat as type cv::Mat (Matrix)

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    //___Public member functions / methods
    void alignBoard();      //___Function to align the board
    Mat yellowDisk(Mat* img);       //___Function to find yellow coins on the board
    Mat redDisk(Mat* img);          //___Function to find red coins on the board
    uint8_t getCamID();        //___Grab camera ID
    void calibration();
    std::vector<uint8_t> getBoardState();
    void visionWorker();
    void init(const uint8_t CameraID = 0);

private:
    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    // std::vector<uint8_t> _boardState;
    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    struct StateRecord{
        bool _total[42];//___holds states filled by disks
        bool _yellow[42];//___holds states filled by yellow disks
        bool _red[42];//___holds states filled by red disks
    };StateRecord stateRec;
    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    //___Camera Struct
    typedef struct{
        Mat _imgOrg, _imgYellow, _imgRed, _imgBlue;
        uint8_t _ID;
    } _Camera;
    _Camera _Cam;

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    //___Private member functions / methods
    Mat _findBlue(Mat* imgOrg);         //___Filters out all colours except for blue (works in HSV space)
    Mat _findYellow(Mat* imgOrg);       //___Filters out all colours except for yellow (works in HSV space)
    Mat _findRed(Mat* imgOrg);       //___Filters out all colours except for red (works in HSV space)
    Mat _detectBlob(const Mat Img, bool showImage, cv::String colour);
    Mat _edgeDetection(Mat* img);       //___Detects edges in a given image
    void _calcGrid();
    void _drawGrid(Mat* Img);           //___Draws a checkered grid on the image provided
    StructDiskLoc _keypoints2DiskLoc(std::string colour, const std::vector<cv::KeyPoint> keypoints);
    void _stateLocator(StructDiskLoc* keyPoints);
    StateRecord _getStateRecord();       //___Returns record of the current same status
    void _printBoard(std::vector<uint8_t> _board);
    void _saveRedCal2File();
    void _saveYellowCal2File();
    void _loadRedCalFile();
    void _loadYellowCalFile();

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    //___Hold all the properties of a grid (used in _drawGrid function)
    typedef struct{
        typedef unsigned short uShort;              //___Define uShort as type unsigned short
        const uShort thickness = 2;                 //___Thickness of the lines
        const uShort lineColour[3] = {255,255,0};   //___Colour of the lines
        const uShort lineType = 8;                  //___Linetype 8 is a simple continous line
        const uShort nOfLines_x = 6;                //___Number of vertical lines
        const uShort nOfLines_y = 5;                //___Number of horizontal lines
        const uShort width = 480;                   //___Total width of the grid
        const uShort height = 640;                  //___Total height of the grid
        const uShort gridDisX = 86;                 //___Distance between vertical lines
        const uShort gridDisY = 78;                 //___Distance between horizontal lines
        uShort x = 0;
        uShort y = 0;
        ushort xlim[6];
        ushort ylim[5];
    } __Grid;
    __Grid _Grid;

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    //___Hold all the properties of blob for blobDetection
    typedef struct{
        const uint8_t minThreshold = 10;
        const uint8_t maxThreshold = 150;
        const int minArea = 1200;
        const int maxArea = 3500;
        const float minCircularity = float(0.1);
        // const float minConvexity = float(0.4);
        // const float maxConvexity = float(0.95);
        const float minConvexity = float(0.2);
        const float maxConvexity = float(0.99);
        const float minInertia = float(0.01);
    } _Blob;
    _Blob Blob;


    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    //===reference: https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
    //___Colour Detection HSV values
    int _red_hsv[12] = {0, 120, 60,       //___Red lower threshold left of spectrum
                                  10, 255, 255,     //___Red upper threshold left of spectrum
                                  170, 120, 60,     //___Red lower threshold right of spectrum
                                  180, 255, 255};   //___Red upper threshold right of spectrum

    int _yellow_hsv[6] = {20, 180, 60,    //___Yellow lower threshold
                                    50, 255, 255};  //___Yellow upper threshold
    //Need to find HSV values
    int _blue_hsv[6] = {100, 150, 0,      //___Blue lower threshold
                                  130, 255, 255};   //___Blue upper threshold
};

extern Vision vision;

#endif // VISION_H
