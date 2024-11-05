#include <opencv2/opencv.hpp>
#include <iostream>
#include<vector>

using namespace cv;
using namespace std;

/*
lab色彩空间中，a表示绿色到红色的分量，b表示蓝色到黄色的分量
当你要提取的颜色 是红黄蓝混合而成时，需要同时调节ab，这很不方便
但你只要提取红、黄、蓝，就只需要调节a或b
例如你想调节黄，因为黄只在b中，所以只需要调b
*/

int main(){
    Mat img,dst;
    Mat img_lab;
    img = imread("/home/fthzzz/Desktop/yolov8/ultralytics-robotdog/y.png");
    // resize(img, img, Size(108, 104), 0, 0);
    cvtColor(img, img_lab, COLOR_BGR2Lab);
    
    int row = img.rows;
    int col = img.cols;
    cout << row;

    Mat img_lab_b(row, col, CV_8UC1);
    
    int chan = 2;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            img_lab_b.at<uchar>(i, j) = img_lab.at<Vec3b>(i, j)[chan];   
        }
    }

    img_lab_b.convertTo(img_lab_b, CV_8UC1);

    /*
    cv_8uc1
    cv_8表示一个像素点8位
    u表示 unsigned int
    c1表示一个通道。   3通道rgb图就是c3
    */
    Mat img_lab_b1(row, col, CV_8UC1);

    uchar **ptr = (uchar **)malloc(row * sizeof(uchar *));
    for (int i = 0;i<row;i++){
        ptr[i] = (uchar *)malloc(col * sizeof(uchar));
    }

    uchar *ptmp = NULL;

    for (int i = 0; i < row;i++){
        ptmp = img_lab_b1.ptr<uchar>(i);
        for (int j = 0; j < col;j++)
        {
            ptr[i][j] = img_lab_b.at<uchar>(i, j);

            if(ptr[i][j]>200){
                ptr[i][j] = 255;
            }
            else{
                ptr[i][j] = 0;
            }

            ptmp[j] = ptr[i][j];
        }
    }
    imshow("new", img_lab_b1);
    waitKey(0);
}