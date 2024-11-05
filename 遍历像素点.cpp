#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

/*
Mat中自带属性img.ptr()，是指向矩阵的指针，我们可以利用这个去遍历像素点。

*/

int main()
{
    Mat img = imread("/home/fthzzz/Desktop/yolov8/ultralytics-robotdog/y.png");
    Mat img_gary;
    cvtColor(img, img_gary, COLOR_BGR2GRAY);

    int row = img.rows;
    int col = img.cols;


    //对于灰度图的遍历
    for (int i = 0; i < row;++i){
        //这个指针相当于二维数组中的行指针
        uchar *row_ptr = img_gary.ptr<uchar>(i);
        for (int j = 0; j < col; ++j)
        {
            uchar pixel = row_ptr[j];
            cout << (int)pixel << endl;
        }
    }

    Mat img2(row, col, CV_8UC1, Scalar(0, 0, 0));

    //对于彩图的遍历

    //法1 针对行指针的使用
    for (int i = 0; i < row; ++i)
    {
        uchar *row_ptr = img.ptr<uchar>(i);
        uchar *row_ptr2 = img2.ptr<uchar>(i);
        for (int j = 0; j < col * img.channels(); ++j) // 因为是3通道，所以列数需要乘3才可以遍历全部
        {
            //输出时顺序是bgr
            uchar pixel = row_ptr[j];
            cout << (int)pixel << endl;

            // 赋值操作
            // cv::saturate_cast 函数可以进行类型转化，同时保证不会溢出，超过时会自动截取为255
            row_ptr2[j] = saturate_cast<uchar>((int)row_ptr[j] + 100);
        }
    }

    //法2 直接遍历
    for (int i = 0; i < row; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            //同样 要进行赋值操作时 用saturate_cast 函数
            cout << (int)img.at<Vec3b>(i, j)[0]<<endl;
        }
    }
}