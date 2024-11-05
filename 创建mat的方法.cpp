#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


int main(){
    /*
    首先Mat是一个类 它分为两部分，矩阵头和一个矩阵，是由一个指针指向这个矩阵。所以在浅拷贝中，因为矩阵指针不变，所以改变一个会影响另一个，这在深拷贝中就不存在。

    第二，opencv读取的灰度图一般为uchar型，而彩色图像一个像素的数据类型为<Vec3b>类型。灰度图一个像素占一个字节，彩色图一个像素占3个字节；且它们是顺序排布的，所以实际上一个像素点它有三列，分别是bgr。我们用的img.cols是把三列看做一列，所以可以输出正确列数。
    
    第三Mat中采用的是智慧指针，会计算引用次数，当次数为0时自动释放内存。

    第四Mat中主要内存都集中在矩阵上，所以想要进行一些操作，我们可以采用传递指针。Mat中自带属性img.ptr()，是指向矩阵的指针，我们可以利用这个去遍历像素点。

    */
    //1 2 3 方法是浅拷贝，用的是拷贝构造函数，只是复制了一个信息头，mat中储存了相同的地址
    
    //方法1
    Mat img1 = imread('path');

    //方法2
    Mat img2(img1);

    //3
    Mat img3;
    img3 = img1;

    //4
    //最后一个定义图中像素值，如果是单通道，就Scalar（0）
    Mat img4(row, col, CV_8UC1,Scalar(255,255,255));

    //5、6是深拷贝，拷贝了一份全新的数据

    //5
    Mat img5;
    img5.copyTo(img1);

    //6
    Mat img6;
    img6=img1.clone();
}