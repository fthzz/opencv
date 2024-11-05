#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

class UnionFind {
private:
    std::vector<int> uf; // 存储每个节点的父节点信息，负数表示根节点，绝对值表示子树的大小

public:
    // 构造函数，初始化并查集
    UnionFind(int n) : uf(n + 1) {
        for (int i = 0; i <= n; ++i) {
            uf[i] = -1; // 初始化时，每个节点都是自己的根节点，子树大小为0,将数组0位置空出
        }
    }

    // 查找p的根节点
    int find(int p) {
        int r = p;
        int temp;
        while(uf[p]>0){
            p=uf[p];
        }
        while(r!=p){  //将路径上所有节点都指向p，生成深度为2的树
            temp=uf[r];
            uf[r]=p;
            r=temp;
        }
        return p;
    }

    // 合并p和q所在的集合
    void unionSet(int p, int q) {
        int proot = find(p);
        int qroot = find(q);
        if (proot != qroot) {
            if (uf[proot] < uf[qroot]) {
                uf[proot] += uf[qroot];
                uf[qroot] = proot;
            } else {
                uf[qroot] += uf[proot];
                uf[proot] = qroot;
            }
        }
    }
};

cv::Mat first_pass(const cv::Mat& labels, UnionFind& uf_set) {
    cv::Mat data = labels.clone();
    int label_counter = 2;
    int offsets[4][2] = {{-1, -1}, {0, -1}, {-1, 0}, {-1, 1}};

    for (int y = 1; y < data.rows - 1; ++y) {
        for (int x = 1; x < data.cols - 1; ++x) {
            //判断是否是黑色背景，是则继续循环
            if (data.at<int>(y, x) == 0) {
                continue;
            }

            std::vector<int> neighbor;
            //循环遍历容器中的变量
            for (int i=0;i<4;i++) {
                int ny = y + offsets[i][0];
                int nx = x + offsets[i][1];
                if (ny >= 0 && ny < data.rows && nx >= 0 && nx < data.cols) {
                    int neighbor_label = data.at<int>(ny, nx);
                    if (neighbor_label != 0) {
                        neighbor.push_back(neighbor_label);
                    }
                }
            }

            std::sort(neighbor.begin(), neighbor.end());
            //去除重复元素
            auto last = std::unique(neighbor.begin(), neighbor.end());
            neighbor.erase(last, neighbor.end());

            if (neighbor.empty()) {
                data.at<int>(y, x) = label_counter++;
            } else if (neighbor.size() == 1) {
                data.at<int>(y, x) = neighbor[0];
            } else {
                data.at<int>(y, x) = neighbor[0];
                for (size_t i = 1; i < neighbor.size(); ++i) {
                    uf_set.unionSet(neighbor[0], neighbor[i]);
                }
            }
        }
    }
    return data;
}

cv::Mat second_pass(cv::Mat& labels, UnionFind& uf_set) {
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            int label = labels.at<int>(y, x);
            if (label != 0) {
                labels.at<int>(y, x) = uf_set.find(label);
            }
        }
    }
    return labels;
}

int main() {
    cv::Mat img = cv::imread("/home/fthzzz/Desktop/yolov8/ultralytics-robotdog/y.png");
    cv::Mat img1;
    img.copyTo(img1);
    
    cv::Mat img_gray;
    cv::cvtColor(img1, img_gray, cv::COLOR_BGR2GRAY);
    cv::Mat img_binary;
    cv::threshold(img_gray, img_binary, 150, 255, cv::THRESH_BINARY);

    int connectivity = 8;
    cv::Mat labels, stats, centroids;
    //label是与image等大的图像
    int retval = cv::connectedComponentsWithStats(img_binary, labels, stats, centroids, connectivity);

    UnionFind set(retval);
    cv::Mat data_process,data_final;
    data_process=first_pass(labels, set);
    data_final=second_pass(data_process, set);

    vector<int> list;
    find_all_index(list,data_final);

    std::srand(static_cast<unsigned>(std::time(0)));
    for(int i=0;i<list.size();i++){
        int label=list[i];
        cv::Vec3b color(std::rand() % 256, std::rand() % 256, std::rand() % 256);
        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                if (labels.at<int>(y, x) == 1) {
                    img1.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                    continue;
                }
                if (labels.at<int>(y, x) == label) {
                    img1.at<cv::Vec3b>(y, x) = colors[0];
                }
            }

        }
    }

    cv::imshow("new", img1);
    cv::waitKey(0);

    return 0;
}

void find_all_index(std::vector<int>& list, const cv::Mat& labels) {
    std::set<int> uniqueLabels;
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            uniqueLabels.insert(labels.at<int>(y, x));
        }
    }
    list.assign(uniqueLabels.begin(), uniqueLabels.end());
}
