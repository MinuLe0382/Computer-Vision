#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>

#define WIDTH 396
#define HEIGHT 400

// 가우시안필터 적용 함수
void Gaussian_Blur(const cv::Mat& InputImage, cv::Mat& OutputImage, int kernelSize)
{
    OutputImage = cv::Mat::zeros(InputImage.size(), InputImage.type());
    int radius = kernelSize / 2;
    double sigma = 1; // 표준편차

    // 가우시안 분포를 따르는 값들로 2차원 필터를 만든다. 
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double value_sum = 0.0; // 가우시안 커널의 모든 요소의 합을 계산 (정규화때 사용)
    for (int x = -radius; x <= radius; ++x)
    {
        for (int y = -radius; y <= radius; ++y)
        {
            double value = exp(-(x * x + y * y) / (2.0 * sigma * sigma));
            kernel[x + radius][y + radius] = value;
            value_sum += value;
        }
    }

    // 정규화 (커널의 합이 1이 되어 블러를 적용할경우, 이미지의 밝기를 유지시킴)
    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
            kernel[i][j] /= value_sum;
    }

    // 가우시안 적용 (컨볼루션)
    for (int i = radius; i < InputImage.rows - radius; ++i)
    {
        for (int j = radius; j < InputImage.cols - radius; ++j)
        {
            double blur_sum = 0.0;
            for (int x = -radius; x <= radius; ++x)
            {
                for (int y = -radius; y <= radius; ++y)
                    blur_sum += InputImage.at<uchar>(i + x, j + y) * kernel[x + radius][y + radius];
            }
            OutputImage.at<uchar>(i, j) = static_cast<uchar>(blur_sum);
        }
    }
}

//non maximum suppression 함수
void Mynon_Max_Suppression(const cv::Mat& InputImage, cv::Mat& OutputImage)
{
    std::vector<uchar> input_copy(InputImage.begin<uchar>(), InputImage.end<uchar>());
    std::vector<int> output(WIDTH * HEIGHT, CV_8UC1);
    std::vector<int> direction(WIDTH * HEIGHT, CV_8UC1); // 각도 저장
    double* Gradient = new double[HEIGHT * WIDTH]; // 그래디언트의 크기 저장
    double maximum_grad = 0;

    int sobel_x[3][3] =
    {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int sobel_y[3][3] =
    {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int i = 1; i < HEIGHT - 1; i++)
    {
        for (int j = 1; j < WIDTH - 1; j++)
        {
            // 소벨필터를 적용하는 것은 편미분을 하는 것과 같은 효과
            double gradx = 0;
            double grady = 0;
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    gradx = gradx + sobel_x[1 - x][1 - y] * input_copy[(i + x) * WIDTH + j + y];
                    grady = grady + sobel_y[1 - x][1 - y] * input_copy[(i + x) * WIDTH + j + y];
                }
            }

            // 그래디언트와 그래디언트의 방향을 구한다
            Gradient[i * WIDTH + j] = std::sqrt(std::pow(gradx, 2) + std::pow(grady, 2));
            double atanResult = atan2(gradx, grady) * 180.0 / CV_PI;
            direction[i * WIDTH + j] = (int)(180.0 + atanResult);

            if (Gradient[i * WIDTH + j] > maximum_grad)
                maximum_grad = Gradient[i * WIDTH + j]; // 정규화할때 사용할 그래디언트의 최댓값

            
            // 이미지의 가장자리에서는 소벨필터를 적용할 수 없으므로, 인접한 위치의 그래디언트, 방향을 복사한다.
            if (i == 1)
            {
                Gradient[i * WIDTH + j - 1] = Gradient[i * WIDTH + j];
                direction[i * WIDTH + j - 1] = direction[i * WIDTH + j];
            }
            else if (j == 1)
            {
                Gradient[(i - 1) * WIDTH + j] = Gradient[i * WIDTH + j];
                direction[(i - 1) * WIDTH + j] = direction[i * WIDTH + j];
            }
            else if (i == HEIGHT - 1)
            {
                Gradient[i * WIDTH + j + 1] = Gradient[i * WIDTH + j];
                direction[i * WIDTH + j + 1] = direction[i * WIDTH + j];
            }
            else if (j == WIDTH - 1)
            {
                Gradient[(i + 1) * WIDTH + j] = Gradient[i * WIDTH + j];
                direction[(i + 1) * WIDTH + j] = direction[i * WIDTH + j];
            }

            // 이미지의 모서리에서는 소벨필터를 적용할 수 없으므로, 인접한 모서리의 그래디언트, 방향을 복사한다.
            if (i == 1 && j == 1)
            {
                Gradient[(i - 1) * WIDTH + j - 1] = Gradient[i * WIDTH + j];
                direction[(i - 1) * WIDTH + j - 1] = direction[i * WIDTH + j];
            }
            else if (i == 1 && j == WIDTH - 1)
            {
                Gradient[(i - 1) * WIDTH + j + 1] = Gradient[i * WIDTH + j];
                direction[(i - 1) * WIDTH + j + 1] = direction[i * WIDTH + j];
            }
            else if (i == HEIGHT - 1 && j == 1)
            {
                Gradient[(i + 1) * WIDTH + j - 1] = Gradient[i * WIDTH + j];
                direction[(i + 1) * WIDTH + j - 1] = direction[i * WIDTH + j];
            }
            else if (i == HEIGHT - 1 && j == WIDTH - 1)
            {
                Gradient[(i + 1) * WIDTH + j + 1] = Gradient[i * WIDTH + j];
                direction[(i + 1) * WIDTH + j + 1] = direction[i * WIDTH + j];
            }

            // 각도는 각각 4개(0, 45, 90, 135)로 변환 (반올림을 이용)
            direction[i * WIDTH + j] = round(direction[i * WIDTH + j] / 45) * 45;
            
        }
    }

    // non-maximum-suppression
    for (int i = 1; i < HEIGHT - 1; i++)
    {
        for (int j = 1; j < WIDTH - 1; j++)
        {
            // 수평방향으로 비교 (좌우로 더 큰 그래디언트가 있으면 현재 픽셀의 그래디언트를 0으로)
            if (direction[i * WIDTH + j] == 0 || direction[i * WIDTH + j] == 180)
            {
                if (Gradient[i * WIDTH + j] < Gradient[i * WIDTH + j - 1] || Gradient[i * WIDTH + j] < Gradient[i * WIDTH + j + 1])
                    Gradient[i * WIDTH + j] = 0;
            }
            // 대각선 방향으로 비교 (대각선으로 더 큰 그래디언트가 있으면 현재 픽셀의 그래디언트를 0으로)
            else if (direction[i * WIDTH + j] == 45 || direction[i * WIDTH + j] == 225)
            {
                if (Gradient[i * WIDTH + j] < Gradient[(i + 1) * WIDTH + j + 1] || Gradient[i * WIDTH + j] < Gradient[(i - 1) * WIDTH + j - 1])
                    Gradient[i * WIDTH + j] = 0;
            }
            // 수직방향으로 비교 (위아래로 더 큰 그래디언트가 있으면 현재 픽셀의 그래디언트를 0으로)
            else if (direction[i * WIDTH + j] == 90 || direction[i * WIDTH + j] == 270)
            {
                if (Gradient[i * WIDTH + j] < Gradient[(i + 1) * WIDTH + j] || Gradient[i * WIDTH + j] < Gradient[(i - 1) * WIDTH + j])
                    Gradient[i * WIDTH + j] = 0;
            }
            // 대각선 방향으로 비교 (대각선으로 더 큰 그래디언트가 있으면 현재 픽셀의 그래디언트를 0으로)
            else
            {
                if (Gradient[i * WIDTH + j] < Gradient[(i + 1) * WIDTH + j - 1] || Gradient[i * WIDTH + j] < Gradient[(i - 1) * WIDTH + j + 1])
                    Gradient[i * WIDTH + j] = 0;
            }
            //정규화해서 저장 (그래디언트의 크기가 255 이상인 것이 존재할 수 있음)
            output[i * WIDTH + j] = (int)(Gradient[i * WIDTH + j] * (255.0 / maximum_grad));
        }
    }

    for (int i = 1; i < HEIGHT - 1; i++) // 높이를 위한 반복문
    {
        for (int j = 1; j < WIDTH - 1; j++) // 너비를 위한 반복문
            OutputImage.at<uchar>(i, j) = static_cast<uchar>(output[i * WIDTH + j]);
    }
}

void DoubleThreshold_EdgeTracking(cv::Mat& suppressed, cv::Mat& edge_image, int lowThreshold, int highThreshold)
{
    edge_image = cv::Mat::zeros(suppressed.size(), CV_8UC1);
    for (int i = 1; i < suppressed.rows - 1; i++)
    {
        for (int j = 1; j < suppressed.cols - 1; j++)
        {
            if (suppressed.at<uchar>(i, j) >= highThreshold)
                edge_image.at<uchar>(i, j) = 255; // 이미지의 픽셀값이 highThreshold이상이면 edge로 설정 (255)
            else if (suppressed.at<uchar>(i, j) >= lowThreshold)
            {   // 현재 픽셀 값이 lowThreshold 이상 highThreshold 미만이면, 인접한 픽셀 중 highThreshold를 넘는 픽셀이 있는지 찾는다
                // 만일 하나라도 highThreshold를 넘는 픽셀이 있으면 현재픽셀은 확인한 경계선과 연결된 약한 경계선이다.
                int has_strong_neighbor = 0;
                for (int dx = -1; dx <= 1; dx++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        if (suppressed.at<uchar>(i + dx, j + dy) >= highThreshold)
                        {
                            has_strong_neighbor = 1;
                            break;
                        }
                    }
                    if (has_strong_neighbor)
                        break;
                }
                if (has_strong_neighbor)
                    edge_image.at<uchar>(i, j) = 255;
            }
        }
    }
}

// 그래디언트 기반 허프변환함수 (구현실패)
void HoughGradientVoting(const cv::Mat& edge_Image, std::vector<int>& x_point, std::vector<int>& y_point)
{
    cv::Mat line_draw = cv::Mat::zeros(edge_Image.size(), CV_32F); // 투표용 행렬 초기화

    for (int i = 1; i < HEIGHT - 1; ++i)
    {
        for (int j = 1; j < WIDTH - 1; ++j)
        {
            if (edge_Image.at<uchar>(i, j) == 255) // 에지 점 찾기
            {
                // Sobel 필터를 이용하여 그래디언트 계산
                float grad_x = (edge_Image.at<uchar>(i + 1, j - 1) + 2 * edge_Image.at<uchar>(i + 1, j) + edge_Image.at<uchar>(i + 1, j + 1)) -
                    (edge_Image.at<uchar>(i - 1, j - 1) + 2 * edge_Image.at<uchar>(i - 1, j) + edge_Image.at<uchar>(i - 1, j + 1));

                float grad_y = (edge_Image.at<uchar>(i - 1, j + 1) + 2 * edge_Image.at<uchar>(i, j + 1) + edge_Image.at<uchar>(i + 1, j + 1)) -
                    (edge_Image.at<uchar>(i - 1, j - 1) + 2 * edge_Image.at<uchar>(i, j - 1) + edge_Image.at<uchar>(i + 1, j - 1));

                // 그래디언트 방향 계산
                float direction = std::atan2(grad_y, grad_x);

                // 투표 (직선을 따라)
                for (int r = 0; r <= 70; ++r) // 직선그리는 모양을 결정
                {
                    int x = i + r * std::cos(direction);
                    int y = j + r * std::sin(direction);

                    if (x >= 0 && x < HEIGHT && y >= 0 && y < WIDTH)
                        line_draw.at<float>(x, y) += 1;
                }
            }
        }
    }
    // 그래디언트로 인한 직선이 어떻게 찍히는지 확인하기위한 용도
    /*
    cv::Mat overlap_line_display;
    double min, max;
    cv::minMaxLoc(line_draw, &min, &max); // 최솟값과 최댓값 찾기
    line_draw.convertTo(overlap_line_display, CV_8U, 255 / max);

    // 결과 이미지 보여주기
    cv::imshow("Overlap Line", overlap_line_display);
    cv::waitKey(0);
    */

    for (int i = 1; i < HEIGHT - 1; ++i)
    {
        for (int j = 1; j < WIDTH - 1; ++j)
        {
            if (line_draw.at<float>(i, j) > 13)
            {
                std::cout << "Value at (" << i << ", " << j << "): " << line_draw.at<float>(i, j) << std::endl;
                int close_to_existing_center = 0;
                for (size_t k = 0; k < x_point.size(); ++k)
                {
                    if (std::abs(x_point[k] - i) < 25 && std::abs(y_point[k] - j) < 25)
                    {
                        close_to_existing_center = 1;
                        break;
                    }
                }
                if (!close_to_existing_center)
                {
                    x_point.push_back(i);
                    y_point.push_back(j);
                }
            }
        }
    }
}

// (a, b, r) 투표기반 허프변환
void MyHoughCircles(const cv::Mat& edge_Image, std::vector<cv::Vec3f>& circles, int minRad, int maxRad)
{
    std::vector<int> x_point, y_point, rad_len; // 원의 중심과 반지름을 저장할 백터

    for (int rad = minRad; rad <= maxRad; rad++) // 반지름의 범위를 알면 투표시간을 단축시킬 수 있다.
    {
        cv::Mat circle_draw = cv::Mat::zeros(edge_Image.size(), CV_32F); // 결과저장용
        for (int i = 0; i < edge_Image.rows; i++)
        {
            for (int j = 0; j < edge_Image.cols; j++)
            {
                if (edge_Image.at<uchar>(i, j) == 255) // EDGE가 있는 점을 찾는다
                {
                    for (int angle = 0; angle < 360; angle++) // 모든 방향에 대하여 연산
                    {   // 주어진 좌표 i, j를 중심으로 반지름이 rad인 원의 둘레에 위치한 점의 좌표 (a, b)를 계산
                        int a = i - rad * std::cos((angle * CV_PI) / 180.0); // 1도 = pi / 180 rad
                        int b = j - rad * std::sin((angle * CV_PI) / 180.0);
                        if (a >= 0 && a < edge_Image.rows && b >= 0 && b < edge_Image.cols) // 범위 제한
                            circle_draw.at<float>(a, b)++;
                    }
                }
            }
        }

        int temp = 170; //투표수의 임계점
        for (int i = 0; i < circle_draw.rows; i++)
        {
            for (int j = 0; j < circle_draw.cols; j++)
            {
                if (circle_draw.at<float>(i, j) > temp)
                {
                    x_point.push_back(i); // 전역변수벡터의 끝에 중심의 좌표와 반지름을 append
                    y_point.push_back(j);
                    rad_len.push_back(rad);
                }
            }
        }
    }

    int Center_num = x_point.size(); // 중복된 중심좌표는 제거필요, 총 중심좌표갯수
    for (int i = 0; i < x_point.size(); i++)
    {
        int checker_x = x_point[i];
        int checker_y = y_point[i];
        for (int j = 0; j < Center_num; j++) // 다른 중심좌표와 비교한다.
        {
            if (i != j)
            {
                if ((x_point[j] + 15 >= checker_x && y_point[j] + 15 >= checker_y) && (x_point[j] - 15 <= checker_x && y_point[j] - 15 <= checker_y))
                { // 15픽셀 이내에 있으면 두 원의 중심이 매우 가까운 것으로 판단하여 제거
                    x_point.erase(x_point.begin() + j);
                    y_point.erase(y_point.begin() + j);
                    rad_len.erase(rad_len.begin() + j);

                    Center_num--; // 갯수 감소
                    j = 0; // 초기화
                    i = 0;
                    checker_x = x_point[i];
                    checker_y = y_point[i];
                }
            }
        }
    }

    for (size_t i = 0; i < x_point.size(); i++)
        circles.push_back(cv::Vec3f(y_point[i], x_point[i], rad_len[i]));
    // circles에는 원의 중심좌표, 반지름이 벡터형식으로 저장
}


int main(void)
{
    std::ifstream file; // 파일 읽기
    file.open("coins_396x400.raw", std::ios::binary);
    if (!file.is_open())
    {
        printf("Cannot open file\n");
        return -1;
    }

    uchar* buffer = new uchar[WIDTH * HEIGHT];
    file.read(reinterpret_cast<char*>(buffer), WIDTH * HEIGHT);
    file.close();

    cv::Mat grayImage(HEIGHT, WIDTH, CV_8UC1, buffer); // 원본 이미지
    cv::Mat blurredImage;
    Gaussian_Blur(grayImage, blurredImage, 5); // 가우시안 필터를 통과하여 노이즈를 감소시킨 이미지

    cv::Mat magnitude, angle;
    cv::imshow("Original Image", grayImage);
    cv::imshow("Blurred Image", blurredImage);
    cv::waitKey(0);

    cv::Mat suppressed2(HEIGHT, WIDTH, CV_8UC1);
    Mynon_Max_Suppression(blurredImage, suppressed2);
    cv::imshow("suppressed2", suppressed2); // 이미지의 얇은 선을 검출한다
    cv::waitKey(0);

    cv::Mat edges2; // 최종 엣지가 저장되는 객체
    DoubleThreshold_EdgeTracking(suppressed2, edges2, 30, 55); // 임계값을 두개 사용하여 검출
    cv::imshow("edges2", edges2);
    cv::waitKey(0);

    std::vector<cv::Vec3f> circles;

    std::time_t time_func_call = std::time(nullptr); // MyHoughCircles 시작 시간 측정
    MyHoughCircles(edges2, circles, 20, 100); // (a, b, r)투표 기반의 허프변환
    std::time_t time_func_close = std::time(nullptr); // MyHoughCircles 종료 시간 측정
    std::cout << "Execution time of  MyHoughCircles(): " << difftime(time_func_close, time_func_call) << " sec" << std::endl; // 소요 시간 출력

    cv::Mat detected_circle;
    cvtColor(grayImage, detected_circle, cv::COLOR_GRAY2BGR); // 원본이미지 위에 검출한 원의 그림 출력

    for (const cv::Vec3f& c : circles)
    {
        cv::Point center(c[0], c[1]);
        int radius = c[2];
        std::cout << "Center: (" << center.x << ", " << center.y << "), Radius: " << radius << std::endl;
        cv::circle(detected_circle, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        cv::circle(detected_circle, center, radius, cv::Scalar(255, 0, 50), 2, cv::LINE_AA);
    }

    cv::imshow("Detected Circles", detected_circle);
    cv::waitKey(0);

    // gradient method 방식의 허프변환결과 (구현실패)
    /*
    std::vector<int> center_x2, center_y2;
    std::time_t time_func2_call = std::time(nullptr); // MyHoughCircles 시작 시간 측정
    HoughGradientVoting(edges2, center_x2, center_y2);
    std::time_t time_func2_close = std::time(nullptr); // MyHoughCircles 종료 시간 측정
    std::cout << "Execution time of  HoughGradientVoting(): " << difftime(time_func2_close, time_func2_call) << " sec" << std::endl; // 소요 시간 출력

    cv::Mat result;
    cv::cvtColor(edges2, result, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < center_x2.size(); ++i)
    {
        cv::Point center(center_y2[i], center_x2[i]);
        cv::circle(result, center, 3, cv::Scalar(0, 255, 0), -1);
    }
    cv::imshow("Detected Centers", result);
    cv::waitKey(0);
    */

    return 0;
}