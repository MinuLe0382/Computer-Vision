#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>

#define WIDTH 396
#define HEIGHT 400

// ����þ����� ���� �Լ�
void Gaussian_Blur(const cv::Mat& InputImage, cv::Mat& OutputImage, int kernelSize)
{
    OutputImage = cv::Mat::zeros(InputImage.size(), InputImage.type());
    int radius = kernelSize / 2;
    double sigma = 1; // ǥ������

    // ����þ� ������ ������ ����� 2���� ���͸� �����. 
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double value_sum = 0.0; // ����þ� Ŀ���� ��� ����� ���� ��� (����ȭ�� ���)
    for (int x = -radius; x <= radius; ++x)
    {
        for (int y = -radius; y <= radius; ++y)
        {
            double value = exp(-(x * x + y * y) / (2.0 * sigma * sigma));
            kernel[x + radius][y + radius] = value;
            value_sum += value;
        }
    }

    // ����ȭ (Ŀ���� ���� 1�� �Ǿ� ���� �����Ұ��, �̹����� ��⸦ ������Ŵ)
    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
            kernel[i][j] /= value_sum;
    }

    // ����þ� ���� (�������)
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

//non maximum suppression �Լ�
void Mynon_Max_Suppression(const cv::Mat& InputImage, cv::Mat& OutputImage)
{
    std::vector<uchar> input_copy(InputImage.begin<uchar>(), InputImage.end<uchar>());
    std::vector<int> output(WIDTH * HEIGHT, CV_8UC1);
    std::vector<int> direction(WIDTH * HEIGHT, CV_8UC1); // ���� ����
    double* Gradient = new double[HEIGHT * WIDTH]; // �׷����Ʈ�� ũ�� ����
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
            // �Һ����͸� �����ϴ� ���� ��̺��� �ϴ� �Ͱ� ���� ȿ��
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

            // �׷����Ʈ�� �׷����Ʈ�� ������ ���Ѵ�
            Gradient[i * WIDTH + j] = std::sqrt(std::pow(gradx, 2) + std::pow(grady, 2));
            double atanResult = atan2(gradx, grady) * 180.0 / CV_PI;
            direction[i * WIDTH + j] = (int)(180.0 + atanResult);

            if (Gradient[i * WIDTH + j] > maximum_grad)
                maximum_grad = Gradient[i * WIDTH + j]; // ����ȭ�Ҷ� ����� �׷����Ʈ�� �ִ�

            
            // �̹����� �����ڸ������� �Һ����͸� ������ �� �����Ƿ�, ������ ��ġ�� �׷����Ʈ, ������ �����Ѵ�.
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

            // �̹����� �𼭸������� �Һ����͸� ������ �� �����Ƿ�, ������ �𼭸��� �׷����Ʈ, ������ �����Ѵ�.
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

            // ������ ���� 4��(0, 45, 90, 135)�� ��ȯ (�ݿø��� �̿�)
            direction[i * WIDTH + j] = round(direction[i * WIDTH + j] / 45) * 45;
            
        }
    }

    // non-maximum-suppression
    for (int i = 1; i < HEIGHT - 1; i++)
    {
        for (int j = 1; j < WIDTH - 1; j++)
        {
            // ����������� �� (�¿�� �� ū �׷����Ʈ�� ������ ���� �ȼ��� �׷����Ʈ�� 0����)
            if (direction[i * WIDTH + j] == 0 || direction[i * WIDTH + j] == 180)
            {
                if (Gradient[i * WIDTH + j] < Gradient[i * WIDTH + j - 1] || Gradient[i * WIDTH + j] < Gradient[i * WIDTH + j + 1])
                    Gradient[i * WIDTH + j] = 0;
            }
            // �밢�� �������� �� (�밢������ �� ū �׷����Ʈ�� ������ ���� �ȼ��� �׷����Ʈ�� 0����)
            else if (direction[i * WIDTH + j] == 45 || direction[i * WIDTH + j] == 225)
            {
                if (Gradient[i * WIDTH + j] < Gradient[(i + 1) * WIDTH + j + 1] || Gradient[i * WIDTH + j] < Gradient[(i - 1) * WIDTH + j - 1])
                    Gradient[i * WIDTH + j] = 0;
            }
            // ������������ �� (���Ʒ��� �� ū �׷����Ʈ�� ������ ���� �ȼ��� �׷����Ʈ�� 0����)
            else if (direction[i * WIDTH + j] == 90 || direction[i * WIDTH + j] == 270)
            {
                if (Gradient[i * WIDTH + j] < Gradient[(i + 1) * WIDTH + j] || Gradient[i * WIDTH + j] < Gradient[(i - 1) * WIDTH + j])
                    Gradient[i * WIDTH + j] = 0;
            }
            // �밢�� �������� �� (�밢������ �� ū �׷����Ʈ�� ������ ���� �ȼ��� �׷����Ʈ�� 0����)
            else
            {
                if (Gradient[i * WIDTH + j] < Gradient[(i + 1) * WIDTH + j - 1] || Gradient[i * WIDTH + j] < Gradient[(i - 1) * WIDTH + j + 1])
                    Gradient[i * WIDTH + j] = 0;
            }
            //����ȭ�ؼ� ���� (�׷����Ʈ�� ũ�Ⱑ 255 �̻��� ���� ������ �� ����)
            output[i * WIDTH + j] = (int)(Gradient[i * WIDTH + j] * (255.0 / maximum_grad));
        }
    }

    for (int i = 1; i < HEIGHT - 1; i++) // ���̸� ���� �ݺ���
    {
        for (int j = 1; j < WIDTH - 1; j++) // �ʺ� ���� �ݺ���
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
                edge_image.at<uchar>(i, j) = 255; // �̹����� �ȼ����� highThreshold�̻��̸� edge�� ���� (255)
            else if (suppressed.at<uchar>(i, j) >= lowThreshold)
            {   // ���� �ȼ� ���� lowThreshold �̻� highThreshold �̸��̸�, ������ �ȼ� �� highThreshold�� �Ѵ� �ȼ��� �ִ��� ã�´�
                // ���� �ϳ��� highThreshold�� �Ѵ� �ȼ��� ������ �����ȼ��� Ȯ���� ��輱�� ����� ���� ��輱�̴�.
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

// �׷����Ʈ ��� ������ȯ�Լ� (��������)
void HoughGradientVoting(const cv::Mat& edge_Image, std::vector<int>& x_point, std::vector<int>& y_point)
{
    cv::Mat line_draw = cv::Mat::zeros(edge_Image.size(), CV_32F); // ��ǥ�� ��� �ʱ�ȭ

    for (int i = 1; i < HEIGHT - 1; ++i)
    {
        for (int j = 1; j < WIDTH - 1; ++j)
        {
            if (edge_Image.at<uchar>(i, j) == 255) // ���� �� ã��
            {
                // Sobel ���͸� �̿��Ͽ� �׷����Ʈ ���
                float grad_x = (edge_Image.at<uchar>(i + 1, j - 1) + 2 * edge_Image.at<uchar>(i + 1, j) + edge_Image.at<uchar>(i + 1, j + 1)) -
                    (edge_Image.at<uchar>(i - 1, j - 1) + 2 * edge_Image.at<uchar>(i - 1, j) + edge_Image.at<uchar>(i - 1, j + 1));

                float grad_y = (edge_Image.at<uchar>(i - 1, j + 1) + 2 * edge_Image.at<uchar>(i, j + 1) + edge_Image.at<uchar>(i + 1, j + 1)) -
                    (edge_Image.at<uchar>(i - 1, j - 1) + 2 * edge_Image.at<uchar>(i, j - 1) + edge_Image.at<uchar>(i + 1, j - 1));

                // �׷����Ʈ ���� ���
                float direction = std::atan2(grad_y, grad_x);

                // ��ǥ (������ ����)
                for (int r = 0; r <= 70; ++r) // �����׸��� ����� ����
                {
                    int x = i + r * std::cos(direction);
                    int y = j + r * std::sin(direction);

                    if (x >= 0 && x < HEIGHT && y >= 0 && y < WIDTH)
                        line_draw.at<float>(x, y) += 1;
                }
            }
        }
    }
    // �׷����Ʈ�� ���� ������ ��� �������� Ȯ���ϱ����� �뵵
    /*
    cv::Mat overlap_line_display;
    double min, max;
    cv::minMaxLoc(line_draw, &min, &max); // �ּڰ��� �ִ� ã��
    line_draw.convertTo(overlap_line_display, CV_8U, 255 / max);

    // ��� �̹��� �����ֱ�
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

// (a, b, r) ��ǥ��� ������ȯ
void MyHoughCircles(const cv::Mat& edge_Image, std::vector<cv::Vec3f>& circles, int minRad, int maxRad)
{
    std::vector<int> x_point, y_point, rad_len; // ���� �߽ɰ� �������� ������ ����

    for (int rad = minRad; rad <= maxRad; rad++) // �������� ������ �˸� ��ǥ�ð��� �����ų �� �ִ�.
    {
        cv::Mat circle_draw = cv::Mat::zeros(edge_Image.size(), CV_32F); // ��������
        for (int i = 0; i < edge_Image.rows; i++)
        {
            for (int j = 0; j < edge_Image.cols; j++)
            {
                if (edge_Image.at<uchar>(i, j) == 255) // EDGE�� �ִ� ���� ã�´�
                {
                    for (int angle = 0; angle < 360; angle++) // ��� ���⿡ ���Ͽ� ����
                    {   // �־��� ��ǥ i, j�� �߽����� �������� rad�� ���� �ѷ��� ��ġ�� ���� ��ǥ (a, b)�� ���
                        int a = i - rad * std::cos((angle * CV_PI) / 180.0); // 1�� = pi / 180 rad
                        int b = j - rad * std::sin((angle * CV_PI) / 180.0);
                        if (a >= 0 && a < edge_Image.rows && b >= 0 && b < edge_Image.cols) // ���� ����
                            circle_draw.at<float>(a, b)++;
                    }
                }
            }
        }

        int temp = 170; //��ǥ���� �Ӱ���
        for (int i = 0; i < circle_draw.rows; i++)
        {
            for (int j = 0; j < circle_draw.cols; j++)
            {
                if (circle_draw.at<float>(i, j) > temp)
                {
                    x_point.push_back(i); // �������������� ���� �߽��� ��ǥ�� �������� append
                    y_point.push_back(j);
                    rad_len.push_back(rad);
                }
            }
        }
    }

    int Center_num = x_point.size(); // �ߺ��� �߽���ǥ�� �����ʿ�, �� �߽���ǥ����
    for (int i = 0; i < x_point.size(); i++)
    {
        int checker_x = x_point[i];
        int checker_y = y_point[i];
        for (int j = 0; j < Center_num; j++) // �ٸ� �߽���ǥ�� ���Ѵ�.
        {
            if (i != j)
            {
                if ((x_point[j] + 15 >= checker_x && y_point[j] + 15 >= checker_y) && (x_point[j] - 15 <= checker_x && y_point[j] - 15 <= checker_y))
                { // 15�ȼ� �̳��� ������ �� ���� �߽��� �ſ� ����� ������ �Ǵ��Ͽ� ����
                    x_point.erase(x_point.begin() + j);
                    y_point.erase(y_point.begin() + j);
                    rad_len.erase(rad_len.begin() + j);

                    Center_num--; // ���� ����
                    j = 0; // �ʱ�ȭ
                    i = 0;
                    checker_x = x_point[i];
                    checker_y = y_point[i];
                }
            }
        }
    }

    for (size_t i = 0; i < x_point.size(); i++)
        circles.push_back(cv::Vec3f(y_point[i], x_point[i], rad_len[i]));
    // circles���� ���� �߽���ǥ, �������� ������������ ����
}


int main(void)
{
    std::ifstream file; // ���� �б�
    file.open("coins_396x400.raw", std::ios::binary);
    if (!file.is_open())
    {
        printf("Cannot open file\n");
        return -1;
    }

    uchar* buffer = new uchar[WIDTH * HEIGHT];
    file.read(reinterpret_cast<char*>(buffer), WIDTH * HEIGHT);
    file.close();

    cv::Mat grayImage(HEIGHT, WIDTH, CV_8UC1, buffer); // ���� �̹���
    cv::Mat blurredImage;
    Gaussian_Blur(grayImage, blurredImage, 5); // ����þ� ���͸� ����Ͽ� ����� ���ҽ�Ų �̹���

    cv::Mat magnitude, angle;
    cv::imshow("Original Image", grayImage);
    cv::imshow("Blurred Image", blurredImage);
    cv::waitKey(0);

    cv::Mat suppressed2(HEIGHT, WIDTH, CV_8UC1);
    Mynon_Max_Suppression(blurredImage, suppressed2);
    cv::imshow("suppressed2", suppressed2); // �̹����� ���� ���� �����Ѵ�
    cv::waitKey(0);

    cv::Mat edges2; // ���� ������ ����Ǵ� ��ü
    DoubleThreshold_EdgeTracking(suppressed2, edges2, 30, 55); // �Ӱ谪�� �ΰ� ����Ͽ� ����
    cv::imshow("edges2", edges2);
    cv::waitKey(0);

    std::vector<cv::Vec3f> circles;

    std::time_t time_func_call = std::time(nullptr); // MyHoughCircles ���� �ð� ����
    MyHoughCircles(edges2, circles, 20, 100); // (a, b, r)��ǥ ����� ������ȯ
    std::time_t time_func_close = std::time(nullptr); // MyHoughCircles ���� �ð� ����
    std::cout << "Execution time of  MyHoughCircles(): " << difftime(time_func_close, time_func_call) << " sec" << std::endl; // �ҿ� �ð� ���

    cv::Mat detected_circle;
    cvtColor(grayImage, detected_circle, cv::COLOR_GRAY2BGR); // �����̹��� ���� ������ ���� �׸� ���

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

    // gradient method ����� ������ȯ��� (��������)
    /*
    std::vector<int> center_x2, center_y2;
    std::time_t time_func2_call = std::time(nullptr); // MyHoughCircles ���� �ð� ����
    HoughGradientVoting(edges2, center_x2, center_y2);
    std::time_t time_func2_close = std::time(nullptr); // MyHoughCircles ���� �ð� ����
    std::cout << "Execution time of  HoughGradientVoting(): " << difftime(time_func2_close, time_func2_call) << " sec" << std::endl; // �ҿ� �ð� ���

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