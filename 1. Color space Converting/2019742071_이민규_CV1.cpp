// 2019742071_�̹α�_CV Assignment1

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

void RGB_to_YCbCr(void)
{
	#define WIDTH 352
	#define HEIGHT 288

	std::ifstream file;
	file.open("Suzie_CIF_352x288.raw", std::ios::binary);
	if (!file.is_open())
	{
		printf("������ �� �� �����ϴ�.\n");
		return;
	} // ���Ͽ���

	uchar* buffer = new uchar[WIDTH * HEIGHT * 3]; // �̹��� �����͸� ������ uchar buffer����
	file.read(reinterpret_cast<char*>(buffer), WIDTH * HEIGHT * 3); // ���Ϸκ��� �����͸� �о���δ�
	file.close();

	cv::Mat channel_B(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat channel_G(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat channel_R(HEIGHT, WIDTH, CV_8UC1);

	cv::Mat channel_Y(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat channel_Cr(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat channel_Cb(HEIGHT, WIDTH, CV_8UC1);

	for (int i = 0; i < HEIGHT * WIDTH; i++)
	{
		//�ѹ��� �����͸� �о� �����ϴ°� �ƴ϶� ä�κ��� ���ε��� �о� ���ε��� �����ϴ� ���� �ٽ�
		channel_B.data[i] = buffer[i + HEIGHT * WIDTH * 2];
		channel_G.data[i] = buffer[i + HEIGHT * WIDTH];
		channel_R.data[i] = buffer[i];

		// �־��� ���Ŀ� ���� RGB���� YCbCr���·� �ٲپ ����
		channel_Y.data[i] = (uchar)(0.299 * channel_R.data[i] + 0.587 * channel_G.data[i] + 0.114 * channel_B.data[i]);
		channel_Cb.data[i] = (uchar)(128 - 0.169 * channel_R.data[i] - 0.331 * channel_G.data[i] + 0.500 * channel_B.data[i]);
		channel_Cr.data[i] = (uchar)(128 + 0.500 * channel_R.data[i] - 0.419 * channel_G.data[i] - 0.0813 * channel_B.data[i]);
	}

	cv::Mat imageRGB(HEIGHT, WIDTH, CV_8UC3); // R, G, Bä�� ������ ��ĥ ���� ����

	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			uchar B = channel_B.at<uchar>(y, x);
			uchar G = channel_G.at<uchar>(y, x);
			uchar R = channel_R.at<uchar>(y, x);

			imageRGB.at<cv::Vec3b>(y, x) = cv::Vec3b(B, G, R);
		} //�� �ȼ� ��ġ���� ���� RGB������ Vec3b��ü�� ����� �̸� imageRGB�� ���� ��ġ�� ���ļ� ����
	}

	// RGB�� ��ģ �̹����� R, G, B �� ä���� �̹����� ���
	cv::imshow("RGB", imageRGB);
	cv::waitKey(0);
	cv::imshow("R", channel_R);
	cv::waitKey(0);
	cv::imshow("G", channel_G);
	cv::waitKey(0);
	cv::imshow("B", channel_B);
	cv::waitKey(0);

	cv::Mat channel_YCbCr_R(HEIGHT, WIDTH, CV_8UC1); // YCbCr�����͸� �ٽ� RGB�� ������ �� ����� ���� ����
	cv::Mat channel_YCbCr_G(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat channel_YCbCr_B(HEIGHT, WIDTH, CV_8UC1);

	for (int i = 0; i < HEIGHT * WIDTH; i++)
	{
		channel_YCbCr_R.data[i] = (uchar)(1.000 * channel_Y.data[i] + 1.402 * (channel_Cr.data[i] - 128));
		channel_YCbCr_G.data[i] = (uchar)(1.000 * channel_Y.data[i] - 0.714 * (channel_Cr.data[i] - 128) - 0.344 * (channel_Cb.data[i] - 128));
		channel_YCbCr_B.data[i] = (uchar)(1.000 * channel_Y.data[i] + 1.772 * (channel_Cb.data[i] - 128));
	} // �־��� ���Ŀ� ���� YCbCr���� RGB�� ����

	cv::Mat image_YCbCr(HEIGHT, WIDTH, CV_8UC3);
	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			uchar YCbCr_B = channel_YCbCr_B.at<uchar>(y, x);
			uchar YCbCr_G = channel_YCbCr_G.at<uchar>(y, x);
			uchar YCbCr_R = channel_YCbCr_R.at<uchar>(y, x);

			image_YCbCr.at<cv::Vec3b>(y, x) = cv::Vec3b(YCbCr_B, YCbCr_G, YCbCr_R);
		}
	} //�� �ȼ� ��ġ���� ���� RGB������ Vec3b��ü�� ����� �̸� image_YCbCr�� ���� ��ġ�� ���ļ� ����

	// RGB�� ��ģ �̹����� Y, Cb, Cr �� ä���� �̹����� ���
	cv::imshow("YCbCr 4:4:4", image_YCbCr);
	cv::waitKey(0);
	cv::imshow("Y 4:4:4", channel_Y);
	cv::waitKey(0);
	cv::imshow("Cb 4:4:4", channel_Cb);
	cv::waitKey(0);
	cv::imshow("Cr 4:4:4", channel_Cr);
	cv::waitKey(0);
	
	// CbCr�� �����͸� ������ ������ ���� (ũ��� ������ 1/4)
	cv::Mat channelCb_420(channel_Cb.rows / 2, channel_Cb.cols / 2, CV_8UC1);
	cv::Mat channelCr_420(channel_Cr.rows / 2, channel_Cr.cols / 2, CV_8UC1);

	// Cb ����
	for (int y = 0; y < channelCb_420.rows; ++y)
	{
		for (int x = 0; x < channelCb_420.cols; ++x)
		{
			int sum = 0;
			sum += channel_Cb.at<uchar>(y * 2, x * 2);
			sum += channel_Cb.at<uchar>(y * 2, x * 2 + 1);
			sum += channel_Cb.at<uchar>(y * 2 + 1, x * 2);
			sum += channel_Cb.at<uchar>(y * 2 + 1, x * 2 + 1);
			channelCb_420.at<uchar>(y, x) = sum / 4;
		}
	} // ������ 2X2 �ȼ��� ��հ��� ��ǥ������ �Ͽ� channelCb_420�� �ϳ��� �ȼ��� ����

	// Cr ����
	for (int y = 0; y < channelCr_420.rows; ++y)
	{
		for (int x = 0; x < channelCr_420.cols; ++x)
		{
			int sum = 0;
			sum += channel_Cr.at<uchar>(y * 2, x * 2);
			sum += channel_Cr.at<uchar>(y * 2, x * 2 + 1);
			sum += channel_Cr.at<uchar>(y * 2 + 1, x * 2);
			sum += channel_Cr.at<uchar>(y * 2 + 1, x * 2 + 1);
			channelCr_420.at<uchar>(y, x) = sum / 4;
		}
	} // ������ 2X2 �ȼ��� ��հ��� ��ǥ������ �Ͽ� channelCr_420�� �ϳ��� �ȼ��� ����
	
	// ������ �����͸� �ٽ� ��������� ������ ����. RGB�� ��ȯ�� �� ���
	cv::Mat channelCb_sizeup(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat channelCr_sizeup(HEIGHT, WIDTH, CV_8UC1);

	// Cb �������
	for (int y = 0; y < HEIGHT; ++y)
		for (int x = 0; x < WIDTH; ++x)
			channelCb_sizeup.at<uchar>(y, x) = channelCb_420.at<uchar>(int(y / 2), int(x / 2));
			// �ϳ��� ����� ���� 2X2 �ȼ��� �����Ѵ�. 

	// Cr �������
	for (int y = 0; y < HEIGHT; ++y) 
		for (int x = 0; x < WIDTH; ++x) 
			channelCr_sizeup.at<uchar>(y, x) = channelCr_420.at<uchar>(int(y / 2), int(x / 2));
			// �ϳ��� ����� ���� 2X2 �ȼ��� �����Ѵ�.

	// ��������� ���� RGB������ ��ȯ�Ͽ� ������ ������ ����
	cv::Mat channel_YCbCr_420_R(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat channel_YCbCr_420_G(HEIGHT, WIDTH, CV_8UC1);
	cv::Mat channel_YCbCr_420_B(HEIGHT, WIDTH, CV_8UC1);

	for (int i = 0; i < HEIGHT * WIDTH; i++)
	{
		channel_YCbCr_420_R.data[i] = (uchar)(1.000 * channel_Y.data[i] + 1.402 * (channelCr_sizeup.data[i] - 128));
		channel_YCbCr_420_G.data[i] = (uchar)(1.000 * channel_Y.data[i] - 0.714 * (channelCr_sizeup.data[i] - 128) - 0.344 * (channelCb_sizeup.data[i] - 128));
		channel_YCbCr_420_B.data[i] = (uchar)(1.000 * channel_Y.data[i] + 1.772 * (channelCb_sizeup.data[i] - 128));
	} // �־��� ���Ŀ� ���� YCbCr���� RGB�� ����

	cv::Mat image_YCbCr_420(HEIGHT, WIDTH, CV_8UC3);

	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			uchar YCbCr_420_B = channel_YCbCr_420_B.at<uchar>(y, x);
			uchar YCbCr_420_G = channel_YCbCr_420_G.at<uchar>(y, x);
			uchar YCbCr_420_R = channel_YCbCr_420_R.at<uchar>(y, x);

			image_YCbCr_420.at<cv::Vec3b>(y, x) = cv::Vec3b(YCbCr_420_B, YCbCr_420_G, YCbCr_420_R);
		}
	}//�� �ȼ� ��ġ���� ���� RGB������ Vec3b��ü�� ����� �̸� image_YCbCr_420�� ���� ��ġ�� ���ļ� ����

	// RGB�� ��ģ �̹����� Y, Cb, Cr �� ä���� �̹����� ���
	cv::imshow("YCbCr 4:2:0", image_YCbCr_420);
	cv::waitKey(0);
	cv::imshow("Y 4:2:0", channel_Y);
	cv::waitKey(0);;
	cv::imshow("Cb 4:2:0", channelCb_420);
	cv::waitKey(0);
	cv::imshow("Cr 4:2:0", channelCr_420);
	cv::waitKey(0);

	// ������ ������ ����
	std::ofstream yuv444;
	yuv444.open("YCbCr_444.yuv", std::ios::binary);

	std::ofstream yuv420;
	yuv420.open("YCbCr_420.yuv", std::ios::binary);

	if (!yuv444.is_open())
	{
		printf("���� ���� ����.\n");
		return;
	}

	if (!yuv420.is_open())
	{
		printf("���� ���� ����.\n");
		return;
	}

	// reinterpret_cast�� ������� ������ ���ڿ��� ����ȴ�.
	yuv444.write(reinterpret_cast<char*>(channel_Y.data), WIDTH * HEIGHT);
	yuv444.write(reinterpret_cast<char*>(channel_Cb.data), WIDTH * HEIGHT);
	yuv444.write(reinterpret_cast<char*>(channel_Cr.data), WIDTH * HEIGHT);
	yuv444.close();

	yuv420.write(reinterpret_cast<char*>(channel_Y.data), WIDTH * HEIGHT);
	yuv420.write(reinterpret_cast<char*>(channelCb_420.data), WIDTH * HEIGHT / 4);
	yuv420.write(reinterpret_cast<char*>(channelCr_420.data), WIDTH * HEIGHT / 4);
	yuv420.close();

	return;
}

void YCbCr_display(void)
{
	#define WIDTH 416
	#define HEIGHT 240
	#define FRAME_RATE 30

	std::ifstream file;
	file.open("RaceHorses_416x240_30.yuv", std::ios::binary);
	if (!file.is_open())
	{
		printf("������ �� �� �����ϴ�.\n");
		return;
	}

	uchar* buffer = new uchar[WIDTH * HEIGHT * 3 / 2]; // 4:2:0�� 4:4:4�� ������ �뷮�� �ʿ�

	std::ofstream raw;
	raw.open("output.raw", std::ios::binary);

	while (!file.eof()) // while�� 1ȸ�� ������ �� �������� �д� ���ÿ� �� �������� �����Ѵ�.
	{
		file.read(reinterpret_cast<char*>(buffer), WIDTH * HEIGHT * 3 / 2); // ������ �� �����Ӿ� �д´�
		cv::Mat Y(HEIGHT, WIDTH, CV_8UC1);
		cv::Mat U(HEIGHT / 2, WIDTH / 2, CV_8UC1);
		cv::Mat V(HEIGHT / 2, WIDTH / 2, CV_8UC1);

		// �� ���������ӿ� �����Ͱ� Y,U,V ������ ����Ǿ������Ƿ� ���������� �д´�
		for (int y = 0; y < HEIGHT; ++y) 
			for (int x = 0; x < WIDTH; ++x) 
				Y.at<uchar>(y, x) = buffer[y * WIDTH + x];

		for (int y = 0; y < HEIGHT / 2; ++y) 
			for (int x = 0; x < WIDTH / 2; ++x) 
				U.at<uchar>(y, x) = buffer[WIDTH * HEIGHT + y * (WIDTH / 2) + x];
			
		for (int y = 0; y < HEIGHT / 2; ++y) 
			for (int x = 0; x < WIDTH / 2; ++x) 
				V.at<uchar>(y, x) = buffer[WIDTH * HEIGHT + (WIDTH * HEIGHT / 4) + y * (WIDTH / 2) + x];
		
		// U�� V�� �����͸� ���� ũ��� �ǵ��� ������ ����
		cv::Mat U_sizeup(HEIGHT, WIDTH, CV_8UC1);
		cv::Mat V_sizeup(HEIGHT, WIDTH, CV_8UC1);

		for (int y = 0; y < HEIGHT; ++y) 
			for (int x = 0; x < WIDTH; ++x) 
				U_sizeup.at<uchar>(y, x) = U.at<uchar>(int(y / 2), int(x / 2));

		for (int y = 0; y < HEIGHT; ++y) 
			for (int x = 0; x < WIDTH; ++x) 
				V_sizeup.at<uchar>(y, x) = V.at<uchar>(int(y / 2), int(x / 2));
		// �ϳ��� �ٿ���õ� ���� 2X2 �ȼ��� �����Ѵ�. 

		cv::Mat frame_R(HEIGHT, WIDTH, CV_8UC1);
		cv::Mat frame_G(HEIGHT, WIDTH, CV_8UC1);
		cv::Mat frame_B(HEIGHT, WIDTH, CV_8UC1);

		for (int i = 0; i < HEIGHT * WIDTH; i++)
		{
			frame_R.data[i] = (uchar)(1.000 * Y.data[i] + 1.402 * (V_sizeup.data[i] - 128));
			frame_G.data[i] = (uchar)(1.000 * Y.data[i] - 0.714 * (V_sizeup.data[i] - 128) - 0.344 * (U_sizeup.data[i] - 128));
			frame_B.data[i] = (uchar)(1.000 * Y.data[i] + 1.772 * (U_sizeup.data[i] - 128));
		}// �־��� ���Ŀ� ���� YCbCr���� RGB�� ����

		cv::Mat one_frame(HEIGHT, WIDTH, CV_8UC3);

		for (int y = 0; y < HEIGHT; ++y)
		{
			for (int x = 0; x < WIDTH; ++x)
			{
				uchar YUV_B = frame_B.at<uchar>(y, x);
				uchar YUV_G = frame_G.at<uchar>(y, x);
				uchar YUV_R = frame_R.at<uchar>(y, x);

				one_frame.at<cv::Vec3b>(y, x) = cv::Vec3b(YUV_B, YUV_G, YUV_R);
			}
		}//�� �ȼ� ��ġ���� ���� RGB������ Vec3b��ü�� ����� �̸� one_frame�� ���� ��ġ�� ���ļ� ����

		cv::imshow("YCbCr 4:2:0 video", one_frame); // ���������� ���
		cv::waitKey(1000 / FRAME_RATE); // �̹����� �̹������� ��� ������ 1�� / Frame_rate�� ����

		for (int i = 0; i < one_frame.rows; ++i) 
			raw.write(reinterpret_cast<char*>(one_frame.ptr(i)), one_frame.cols * one_frame.elemSize());
			//�������� �� ���� ù��° ������ �� ���� ��ü ����Ʈ ����ŭ ���Ͽ� �����͸� write
	}
	raw.close();
	delete[] buffer;
	return;
}

int main(void)
{
	RGB_to_YCbCr();
	YCbCr_display();
	return 0;
}