#pragma warning(disable:4996)
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

// �̹��� �Է����� ����Ʈ
string img_name[] =
{
	"input//images//1_600x600.png",
	"input//images//2_600x600.png",
	"input//images//3_480x480.png",
	"input//images//4_480x480.png",
	"input//images//5_500x500.png",
	"input//images//6_500x500.png"
};

// Ư¡�� ����Ʈ
string txt_name[] =
{
	"input//labels//1_600x600.txt",
	"input//labels//2_600x600.txt",
	"input//labels//3_480x480.txt",
	"input//labels//4_480x480.txt",
	"input//labels//5_500x500.txt",
	"input//labels//6_500x500.txt"
};

// �ﰢ��
struct Triangle
{
	Point2f p1, p2, p3;
	// ����Ʈ�� �ﰢ�� ���ο� ���ԵǾ� �ִ��� Ȯ��
	bool contains(Point2f pt) const;
	// ����Ʈ�� �ﰢ���� �������� ���ԵǾ� �ִ��� Ȯ��
	bool circum_circle_contains(Point2f pt) const;
	// �ﰢ���� �������� ��
	bool operator==(const Triangle& other) const;
};

bool Triangle::contains(Point2f pt) const
{	// ����Ʈ�� �ﰢ�� ���ο� ���ԵǾ� �ִ��� Ȯ��
	
	float d1, d2, d3; // �Էµ� ���� �ﰢ���� �� ���� �������� ���Ⱚ�� ����� �����Ѵ�
	bool has_neg, has_pos;

	d1 = (pt.x - p2.x) * (p1.y - p2.y) - (p1.x - p2.x) * (pt.y - p2.y);
	d2 = (pt.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (pt.y - p3.y);
	d3 = (pt.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (pt.y - p1.y);

	has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
	has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

	return !(has_neg && has_pos);
	// ���� d1, d2, d3 �߿� ������ ����� ��� �ִٸ�, �� pt�� �ﰢ�� ���ο� ���� �ʴ�
	// �ﰢ�� ���ο� �ִ� ���� �ﰢ���� �� ���� ���� ��� ���� ���⿡ �־�� �Ѵ�
	// ����, !(has_neg && has_pos)�� ���̸� �� pt�� �ﰢ�� ���ο� �ִ�
}

bool Triangle::circum_circle_contains(Point2f pt) const
{
	// ����Ʈ�� �ﰢ���� �������� ���ԵǾ� �ִ��� Ȯ��
	float ab = p1.x * p1.x + p1.y * p1.y;
	float cd = p2.x * p2.x + p2.y * p2.y;
	float ef = p3.x * p3.x + p3.y * p3.y;

	float circum_x = (ab * (p3.y - p2.y) + cd * (p1.y - p3.y) + ef * (p2.y - p1.y)) /
		(p1.x * (p3.y - p2.y) + p2.x * (p1.y - p3.y) + p3.x * (p2.y - p1.y)) / 2.f;
	float circum_y = (ab * (p3.x - p2.x) + cd * (p1.x - p3.x) + ef * (p2.x - p1.x)) /
		(p1.y * (p3.x - p2.x) + p2.y * (p1.x - p3.x) + p3.y * (p2.x - p1.x)) / 2.f;
	float circum_radius = sqrt((p1.x - circum_x) * (p1.x - circum_x) + (p1.y - circum_y) * (p1.y - circum_y));
	// �������� �߽��� ��ǥ, ������ ���
	float dist = sqrt((pt.x - circum_x) * (pt.x - circum_x) + (pt.y - circum_y) * (pt.y - circum_y));
	// �־��� ���� �������� �߽ɰ��� �Ÿ�
	return dist <= circum_radius;
}

bool Triangle::operator==(const Triangle& other) const
{
	// �� �������� ���Ͽ� ���� �ﰢ������ �Ǵ�
	return (p1 == other.p1 && p2 == other.p2 && p3 == other.p3) ||
		(p1 == other.p2 && p2 == other.p3 && p3 == other.p1) ||
		(p1 == other.p3 && p2 == other.p1 && p3 == other.p2);
}

vector<vector<int>> compute_delaunay_triangulation(vector<Point2f>& points, Rect rect) {

	// �ʱ� �ﰢ���� �ſ� ũ�Լ���
	float margin = 2000;
	Point2f p1(rect.x - margin, rect.y - margin);
	Point2f p2(rect.x + rect.width + margin, rect.y - margin);
	Point2f p3(rect.x + rect.width / 2.0f, rect.y + rect.height + margin);
	vector<Triangle> triangles = { { p1, p2, p3 } };

	for (const auto& pt : points)
	{
		vector<Triangle> badTriangles; // �������� �� pt�� ���Ե� �ﰢ�� ����
		vector<pair<Point2f, Point2f>> polygon;  // ���ŵ� �ﰢ���� �� ���� �� ���� �̿��� ���ο� �ﰢ���� ����
		// �������� �����ؼ� 
		// �������� ���Կ��� �˻�.
		for (const auto& tri : triangles)
		{
			if (tri.circum_circle_contains(pt))
			{
				badTriangles.push_back(tri);
				polygon.push_back({ tri.p1, tri.p2 });
				polygon.push_back({ tri.p2, tri.p3 });
				polygon.push_back({ tri.p3, tri.p1 });
			}
		}
		//�ش� �ﰢ�� ���� 
		triangles.erase(remove_if(triangles.begin(), triangles.end(), [&](const Triangle& t)
			{	// �� �ﰢ�� t�� ���ܴ������ Ȯ��
				return find(badTriangles.begin(), badTriangles.end(), t) != badTriangles.end();
			}	// badTriangles ���Ϳ��� t�� ã����, �ش� �ﰢ���� ���� ���
		), triangles.end());
		// remove_if �Լ��� ��ȯ�� ���ο� ������ ������ ���� �������� ��ҵ��� ����
		
		// �ﰢ�� �����.
		vector<pair<Point2f, Point2f>> uniqueEdges;
		for (auto& e1 : polygon)
		{
			bool isEdgeUnique = true;
			for (auto& e2 : polygon)
			{
				if ((e1.first == e2.second) && (e1.second == e2.first))
				{
					isEdgeUnique = false;
					break;
				}
			}
			if (isEdgeUnique)
				uniqueEdges.push_back(e1);
			// �ٸ� ���� �ߺ����� �ʴ� ���� ã�´�.
		}

		for (const auto& edge : uniqueEdges)
			triangles.push_back({ edge.first, edge.second, pt });
		// ���� �߰��� Ư¡���� �ﰢ���� ���� �����Ͽ� ���ο� �ﰢ���� �����
	}

	// ����� �� ����.
	triangles.erase(remove_if(triangles.begin(), triangles.end(),
		[&](const Triangle& t)
		{
			return (t.p1 == p1) || (t.p1 == p2) || (t.p1 == p3) ||
				(t.p2 == p1) || (t.p2 == p2) || (t.p2 == p3) ||
				(t.p3 == p1) || (t.p3 == p2) || (t.p3 == p3);
		}), triangles.end());

	vector<vector<int>> delaunayTri;
	for (const auto& tri : triangles)
	{
		vector<int> indices;
		for (size_t i = 0; i < points.size(); ++i)
		{
			if (points[i] == tri.p1 || points[i] == tri.p2 || points[i] == tri.p3)
				indices.push_back(i);
		}
		if (indices.size() == 3)
			delaunayTri.push_back(indices);
	}

	return delaunayTri; // �ﰢ������ �� �ε����� ��ȯ
}
void calculateAffineTransform(vector<Point2f>& src, vector<Point2f>& dst, Mat& M)
{	// ���� ��ȯ�� ��Ʈ����
	Mat A(6, 6, CV_32F), B(6, 1, CV_32F);
	for (int i = 0; i < 3; i++)
	{
		A.at<float>(i * 2, 0) = src[i].x;
		A.at<float>(i * 2, 1) = src[i].y;
		A.at<float>(i * 2, 2) = 1;
		A.at<float>(i * 2, 3) = 0;
		A.at<float>(i * 2, 4) = 0;
		A.at<float>(i * 2, 5) = 0;

		A.at<float>(i * 2 + 1, 0) = 0;
		A.at<float>(i * 2 + 1, 1) = 0;
		A.at<float>(i * 2 + 1, 2) = 0;
		A.at<float>(i * 2 + 1, 3) = src[i].x;
		A.at<float>(i * 2 + 1, 4) = src[i].y;
		A.at<float>(i * 2 + 1, 5) = 1;

		B.at<float>(i * 2, 0) = dst[i].x;
		B.at<float>(i * 2 + 1, 0) = dst[i].y;
	}

	Mat X = A.inv(DECOMP_SVD) * B;
	M = (Mat_<float>(2, 3) << X.at<float>(0, 0), X.at<float>(1, 0), X.at<float>(2, 0),
		X.at<float>(3, 0), X.at<float>(4, 0), X.at<float>(5, 0));
}

void warpAffineCustom(Mat& src, Mat& dst, Mat& M, Size size)
{	// ��ȯ ��Ʈ������ ����Ͽ� ���� �̹����� ��ȯ�� �̹����� ��ȯ
	dst = Mat::zeros(size, src.type());
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{	// ���� ��ġ pt�� ��ȯ�� �̹����� ���� ���� �ִ��� Ȯ���� ��,
			// ���� �̹����� �ش� �ȼ� ���� ��ȯ�� �̹����� ���� ��ġ�� ����
			Point2f pt = Point2f(M.at<float>(0, 0) * x + M.at<float>(0, 1) * y + M.at<float>(0, 2),
				M.at<float>(1, 0) * x + M.at<float>(1, 1) * y + M.at<float>(1, 2));
			if (pt.x >= 0 && pt.x < size.width && pt.y >= 0 && pt.y < size.height)
				dst.at<Vec3b>(pt) = src.at<Vec3b>(y, x);
		}
	}
}

void applyCustomAffineTransform(Mat& warpImage, Mat& src, vector<Point2f>& srcTri, vector<Point2f>& dstTri)
{
	Mat M;
	calculateAffineTransform(srcTri, dstTri, M);
	warpAffineCustom(src, warpImage, M, warpImage.size());
}

void applyAffineTransform(Mat& warpImage, Mat& src, vector<Point2f>& srcTri, vector<Point2f>& dstTri)
{
	Mat warpMat = getAffineTransform(srcTri, dstTri);
	warpAffine(src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// ���� ����Ʈ�� ����.
void morphTriangle(Mat& img1, Mat& img2, Mat& dst, vector<Point2f>& t1, vector<Point2f>& t2, vector<Point2f>& t, double alpha)
{
	Rect r = boundingRect(t); // �ﰢ���� �����ϴ� ���� ���� �簢���� ���
	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	vector<Point2f> t1Rect, t2Rect, tRect;
	vector<Point> tRectInt;

	for (int i = 0; i < 3; i++) // �ﰢ���� ������ r, r1, r2 �簢���� ��ǥ�� ��ȯ�Ѵ�.
	{
		tRect.push_back(Point2f(t[i].x - r.x, t[i].y - r.y));
		tRectInt.push_back(Point(t[i].x - r.x, t[i].y - r.y));
		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
	}

	Mat img1Rect, img2Rect;
	img1(r1).copyTo(img1Rect);
	img2(r2).copyTo(img2Rect);

	Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type()); 
	Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type()); 

	applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect); // �ι��� ������ ��ȯ ����
	applyAffineTransform(warpImage2, warpImage1, t2Rect, tRect);
	
	imshow("Morphed Image bef1", warpImage1);
	waitKey(0);
	imshow("Morphed Image bef2", warpImage2);
	waitKey(0);
	

	Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2; // alpha ���� ���� �̹����� �����ϰ� ȥ��

	for (int i = 0; i < r.height; i++)
	{
		for (int j = 0; j < r.width; j++)
		{
			if (pointPolygonTest(tRectInt, Point2f(j, i), false) >= 0) // (j, i) ��ǥ�� tRectInt �ﰢ�� ���ο� �ִ��� Ȯ��
			{	// �ﰢ�� ���ο� �ִ� ��� imgRect�� �ȼ� ���� dst�� ���� ��ġ�� ����
				dst.at<Vec3b>(i + r.y, j + r.x) = imgRect.at<Vec3b>(i, j);
			}
		}
	}
}
void readKeypoints(string filename, Point key_points[], int num_points)
{
	ifstream file(filename);
	if (!file.is_open())
	{
		cout << "Failed to open " << filename << endl;
		return;
	}

	string line;
	getline(file, line); // ù��° ������ �����ϰ� �о���δ�.

	for (int i = 0; i < num_points; ++i)
	{
		getline(file, line);
		sscanf(line.c_str(), "%d,%d,%*d", &key_points[i].y, &key_points[i].x);
	}

	file.close();
}

int main()
{
	for (int i = 0; i < 5; i++)
	{
		Mat img1 = imread(img_name[i]);
		Mat img2 = imread(img_name[i + 1]);

		if (img1.empty() || img2.empty())
		{
			cout << "Could not open or find the images!" << endl;
			return -1;
		}

		Point key_point1[41];
		Point key_point2[41];

		readKeypoints(txt_name[i], key_point1, 41);
		readKeypoints(txt_name[i + 1], key_point2, 41);

		vector<Point2f> points1, points2, points;
		for (int i = 0; i < 41; ++i)
		{
			points1.push_back(Point2f(key_point1[i].x, key_point1[i].y));
			points2.push_back(Point2f(key_point2[i].x, key_point2[i].y));
			points.push_back(Point2f((key_point1[i].x + key_point2[i].x) * 0.5, (key_point1[i].y + key_point2[i].y) * 0.5));
		}
		// key_point1�� key_point2 �迭�� �� Ű����Ʈ�� Point2f Ÿ���� ���ͷ� ��ȯ�Ͽ� points1�� points2�� ����
		// points ���ʹ� points1�� points2�� �߰�������, �� �̹��� ������ ������ ����ϴ� �� ���

		Rect rect(0, 0, img1.cols, img1.rows); // �̹��� ��ü������ ��Ÿ���� �簢��
		vector<vector<int>> delaunayTri1 = compute_delaunay_triangulation(points1, rect);
		vector<vector<int>> delaunayTri2 = compute_delaunay_triangulation(points2, rect);
		vector<vector<int>> delaunayTri = compute_delaunay_triangulation(points, rect);

		/*
		Mat img = img1.clone();
		for (const auto& tri : delaunayTri1)
		{
			line(img, points[tri[0]], points[tri[1]], Scalar(0, 255, 0), 1);
			line(img, points[tri[1]], points[tri[2]], Scalar(0, 255, 0), 1);
			line(img, points[tri[2]], points[tri[0]], Scalar(0, 255, 0), 1);
		}
		imshow("Delaunay Triangulation", img);
		waitKey(0);
		*/
		/*
		Mat img2_clone = img2.clone();
		for (const auto& tri : delaunayTri2) {
			line(img2_clone, points2[tri[0]], points2[tri[1]], Scalar(0, 255, 0), 1);
			line(img2_clone, points2[tri[1]], points2[tri[2]], Scalar(0, 255, 0), 1);
			line(img2_clone, points2[tri[2]], points2[tri[0]], Scalar(0, 255, 0), 1);
		}
		imshow("Delaunay Triangulation img2", img2_clone);
		waitKey(0);
		*/
		

		double alpha = 0.5; // Change this value for different morphing stages
		Mat morphedImage = Mat::zeros(img1.size(), img1.type());

		for (size_t i = 0; i < delaunayTri.size(); i++)
		{
			vector<Point2f> t1, t2, t;
			for (size_t j = 0; j < 3; j++)
			{
				t1.push_back(points1[delaunayTri[i][j]]);
				t2.push_back(points2[delaunayTri[i][j]]);
				t.push_back(points[delaunayTri[i][j]]);
			}

			morphTriangle(img1, img2, morphedImage, t1, t2, t, alpha);
		}
		
		imshow("Morphed Image", morphedImage);
		imwrite(to_string(i) + "c.png", morphedImage);
		waitKey(0);

		i++;
	}
	return 0;
}
