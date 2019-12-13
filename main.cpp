#include "anomalies-detecting.h"

//#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <time.h>


using namespace cv;

void drawProximity(const cv::Mat& coeffs, Mat& imageGray, 
    int offsetX, int offsetY, int squeezeMultiplyFactor, float threshold)
{
    cv::Rect rct(offsetX, offsetY, coeffs.cols * squeezeMultiplyFactor, coeffs.rows * squeezeMultiplyFactor);
    cv::Scalar clr{ 255, 0, 255 };
    cv::rectangle(imageGray, rct, clr);

    const float minValue = threshold;
    const float maxValue = 1;
    const float medium = (minValue + maxValue) / 2;

    for (int y = 0; y < coeffs.rows; ++y)
        for (int x = 0; x < coeffs.cols; ++x)
        {
            auto value = coeffs.at<float>({ x, y });
            if (value < minValue)
                continue;

            for (int i = 0; i < squeezeMultiplyFactor; ++i)
                for (int j = 0; j < squeezeMultiplyFactor; ++j)
                {
                    Point pt(x * squeezeMultiplyFactor + i + offsetX, y * squeezeMultiplyFactor + j + offsetY);
                    if (pt.x < 0 || pt.x >= imageGray.cols || pt.y < 0 || pt.y >= imageGray.rows)
                        continue;

                    // get pixel
                    auto color = imageGray.at<Vec3b>(pt);

                    // ... do something to the color ....

                    if (value < medium)
                    {
                        const auto coeff = (value - minValue) / (medium - minValue);
                        color[1] = 255 * coeff;
                        color[0] = 255 * (1.f - coeff);
                        color[2] = 0;
                    }
                    else
                    {
                        const auto coeff = (value - medium) / (maxValue - medium);
                        color[2] =  255 * coeff;
                        color[1] =  255 * (1.f - coeff);
                        color[0] = 0;
                    }

                    // set pixel
                    imageGray.at<Vec3b>(pt) = color;
                }
        }
}


void ConnectedComponentsStats(cv::Mat& img, cv::Mat& output,
    int offsetX, int offsetY, int squeezeMultiplyFactor)
{
    using namespace std;

    // Use connected components with stats
    Mat labels, stats, centroids;
    auto num_objects = connectedComponentsWithStats(img, labels, stats, centroids);
    // Check the number of objects detected
    if (num_objects < 2) {
        cout << "No objects detected" << endl;
        return;
    }
    else {
        cout << "Number of objects detected: " << num_objects - 1 << endl;
    }
    // Create output image coloring the objects and show area
    for (auto i = 1; i < num_objects; i++) {

        cv::Rect rct(
            stats.at<int>(i, CC_STAT_LEFT) * squeezeMultiplyFactor + offsetX,
            stats.at<int>(i, CC_STAT_TOP) * squeezeMultiplyFactor + offsetY,
            stats.at<int>(i, CC_STAT_WIDTH) * squeezeMultiplyFactor,
            stats.at<int>(i, CC_STAT_HEIGHT) * squeezeMultiplyFactor
        );
        cv::Scalar clr{ 255, 0, 255 };
        cv::rectangle(output, rct, clr);
    }
}


int main(int argc, char** argv)
{
    try
    {

        clock_t start = clock();

        // set default values for tracking algorithm and video
        const char* videoPath = (argc >= 2) ? argv[1] : "videos/run.mp4";


        // create a video capture object to read videos
        cv::VideoCapture cap(videoPath);

        Mat frame;

        if (!cap.isOpened())
        {
            frame = cv::imread(videoPath);
            if (frame.empty())
            {
                std::cerr << "Unable to open the file. Exiting!\n";
                return -1;
            }
        }
        else
        {
            // Capture the current frame
            cap >> frame;
        }

        //char ch;
        Mat flowImageGray;
        const char windowName[] = "Anomalies";
        namedWindow(windowName, WINDOW_NORMAL);


        if (frame.empty())
        {
            std::cerr << "No image in the file. Exiting!\n";
            return -1;
        }

        const int SqueezeMultiplyFactor = 4;
        const int FragmentSize = 256;
        //const int offsetX = 300;

        //const int offsetX = 10;
        //const int offsetY = 10;

        //const int offsetX = 400;
        //const int offsetY = 200;

        const struct { int offsetX, offsetY; } positions[]{
            { 10, 10 },
            { 400, 10 },
            { 400, 280 },
            { 800, 300 },
        };

        const float threshold = 0.5f;

        // Iterate until the user presses the Esc key
        while (true)
        {
            // Convert to grayscale
            //cvtColor(frame, flowImageGray, COLOR_BGR2GRAY);
            flowImageGray = frame;

            for (auto& pos : positions) {
                Rect rect(pos.offsetX, pos.offsetY, FragmentSize, FragmentSize);
                const auto proximity = getAnomalies(frame, rect, SqueezeMultiplyFactor);
                // Draw the optical flow map
                drawProximity(proximity, flowImageGray, pos.offsetX, pos.offsetY, SqueezeMultiplyFactor, threshold);

                cv::Mat img_thr(proximity.rows, proximity.cols, CV_8UC1);
                for (int y = 0; y < proximity.rows; ++y)
                    for (int x = 0; x < proximity.cols; ++x)
                    {
                        auto value = proximity.at<float>({ x, y });
                        img_thr.at<uchar>({ x, y }) = (value >= threshold) ? 255 : 0;
                    }

                ConnectedComponentsStats(img_thr, flowImageGray, pos.offsetX, pos.offsetY, SqueezeMultiplyFactor);
            }

            // Display the output image
            imshow(windowName, flowImageGray);

            // Break out of the loop if the user presses the Esc key
            char ch = waitKey(10);
            if (ch == 27)
                break;

            // Capture the current frame
            Mat newFrame;
            cap >> newFrame;

            if (newFrame.empty())
            {
                std::cout << "Handling mapping in " << (double)(clock() - start) / CLOCKS_PER_SEC << " seconds" << std::endl;
                waitKey(0);
                break;
            }

            std::swap(frame, newFrame);
        }

        if (argc > 2)
        {
            imwrite(argv[2], flowImageGray);
        }

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal: " << ex.what() << '\n';
        return 1;
    }
}
