// anomalies-detecting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include "nanoflann.hpp"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

#include <future>

#include <numeric>

#include <utility>


#include <float.h>

#include <time.h>

using namespace cv;
using namespace std;

using namespace nanoflann;

//////////////////////////////////////////////////////////////////////////////

// https://github.com/jmtilli/fastdiv

struct fastdivctx {
    uint32_t mult;
    uint32_t mod;
    uint32_t shift1, shift2;
};

constexpr inline
uint8_t ilog(uint32_t i)
{
    uint8_t result = 0;
    while (i >>= 1)
    {
        result++;
    }
    return result;
}

//static 
inline void init_fastdivctx(fastdivctx *ctx, uint32_t divisor)
{
    uint8_t ilogd = ilog(divisor);
    int power_of_2 = (divisor & (divisor - 1)) == 0;
    if (divisor == 0 || divisor >= (1U << 31))
    {
        abort(); // Not supported
    }
    if (power_of_2)
    {
        ctx->shift1 = 0;
    }
    else
    {
        ctx->shift1 = 1;
    }
    ctx->shift2 = ilogd;
    ctx->mod = divisor;
    ctx->mult = (1ULL << (32 + ctx->shift1 + ctx->shift2)) / divisor + 1;
}

//static 
inline uint32_t fastmod(const fastdivctx *ctx, uint32_t eax)
{
    uint64_t edxeax = ((uint64_t)eax) * ctx->mult;
    uint32_t edx = edxeax >> 32;
    uint32_t eaxorig = eax;
    eax -= edx;
    eax >>= (ctx->shift1);
    eax += edx;
    eax >>= (ctx->shift2);
    edx = ctx->mod*eax;
    return eaxorig - edx;
}

//static 
inline uint32_t fastdiv(const fastdivctx *ctx, uint32_t eax)
{
    uint64_t edxeax = ((uint64_t)eax) * ctx->mult;
    uint32_t edx = edxeax >> 32;
    eax -= edx;
    eax >>= (ctx->shift1);
    eax += edx;
    eax >>= (ctx->shift2);
    return eax;
}

//static 
inline void fastdivmod(const fastdivctx *ctx, uint32_t eax,
    uint32_t *div, uint32_t *mod)
{
    uint64_t edxeax = ((uint64_t)eax) * ctx->mult;
    uint32_t edx = edxeax >> 32;
    uint32_t eaxorig = eax;
    eax -= edx;
    eax >>= (ctx->shift1);
    eax += edx;
    eax >>= (ctx->shift2);
    *div = eax;
    edx = ctx->mod*eax;
    *mod = eaxorig - edx;
}

template<int divisor>
inline void fastdivmod(uint32_t eax,
    uint32_t *div, uint32_t *mod)
{
    enum { shift1 = (divisor & (divisor - 1)) != 0 };
    enum { shift2 = ilog(divisor) };
    constexpr uint32_t mult = (1ULL << (32 + shift1 + shift2)) / divisor + 1;

    uint64_t edxeax = ((uint64_t)eax) * mult;
    uint32_t edx = edxeax >> 32;
    uint32_t eaxorig = eax;
    eax -= edx;
    eax >>= (shift1);
    eax += edx;
    eax >>= (shift2);
    *div = eax;
    edx = divisor * eax;
    *mod = eaxorig - edx;
}


//////////////////////////////////////////////////////////////////////////////

void OffsetImage(cv::Mat &image, int xoffset, int yoffset, const cv::Scalar& bordercolour = {0, 0, 0})
{
    if (xoffset != 0 && yoffset != 0)
    {
        cv::Mat H = (cv::Mat_<double>(3, 3) <<
            1, 0, xoffset, 0, 1, yoffset, 0, 0, 1);

        cv::Mat aux;
        cv::warpPerspective(image, aux, H, image.size(), cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, bordercolour);
        image = aux;
    }
}

//////////////////////////////////////////////////////////////////////////////

enum { DIMENSION = 3 };

enum { ADDITIONAL = 0 };

enum { NUM_ATTRIBUTES = DIMENSION * DIMENSION + ADDITIONAL };

typedef float AttributeType;

class PointsProvider
{
public:
    PointsProvider(const cv::Mat* mat)
        : m_mat(mat)
        , m_numRows(mat->rows - DIMENSION + 1)
        , m_numCols(mat->cols - DIMENSION + 1)
        , m_coeffs(m_numRows * m_numCols, 1.f)
    {
        init_fastdivctx(&m_fastdivctx, m_numCols);
    }

    size_t kdtree_get_point_count() const
    {
        return m_numRows * m_numCols;
    }

    void get_x_y(const size_t idx, uint32_t& x, uint32_t& y) const
    {
        fastdivmod(&m_fastdivctx, idx, &y, &x);
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        uint32_t x, y;
        get_x_y(idx, x, y);

        const double coeff = get_coeff(idx);
        
        const double v = do_kdtree_get_pt(x, y, dim);

        return v / coeff;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

private:
    float get_coeff(const size_t idx) const
    {
        float result = m_coeffs[idx];
        if (result == 1.f)
        {
            double sq_sum = 0;
            for (int dim = 0; dim < NUM_ATTRIBUTES - ADDITIONAL; ++dim)
            {
                const auto v = do_kdtree_get_pt(idx, dim + ADDITIONAL);
                sq_sum += v * v;
            }

            if (sq_sum > 0)
            {
                result = sqrt(sq_sum);
                const_cast<float&>(m_coeffs[idx]) = result;
            }
        }
        return result;
    }
    float do_kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        uint32_t x, y;
        get_x_y(idx, x, y);
        return do_kdtree_get_pt(x, y, dim);
    }
    float do_kdtree_get_pt(uint32_t x, uint32_t y, size_t dim) const
    {
        uint32_t div, mod;
        fastdivmod<DIMENSION>(dim - ADDITIONAL, &div, &mod);

        const double v = m_mat->at<uchar>(y + div, x + mod);
        return v;
    }

private:
    const cv::Mat* m_mat;
    int m_numRows;
    int m_numCols;
    std::vector<float> m_coeffs;
    fastdivctx m_fastdivctx;
};


    // construct a kd-tree index:
typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<float, PointsProvider >,
    PointsProvider,
    NUM_ATTRIBUTES /* dim */
> my_kd_tree_t;


//////////////////////////////////////////////////////////////////////////////

typedef std::vector<std::pair<Point, float>> MapType;



void drawProximity(const std::vector<MapType>& mappings, Mat& imageGray)
{
    std::vector<std::vector<float>> coeffs(imageGray.rows);
    for (auto& l : coeffs)
    {
        l.resize(imageGray.cols, 1.f);
    }


    for (auto& points : mappings)
    {
        if (points.empty())
            continue;

        std::vector<float> values;
        values.reserve(points.size());

        for (const auto& pair : points)
        {
            const auto value = pair.second;
            values.push_back(value);
        }

        std::sort(values.begin(), values.end());

        for (const auto& pair : points)
        {
            const auto range = std::equal_range(values.begin(), values.end(), pair.second);
            auto value = ((range.first - values.begin()) + (range.second - values.begin())) / (2. * values.size());
            coeffs[pair.first.y][pair.first.x] *= value;
        }
    }

    const float minValue = .4f; // threshold
    const float maxValue = 1;
    const float medium = (minValue + maxValue) / 2;

    for (int y = 0; y < coeffs.size(); ++y)
        for (int x = 0; x < coeffs[0].size(); ++x)
        { 
            auto value = coeffs[y][x];
            if (value < minValue)
                continue;

            Point pt(x, y);
            // get pixel
            auto color = imageGray.at<Vec3b>(pt);

            // ... do something to the color ....

            if (value < medium)
            {
                const auto coeff = (value - minValue) / (medium - minValue);
                color[1] *= coeff;
                color[0] *= (1.f - coeff);
                color[2] = 0;
            }
            else
            {
                const auto coeff = (value - medium) / (maxValue - medium);
                color[2] *= coeff;
                color[1] *= (1.f - coeff);
                color[0] = 0;
            }

            // set pixel
            imageGray.at<Vec3b>(pt) = color;
        }

}


MapType GenerateMap(const Mat& curGray)
{
    PointsProvider provider(&curGray); //&prevGray);

    my_kd_tree_t infos(NUM_ATTRIBUTES, provider);

    //infos.buildIndex();
    infos.fastBuildIndex();

    const auto numRows = curGray.rows - DIMENSION + 1;
    const auto numCols = curGray.cols - DIMENSION + 1;


    auto lam = [&infos, &curGray, &provider](int yBegin, int yEnd) {

        const auto numCols = curGray.cols - DIMENSION + 1;

        MapType shifts;

        // searching
        for (int y = yBegin; y < yEnd; ++y)
        {
            for (int x = 0; x < numCols; ++x)
            {
                AttributeType pos[NUM_ATTRIBUTES];
                unsigned int sq_sum = 0;
                for (int i = 0; i < DIMENSION; ++i)
                    for (int j = 0; j < DIMENSION; ++j)
                    {
                        const auto v = curGray.at<uchar>(y + i, x + j);
                        pos[i * DIMENSION + j + ADDITIONAL] = v;
                        sq_sum += v * v;
                    }

                if (sq_sum > 0)
                {
                    const auto coeff = sqrt(sq_sum);
                    for (auto& v : pos)
                        v /= coeff;
                }

                enum { NUM_RESULTS = 5 };

                size_t ret_index[NUM_RESULTS];
                float out_dist_sqr[NUM_RESULTS];

                const auto num_results = infos.knnSearch(&pos[0], NUM_RESULTS, &ret_index[0], &out_dist_sqr[0]);

                auto results = std::inner_product(
                    ret_index, ret_index + num_results, out_dist_sqr,
                    std::vector<std::pair<size_t, float>>(),
                    [](auto a, auto b) { a.push_back(std::move(b)); return a; },
                    std::make_pair<size_t&, float&>
                );

                results.erase(
                    std::remove_if(results.begin(), results.end(), 
                        [&provider, x, y](const auto& result) { 
                            uint32_t xx, yy;
                            provider.get_x_y(result.first, xx, yy);
                            return hypot(x - int(xx), y - int(yy)) < 20;
                        }),
                    results.end());

                // In case of less points in the tree than requested:

                enum { NN_RESULT = 2 };

                if (results.size() > NN_RESULT)
                {
                    Point ptTo(x + DIMENSION / 2, y + DIMENSION / 2);
                    shifts.push_back({ ptTo, results[NN_RESULT].second });
                }
            }
        }

        return shifts;
    };

    enum { NUM_THREADS = 16 };

    std::vector<std::future<MapType>> proxies;

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        proxies.push_back(std::async(std::launch::async, lam,
            (numRows * i) / NUM_THREADS,
            (numRows * (i + 1)) / NUM_THREADS));
    }

    MapType shifts;

    for (auto& p : proxies)
    {
        auto v = p.get();
        shifts.insert(shifts.end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
    }

    return shifts;
}


int main(int argc, char** argv)
{
    try
    {

        clock_t start = clock();

        // set default values for tracking algorithm and video
    string videoPath = (argc == 2) ? argv[1] : "videos/run.mp4";


    // create a video capture object to read videos
    cv::VideoCapture cap(videoPath);

    Mat frame;

    if (!cap.isOpened())
    {
        frame = cv::imread(videoPath);
        if (frame.empty())
        {
            cerr << "Unable to open the file. Exiting!" << endl;
            return -1;
        }
    }
    else
    {
        // Capture the current frame
        cap >> frame;
    }

    //char ch;
    Mat curGray, /*prevGray,*/ flowImageGray;
    string windowName = "Anomalies";
    namedWindow(windowName, WINDOW_NORMAL);


    if (frame.empty())
    {
        cerr << "No image in the file. Exiting!" << endl;
        return -1;
    }

    const float scalingFactor = 1. / 4; //0.125;

    // Iterate until the user presses the Esc key
    while (true)
    {
        std::vector<MapType> mappings;

        for (int i = 0; i < 16; ++i)
        {
            Mat frameCopy = frame;
            OffsetImage(frameCopy, i / 4, i % 4);

            // Resize the frame
            resize(frameCopy, frameCopy, Size(), scalingFactor, scalingFactor, INTER_AREA);

            // Convert to grayscale
            cvtColor(frameCopy, curGray, COLOR_BGR2GRAY);

            mappings.push_back(GenerateMap(curGray));

            if (i == 0)
            {
                // Convert to 3-channel RGB
                cvtColor(curGray, flowImageGray, COLOR_GRAY2BGR);
            }
        }

        // Draw the optical flow map
        drawProximity(mappings, flowImageGray);

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
