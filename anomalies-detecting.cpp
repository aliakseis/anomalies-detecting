// anomalies-detecting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "anomalies-detecting.h"

#include "nanoflann.hpp"


#include "opencv2/imgproc/imgproc.hpp"



#include <future>

#include <numeric>

#include <utility>


#include <float.h>


using namespace cv;
using namespace std;

using namespace nanoflann;

namespace {

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

cv::Mat OffsetImage(const cv::Mat &image, int xoffset, int yoffset, int coeff, const cv::Size& size)
{
        cv::Mat H = (cv::Mat_<double>(2, 3) <<
            coeff, 0, xoffset, 0, coeff, yoffset);

        cv::Mat aux;
        cv::warpAffine(image, aux, H, size, cv::INTER_AREA | cv::WARP_INVERSE_MAP,
            cv::BORDER_REPLICATE);
        return aux;
}

//////////////////////////////////////////////////////////////////////////////

enum { DIMENSION = 3 };

enum { ADDITIONAL = 1 };

enum { NUM_ATTRIBUTES = DIMENSION * DIMENSION + ADDITIONAL };

typedef float AttributeType;

float getBrightnessValue(float param)
{
    return  param / 200.;
}

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
        const double coeff = get_coeff(idx);

        if (dim == 0)
            return getBrightnessValue(coeff);

        uint32_t x, y;
        get_x_y(idx, x, y);

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


cv::Mat mergeProximity(const std::vector<MapType>& mappings, int rows, int cols)
{
    cv::Mat coeffs(rows, cols, CV_32FC1);

    coeffs = -1.f; // untouched points don't signal
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
            const Point& pt = pair.first;
            auto coeff = coeffs.at<float>(pt);
            if (coeff == -1.f)
                coeff = 1.f;

            const auto range = std::equal_range(values.begin(), values.end(), pair.second);
            auto value = ((range.first - values.begin()) + (range.second - values.begin())) / (2. * values.size());

            coeff *= value;

            coeffs.at<float>(pt) = coeff;
        }
    }

    return coeffs;
}


MapType GenerateMap(const Mat& curGray)
{
    PointsProvider provider(&curGray);

    my_kd_tree_t infos(NUM_ATTRIBUTES, provider);

    //infos.buildIndex();
    infos.fastBuildIndex();

    const auto numRows = curGray.rows - DIMENSION + 1;
    const auto numCols = curGray.cols - DIMENSION + 1;


    auto lam = [&infos, &curGray, &provider](int yBegin, int yEnd) {

        const auto numCols = curGray.cols - DIMENSION + 1;

        MapType shifts;

        enum { NTH_RESULT_INDEX = 2 };

        std::vector<size_t> ret_index;
        std::vector<float> out_dist_sqr;

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

                    pos[0] = getBrightnessValue(coeff);
                }
                else
                    pos[0] = getBrightnessValue(0);

                for (unsigned int bufSize = NTH_RESULT_INDEX + 4;; bufSize *= 2)
                {
                    // resize to fit if needed
                    if (ret_index.size() < bufSize)
                        ret_index.resize(bufSize);
                    if (out_dist_sqr.size() < bufSize)
                        out_dist_sqr.resize(bufSize);

                    const auto num_results = infos.knnSearch(&pos[0], bufSize, &ret_index[0], &out_dist_sqr[0]);

                    auto results = std::inner_product(
                        ret_index.cbegin(), ret_index.cbegin() + num_results, out_dist_sqr.cbegin(),
                        std::vector<std::pair<size_t, float>>(),
                        [](auto a, auto b) { a.push_back(std::move(b)); return a; },
                        std::make_pair<const size_t&, const float&>
                    );

                    results.erase(
                        std::remove_if(results.begin(), results.end(),
                            [&provider, x, y](const auto& result) {
                                uint32_t xx, yy;
                                provider.get_x_y(result.first, xx, yy);
                                return hypot(x - int(xx), y - int(yy)) < 20; // TODO calculate proximity param
                            }),
                        results.end());

                    if (results.size() > NTH_RESULT_INDEX)
                    {
                        Point ptTo(x + DIMENSION / 2, y + DIMENSION / 2);
                        shifts.push_back({ ptTo, results[NTH_RESULT_INDEX].second });

                        break; // ok
                    }

                    // In case of less points in the tree than requested:
                    if (num_results < bufSize)
                        break;
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

} // namespace



cv::Mat getAnomalies(const cv::Mat& frame, const cv::Rect& rect,
    int squeezeMultiplyFactor // must be a multiple of 4
)
{
    enum { MULTIPLY = 4 };

    const auto outSize = rect.size() / squeezeMultiplyFactor;

    std::vector<MapType> mappings;

    const int step = squeezeMultiplyFactor / MULTIPLY;

    for (int i = 0; i < MULTIPLY; ++i)
        for (int j = 0; j < MULTIPLY; ++j)
        {
            Mat frameCopy = OffsetImage(frame,
                rect.x + i * step,
                rect.y + j * step,
                squeezeMultiplyFactor, outSize);

            // Convert to grayscale
            cvtColor(frameCopy, frameCopy, COLOR_BGR2GRAY);

            mappings.push_back(GenerateMap(frameCopy));
        }

    return mergeProximity(mappings, outSize.height, outSize.width);
}

