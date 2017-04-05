/*!
 * @author Hisashi Ikari
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <thread>

#include "Matrix.h"
#include "Operator.h"
#include "Screen.h"

#if defined(BUILD_CPU) || defined(BUILD_CUDA) || defined(BUILD_X5) || defined(BUILD_CL)
#include <opencv2/opencv.hpp>
#include <omp.h>
#endif

#ifdef BUILD_CL
#include <opencv2/core/ocl.hpp>
#endif

#ifdef BUILD_CUDA
template<typename T> using vector = std::vector<T>;
#include <opencv2/core/cuda.hpp>
#endif

#define TCV_E template<class C, typename T>
#define TCV_H template<class C, typename T, typename R, int D>
#define TCV_V Matrix<C, T>
#define TCV_M MatrixPtr<C, T>
#define TCV_P PixelPtr<T>
#define TCV_U PixelPtr<R>
#define TCV_S Shape

using namespace tinycv;

typedef std::chrono::system_clock::time_point elaptime;
const elaptime now(const std::string& function, const bool forced);
const void print_time(const elaptime& start, const char* function, const bool forced);

static bool DEBUG_SAVE_IMAGE = false;
static bool DEBUG_PRINT_MESSAGE = true;

TCV_H static void print_pixels(const TCV_M matrix, const char* title, bool titleonly=true, bool loglimit=true, uint maxy=1)
{
    const TCV_S shape = matrix->shape();
	std::cout << title << "(y: " << shape[0] << ", x: " << shape[1] << ")" << std::endl;
	if (titleonly) return;
	for (uint y = 0; y < (loglimit ? maxy : shape[0]); y++) {
		for (uint x = 0; x < shape[1]; x++) {
			const TCV_U pixel = matrix->template at<R, D>(y, x);
			std::cout << "y:" << y << ", x:" << x << ", r:" << pixel->at(0) << ", g:" << pixel->at(1) << ", b:" << pixel->at(2) << std::endl;
		}
	}
	std::cout << std::endl;
};

inline const void print_line(const char* text)
{
    for (uint i = 0; i < 70; i++) {
	    std::cout << text;
    }
	std::cout << std::endl;
}

inline const elaptime now(const std::string& function, const bool forced=false)
{
    if (DEBUG_PRINT_MESSAGE || forced) {
	    std::cout << function << " is started" << std::endl;
    }
    return std::chrono::system_clock::now();
}
inline const void print_time(const elaptime& start, const char* function, const bool forced=false)
{
	elaptime end = std::chrono::system_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (DEBUG_PRINT_MESSAGE || forced) {
	    std::cout << function << ", elapsed time(microseconds): " << elapsed << std::endl;
    }
}

TCV_E static TCV_M create_dark_channel(const TCV_M image, const uint winsize)
{
    const elaptime start = now(__FUNCTION__);
    TCV_M dark_channel = TCV_V::min(image, {winsize, winsize});
	if (DEBUG_SAVE_IMAGE == true) dark_channel->save("cpp_dark_channel.png", {1, 255.0});
	print_time(start, __FUNCTION__);
	return dark_channel;
}

TCV_E static TCV_M create_atmosphere(const TCV_M image, const TCV_M dark_channel)
{
	const elaptime start = now(__FUNCTION__);
    const TCV_S shape = image->shape();
	const uint pixels_size = shape[0] * shape[1];
	const uint pixels_search = static_cast<uint>(floor(static_cast<real>(pixels_size) * 0.01));

    TCV_M vec_dark, vec_image, accumulator;
    #pragma omp parallel
    #pragma omp sections
    {
        #pragma omp section
	    { vec_dark = dark_channel->reshape({pixels_size, 1, 1}); }
        #pragma omp section
	    { vec_image = image->reshape({pixels_size, 1, shape[2]}); }
        #pragma omp section
	    { accumulator = TCV_V::zeros({1, 1, shape[2]}); }
    }
	TCV_M indices = vec_dark->argsort(0, true);

	for (uint k = 0; k < pixels_search; k++) {
		const uint index = (*indices->template at<uint, 1>(k, 0))[0];
		accumulator = accumulator + (*vec_image)[index];
	}
	TCV_M atmosphere = accumulator / static_cast<T>(pixels_search);
	TCV_P pixel = atmosphere->template at<T, 3>(0, 0);
    if (DEBUG_PRINT_MESSAGE) {
	    std::cout << "atmosphere:" << (*pixel)[0] << ", " << (*pixel)[1] << ", " << (*pixel)[2] << std::endl;
    }
	print_time(start, __FUNCTION__);
	return atmosphere;
}

TCV_E static TCV_M estimate_transmission(const TCV_M image, TCV_M atmosphere, const real omega, const uint winsize)
{
	const elaptime start = now(__FUNCTION__);
	const TCV_S shape = image->shape();
	TCV_M rep_atmosphere = TCV_V::fill({shape[0], shape[1], shape[2]}, atmosphere);
    TCV_M transmission = (static_cast<T>(1.0) - omega * create_dark_channel(image / rep_atmosphere, winsize))->transpose({0, 0, 0});
	if (DEBUG_SAVE_IMAGE == true) rep_atmosphere->save("cpp_rep_atmosphere.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) transmission->save("cpp_transmission.png", {static_cast<T>(shape[2]), 255.0});

	print_time(start, __FUNCTION__);
	return transmission;
}

TCV_E static TCV_M adapt_window_sum_filter(TCV_M image, const uint r) 
{
	const elaptime start = now(__FUNCTION__);

	const TCV_S shape = image->shape();
	const uint sh = shape[0];
	const uint sw = shape[1];

    const T SUM_RATE = 0.4;
	const uint h = static_cast<uint>(sh * SUM_RATE);
	const uint w = static_cast<uint>(sw * SUM_RATE);
	TCV_M result = TCV_V::zeros({h, w, shape[2]});

    TCV_M resized = image->resize({SUM_RATE, SUM_RATE});
    TCV_M cumsum_y = resized->cumsum(0);
    #pragma omp parallel
    #pragma omp sections
    {
        #pragma omp section
        { (*(*result)({1-1, 0, (r+1)-(1-1), w-0})) = (*cumsum_y)({1-1+r, 0, (2*r+1)-(1-1+r), w-0}); }
        #pragma omp section
        { (*(*result)({r+2-1, 0, (h-r)-(r+2-1), w-0})) = (*cumsum_y)({2*r+2-1, 0, (h)-(2*r+2-1), w-0}) - (*cumsum_y)({1-1, 0, (h-2*r-1)-(1-1), w-0}); }
        #pragma omp section
        { (*(*result)({h-r+1-1, 0, (h)-(h-r+1-1), w-0})) = ((*cumsum_y)({h-1, 0, 1, w-0})->repmat({r, 1})) - (*cumsum_y)({h-2*r-1, 0, (h-r-1)-(h-2*r-1), w-0}); }
    }

    TCV_M cumsum_x = result->cumsum(1);
    #pragma omp parallel
    #pragma omp sections
    {
        #pragma omp section
        { (*(*result)({0, 1-1, h-0, (r+1)-(1-1)})) = (*cumsum_x)({0, 1+r-1, h-0, (2*r+1)-(1+r-1)}); }
        #pragma omp section
        { (*(*result)({0, r+2-1, h-0, (w-r)-(r+2-1)})) = (*cumsum_x)({0, 2*r+2-1, h-0, (w)-(2*r+2-1)}) - (*cumsum_x)({0, 1-1, h-0, (w-2*r-1)-(1-1)}); }
        #pragma omp section
        { (*(*result)({0, w-r+1-1, h-0, (w)-(w-r+1-1)})) = ((*cumsum_x)({0, w-1, h-0, 1})->repmat({1, r})) - (*cumsum_x)({0, w-2*r-1, h-0, (w-r-1)-(w-2*r-1)}); } 
    }
    result = result->resize({static_cast<T>(sh), static_cast<T>(sw)}, false);

	print_time(start, __FUNCTION__);
	return result;
}

TCV_E static TCV_M adapt_guided_filter(TCV_M image, const TCV_M target, const real radius, const real eps)
{
	const elaptime start = now(__FUNCTION__);

	const TCV_S shape = image->shape();
	TCV_M avg_denom = adapt_window_sum_filter(TCV_V::ones({shape[0], shape[1], 3}), radius);
    TCV_M mean_g, mean_t, corr_g, corr_gt;
    #pragma omp parallel
    #pragma omp sections
    {
        #pragma omp section
        { mean_g = adapt_window_sum_filter(image, radius) / avg_denom; }
        #pragma omp section
        { mean_t = adapt_window_sum_filter(target, radius) / avg_denom; }
        #pragma omp section
        { corr_g = adapt_window_sum_filter(image * image, radius) / avg_denom; }
        #pragma omp section
        { corr_gt = adapt_window_sum_filter(image * target, radius) / avg_denom; }
    }

	if (DEBUG_SAVE_IMAGE == true) avg_denom->save("cpp_avg_denom.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) mean_g->save("cpp_mean_g.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) mean_t->save("cpp_mean_t.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) corr_g->save("cpp_corr_g.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) corr_gt->save("cpp_corr_gt.png", {static_cast<T>(shape[2]), 255.0});

    TCV_M var_g, cov_gt;
    #pragma omp parallel
    #pragma omp sections
    {
        #pragma omp section
        { var_g = corr_g - (mean_g * mean_g); }	
        #pragma omp section
        { cov_gt = corr_gt - (mean_g * mean_t); }
    }    
	TCV_M a = cov_gt / (var_g + eps);
	TCV_M b = mean_t - a * mean_g;

    TCV_M mean_a, mean_b;
    #pragma omp parallel
    #pragma omp sections
    {
        #pragma omp section
        { mean_a = adapt_window_sum_filter(a, radius) / avg_denom; }
        #pragma omp section
        { mean_b = adapt_window_sum_filter(b, radius) / avg_denom; }
    }
	TCV_M q = mean_a * image + mean_b;

	if (DEBUG_SAVE_IMAGE == true) var_g->save("cpp_var_g.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) cov_gt->save("cpp_cov_gt.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) a->save("cpp_a.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) b->save("cpp_b.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) mean_a->save("cpp_mean_a.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) mean_b->save("cpp_mean_b.png", {static_cast<T>(shape[2]), 255.0});
	if (DEBUG_SAVE_IMAGE == true) q->save("cpp_q.png", {static_cast<T>(shape[2]), 255.0});

	print_time(start, __FUNCTION__);
	return q;
}

TCV_E static TCV_M create_radiance(const TCV_M image, TCV_M transmission, TCV_M atmosphere)
{
	const elaptime start = now(__FUNCTION__);

	TCV_S shape = image->shape();
	TCV_M rep_atmosphere = TCV_V::fill({shape[0], shape[1], shape[2]}, atmosphere);
	TCV_M radiance = TCV_V::absolute(((image - rep_atmosphere) / transmission) + rep_atmosphere); // transmission=max_transmission

	print_time(start, __FUNCTION__);
	return radiance;
}

TCV_E static TCV_M be_grayscale(const TCV_M image)
{
	const elaptime start = now(__FUNCTION__);
	TCV_M grayscale = TCV_V::grayscale(image);
	print_time(start, __FUNCTION__);
	return grayscale;
}

TCV_E static std::vector<TCV_M> facade_dark_channel_prior(const TCV_M image, const uint winsize=5, const real omega=0.95, const uint r=15, const real res=0.001)
{
#if defined(BUILD_CPU) || defined(BUILD_CUDA) || defined(BUILD_X5) || defined(BUILD_CL)
	const elaptime start = now(__FUNCTION__, true);
	const TCV_S shape = image->shape();

	TCV_M channel = create_dark_channel<C, T>(image, winsize);
	TCV_M atmosphere = create_atmosphere<C, T>(image, channel);
	TCV_M gray = be_grayscale<C, T>(image);

	TCV_M trans_estimation = estimate_transmission<C, T>(image, atmosphere, omega, winsize);
	if (DEBUG_SAVE_IMAGE == true) gray->save("cpp_grayscale.png", {static_cast<T>(shape[2]), 255.0});
	TCV_M x = adapt_guided_filter<C>(gray, trans_estimation, r, res);
	if (DEBUG_SAVE_IMAGE == true) x->save("cpp_x.png"); 
	TCV_M transmission = x->reshape({shape[0], shape[1], shape[2]});
	TCV_M radiance = create_radiance<C, T>(image, transmission, atmosphere);
	print_time(start, __FUNCTION__, true);
	return {radiance, x};
#else
    throw std::runtime_error(std::string(__FUNCTION__) + ": Invalid string of calculation resource type");
#endif
}

#ifdef BUILD_CUDA
static void print_gpu_information()
{
	const elaptime start = now(__FUNCTION__);
	const int num = cv::cuda::getCudaEnabledDeviceCount();
	if (num > 0) {
		std::cout << "GPU informaton is following" << std::endl;
		for (int g = 0; g < num; g++) {
			cv::cuda::DeviceInfo info(g);	 
			std::cout << "number: " << g << std::endl;
			std::cout << "name: " << info.name() << std::endl;
			std::cout << "majorVersion: " << info.majorVersion() << std::endl;
			std::cout << "minorVersion: " << info.minorVersion() << std::endl;
			std::cout << "multiProcessorCount: " << info.multiProcessorCount() << std::endl;
			std::cout << "sharedMemPerBlock: " << info.sharedMemPerBlock() << std::endl;
			std::cout << "freeMemory: " << info.freeMemory() << std::endl;
			std::cout << "totalMemory: " << info.totalMemory() << std::endl;
			std::cout << "isCompatible: " << info.isCompatible() << std::endl;
			std::cout << "supports(FEATURE_SET_COMPUTE_10): " 
                << info.supports(cv::cuda::FEATURE_SET_COMPUTE_10) << std::endl;	
			std::cout << "supports(FEATURE_SET_COMPUTE_11): " 
                << info.supports(cv::cuda::FEATURE_SET_COMPUTE_11) << std::endl;	
			std::cout << "supports(FEATURE_SET_COMPUTE_12): " 
                << info.supports(cv::cuda::FEATURE_SET_COMPUTE_12) << std::endl;	
			std::cout << "supports(FEATURE_SET_COMPUTE_13): " 
                << info.supports(cv::cuda::FEATURE_SET_COMPUTE_13) << std::endl;	
			std::cout << "supports(FEATURE_SET_COMPUTE_20): " 
                << info.supports(cv::cuda::FEATURE_SET_COMPUTE_20) << std::endl;	
			std::cout << "supports(FEATURE_SET_COMPUTE_21): " 
                << info.supports(cv::cuda::FEATURE_SET_COMPUTE_21) << std::endl;	
			std::cout << "supports(FEATURE_SET_COMPUTE_30): " 
                << info.supports(cv::cuda::FEATURE_SET_COMPUTE_30) << std::endl;	
			std::cout << "supports(FEATURE_SET_COMPUTE_35): " 
                << info.supports(cv::cuda::FEATURE_SET_COMPUTE_35) << std::endl;	
		}
	} else {
		throw std::runtime_error(std::string(__FUNCTION__) + ": GPU is not found");
	}
	print_time(start, __FUNCTION__);
}	
#endif

#ifdef BUILD_CL
static void print_cl_information()
{
    if (!cv::ocl::haveOpenCL()) {
        throw std::runtime_error("OpenCL not available on this system");
    }
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU)) {
        throw std::runtime_error("Context creation failed");
    }
    std::cout << context.ndevices() << " GPU device(s) detected" << std::endl;
    for (uint i = 0; i < context.ndevices(); i++) {
        cv::ocl::Device device = context.device(i);
        std::cout << "Device " << i << std::endl;
        std::cout << "\tName: " << device.name() << std::endl;
        std::cout << "\tAvailability: " << device.available() << std::endl;
        std::cout << "\tImage Support: " << device.imageSupport() << std::endl;
        std::cout << "\tOpenCL C version: " << device.OpenCL_C_Version() << std::endl;
    }
    cv::ocl::setUseOpenCL(true);
    cv::ocl::Device(context.device(0));
}
#endif

#if defined(BUILD_CPU) || defined(BUILD_CUDA) || defined(BUILD_X5) || defined(BUILD_CL)
#include <QApplication>
#include <QGraphicsItem>
#include <QGraphicsView>
#include <QtConcurrent>

class QGraphicsViewDehazeImpl
{
    typedef std::vector<cv::Mat> Mats;
    public:
        QGraphicsViewDehazeImpl(QGraphicsViewDehaze* parent, QTimer* timer, const char* type, const real rate, const int step) 
            : _parent(parent), _timer(timer), _type(type), _step(step), _rate(rate), _count(0) {};
        ~QGraphicsViewDehazeImpl() { if (_count > 0) _video.release(); }

        const void print_image(const cv::Mat& source) {
            cv::Mat target;
            source.convertTo(target, CV_MAKETYPE(cv::DataType<uchar>::type, (3)), 255.0);
            cv::cvtColor(target, target, CV_RGB2BGR);
            QImage image(target.data, target.cols, target.rows, target.step, QImage::Format_RGB888);
            QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(image));
            QGraphicsScene* scene = _parent->scene();
            scene->clear();
            scene->addItem(item);
        }

#ifdef BUILD_VIDEO
        const void convert_video() {
            if (_count == 0) {
                _video = cv::VideoCapture("wow.mpg");
            }
            _video >> _frame;
            if (_frame.rows == 0 && _frame.cols == 0) {
                _timer->stop();
                return;
            }
            if ((_count % _step) == 0) {
                if (DEBUG_PRINT_MESSAGE) print_line("=");
                std::cout << "frame, x:" << (_frame.cols * _rate) << ", y:" << (_frame.rows * _rate) << std::endl;
                cv::Mat display(_frame.rows * _rate, _frame.cols * _rate, _frame.type());
                cv::resize(_frame, display, display.size(), cv::INTER_CUBIC);

                const elaptime startg = now(__FUNCTION__);
                Mats dehazed = convert_video_concrete(display);
                const elaptime overg = now(__FUNCTION__);
                if (DEBUG_PRINT_MESSAGE) {
                    double elapsedg = std::chrono::duration_cast<std::chrono::microseconds>(overg - startg).count();
                    std::cout << "facade_dark_channel_prior, elapsed time(microseconds):" << elapsedg << std::endl;
                }

                cv::Mat result(_frame.rows * _rate * 2.0, _frame.cols * _rate * 2.0, dehazed[0].type()), gray;
                display.convertTo(display, CV_MAKETYPE(cv::DataType<real>::type, (3)), 1.0/255.0);
                display.copyTo(result(cv::Rect(0, 0, display.cols, display.rows)));
                dehazed[0].copyTo(result(cv::Rect(0, display.rows, dehazed[0].cols, dehazed[0].rows)));
                dehazed[1].copyTo(result(cv::Rect(display.cols, display.rows, dehazed[1].cols, dehazed[1].rows)));

                cv::cvtColor(dehazed[0], gray, CV_BGR2GRAY);
                cv::cvtColor(gray, gray, CV_GRAY2BGR);
                gray.copyTo(result(cv::Rect(display.cols, 0, gray.cols, gray.rows)));
                print_image(result);

                if (_count == 0) {
                    QGraphicsScene* scene = _parent->QGraphicsView::scene();
                    const QRectF& rect = scene->sceneRect();
                    _parent->setFixedSize(rect.width(), rect.height());
                    _parent->setSceneRect(0, 0, rect.width(), rect.height());
                    _parent->fitInView(0, 0, rect.width(), rect.height(), Qt::KeepAspectRatio);
                }
            }
            _count++;
        }
#endif
        Mats convert_video_concrete(cv::Mat& display) {
            if (std::string(_type) == "CPU") {
#ifdef BUILD_CPU
                MatrixPtr<Cpu<real>, real> imageg = tinycv::make_shared<Matrix<Cpu<real>, real>>(&display);
                std::vector<MatrixPtr<Cpu<real>, real>> results = facade_dark_channel_prior<Cpu<real>, real>(imageg);
                return {*((cv::Mat*)results[0]->matrix()), *((cv::Mat*)results[1]->matrix())};
#else
                throw std::runtime_error(std::string(__FUNCTION__) + ": CPU processing is not compiled");
#endif
            }
            if (std::string(_type) == "CUDA") {
#ifdef BUILD_CUDA
                MatrixPtr<Cuda<real>, real> imageg = tinycv::make_shared<Matrix<Cuda<real>, real>>(&display);
                std::vector<MatrixPtr<Cuda<real>, real>> results = facade_dark_channel_prior<Cuda<real>, real>(imageg);
                return {*((cv::Mat*)results[0]->matrix()), *((cv::Mat*)results[1]->matrix())};
#else
                throw std::runtime_error(std::string(__FUNCTION__) + ": CUDA processing is not compiled");
#endif
            }
            if (std::string(_type) == "X5") {
#ifdef BUILD_X5
                MatrixPtr<X5<real>, real> imageg = tinycv::make_shared<Matrix<X5<real>, real>>(&display);
                std::vector<MatrixPtr<X5<real>, real>> results = facade_dark_channel_prior<X5<real>, real>(imageg);
                return {*((cv::Mat*)results[0]->matrix()), *((cv::Mat*)results[1]->matrix())};
#else
                throw std::runtime_error(std::string(__FUNCTION__) + ": X5 processing is not compiled");
#endif
            }
            if (std::string(_type) == "CL") {
#ifdef BUILD_CL
                MatrixPtr<Cl<real>, real> imageg = tinycv::make_shared<Matrix<Cl<real>, real>>(&display);
                std::vector<MatrixPtr<Cl<real>, real>> results = facade_dark_channel_prior<Cl<real>, real>(imageg);
                return {*((cv::Mat*)results[0]->matrix()), *((cv::Mat*)results[1]->matrix())};
#else
                throw std::runtime_error(std::string(__FUNCTION__) + ": OpenCL processing is not compiled");
#endif
            }
            throw std::runtime_error(std::string(__FUNCTION__) + ": Invalid string of calculation resource type");
        }

    private:
        QGraphicsViewDehaze* _parent;
        QTimer* _timer;
        cv::VideoCapture _video;
        cv::Mat _frame;
        const char* _type;
        const int _step;
        const real _rate;
        uint _count;
};


QGraphicsViewDehaze::QGraphicsViewDehaze(QGraphicsScene *scene, QTimer* timer, QWidget* parent, const char* type, const real rate, const int step)
    : QGraphicsView(scene, parent) 
{
    _impl = new QGraphicsViewDehazeImpl(this, timer, type, rate, step);
}

QGraphicsViewDehaze::~QGraphicsViewDehaze()
{
    delete _impl;
}

void QGraphicsViewDehaze::advance()
{
#ifdef BUILD_VIDEO
    _impl->convert_video();
#else
    throw std::runtime_error(std::string(__FUNCTION__) + ": Video processing is not compiled");
#endif
}
#endif

#if defined(BUILD_CPU) || defined(BUILD_CUDA) || defined(BUILD_X5) || defined(BUILD_CL)
template<class C, typename T>
void do_simple_processing(QGraphicsViewDehaze* view)
{
    MatrixPtr<C, T> imageg = Matrix<C, T>::load("sample1.png", {3, 1.0/255.0});
    print_pixels<C, T, T, 3>(imageg, "image of sample1.png", true);
    //imageg = Matrix<C, T>::equalize(imageg);
    if (DEBUG_SAVE_IMAGE == true) imageg->save("cpp_source.png", {3, 255.0});
    std::vector<MatrixPtr<C, T>> dcpg = facade_dark_channel_prior(imageg);
    if (DEBUG_SAVE_IMAGE == true) dcpg[0]->save("cpp_result.png", {3, 255.0});
    view->impl()->print_image(*((cv::Mat*)dcpg[0]->matrix()));
}

#ifdef BUILD_MEASURE
template<class C, typename T>
void measure_performance()
{
    const int LIMIT = 100000;    
    const double PERF_RATE = 0.01;

    std::ofstream ofsg("measure_cpp.csv");
    real rate = 1.0;
    for (int i = 0; i < LIMIT; i++) {
        for (int j = 0; j < LIMIT; j++) {
            if (DEBUG_PRINT_MESSAGE) {
                print_line("=");
                std::cout << "Measure processing is started(index: " << i << ", " << j << ")" << std::endl;
                print_line("-");
            }
            MatrixPtr<C, T> imageg = Matrix<C, T>::load("sample1.png", {3, 1.0/255.0});
            imageg = imageg->resize({rate, rate});
            //imageg = Matrix<C, T>::equalize(imageg);
            print_pixels<C, T, T, 3>(imageg, "image of sample1.png", true);
            if (DEBUG_SAVE_IMAGE == true) imageg->save("cpp_source.png", {3, 255.0});
            const elaptime startg = now(__FUNCTION__);
            std::vector<MatrixPtr<C, T>> dcpg = facade_dark_channel_prior(imageg);
            const elaptime overg = now(__FUNCTION__);
            double elapsedg = std::chrono::duration_cast<std::chrono::microseconds>(overg - startg).count();
            ofsg << (j == 0 ? "," : ",") << elapsedg;
            if (DEBUG_SAVE_IMAGE == true) dcpg[0]->save("cpp_result.png", {3, 255.0});
        }        
        rate += PERF_RATE;
        ofsg << std::endl;
    }
    ofsg.close();
}
#endif
#endif

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "Usage ./Dehaze {CPU|CUDA|X5|CL} {1=SIMPLE|2=MEASURE|3=VIDEO} {f=RATE} ({n=STEP})" << std::endl;
        return 0;
    }

    print_line("=");
    try {
#ifdef BUILD_CUDA
        if (std::string(argv[1]) == "CUDA") {
            print_line("-");
            print_gpu_information();
            print_line("-");
        }
#endif
#ifdef BUILD_CL
        if (std::string(argv[1]) == "CL") {
            print_line("-");
            print_cl_information();
            print_line("-");
        }
#endif

        QApplication app(argc, argv);
        QGraphicsScene scene(Q_NULLPTR);
        QTimer timer;
        QGraphicsViewDehaze view(&scene, &timer, Q_NULLPTR, argv[1], 
            argc > 3 ? std::stof(argv[3]) : 0.25, argc > 4 ? std::stoi(argv[4]) : 10);

        if (std::string(argv[1]) == "CPU") {
#ifdef BUILD_CPU
            if (std::string(argv[2]) == "1") {
                do_simple_processing<Cpu<real>, real>(&view);
            } 
            if (std::string(argv[2]) == "2") {
#ifdef BUILD_MEASURE
                measure_performance<Cpu<real>, real>();
#else
                throw std::runtime_error(std::string(__FUNCTION__) + ": measure_performance is not compiled");
#endif
            } 
#else
            throw std::runtime_error(std::string(__FUNCTION__) + ": CPU processing is not compiled");
#endif
        }

        if (std::string(argv[1]) == "CUDA") {
#ifdef BUILD_CUDA
            if (std::string(argv[2]) == "1") {
                do_simple_processing<Cuda<real>, real>(&view);
            }
            if (std::string(argv[2]) == "2") {
#ifdef BUILD_MEASURE
                measure_performance<Cuda<real>, real>();
#else
                throw std::runtime_error(std::string(__FUNCTION__) + ": measure_performance is not compiled");
#endif
            } 
#else
            throw std::runtime_error(std::string(__FUNCTION__) + ": CUDA processing is not compiled");
#endif
        }

        if (std::string(argv[1]) == "X5") {
#ifdef BUILD_CUDA
            if (std::string(argv[2]) == "1") {
                do_simple_processing<X5<real>, real>(&view);
            }
            if (std::string(argv[2]) == "2") {
#ifdef BUILD_MEASURE
                measure_performance<X5<real>, real>();
#else
                throw std::runtime_error(std::string(__FUNCTION__) + ": measure_performance is not compiled");
#endif
            } 
#else
            throw std::runtime_error(std::string(__FUNCTION__) + ": X5 processing is not compiled");
#endif
        }

        if (std::string(argv[1]) == "CL") {
#ifdef BUILD_CL
            if (std::string(argv[2]) == "1") {
                do_simple_processing<Cl<real>, real>(&view);
            }
            if (std::string(argv[2]) == "2") {
#ifdef BUILD_MEASURE
                measure_performance<Cl<real>, real>();
#else
                throw std::runtime_error(std::string(__FUNCTION__) + ": measure_performance is not compiled");
#endif
            } 
#else
            throw std::runtime_error(std::string(__FUNCTION__) + ": OpenCL processing is not compiled");
#endif
        }

        if (std::string(argv[2]) == "3") {
#ifdef BUILD_VIDEO
            QObject::connect(&timer, SIGNAL(timeout()), &view, SLOT(advance()));
            timer.setInterval(static_cast<int>(1000/30));
            timer.start(0);
#else
            throw std::runtime_error(std::string(__FUNCTION__) + ": convert_video is not compiled");
#endif
        }

        if (std::string(argv[2]) == "1" || std::string(argv[2]) == "3") {
            view.show();
	        app.exec();
        }

    } catch (std::exception& err) {
        std::cout << err.what() << std::endl;
    }

};

