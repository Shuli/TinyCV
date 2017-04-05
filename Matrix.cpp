/*!
 * @author Hisashi Ikari
 */

#include "Matrix.h"

#if defined(BUILD_CPU) || defined(BUILD_CUDA) || defined(BUILD_X5) || defined(BUILD_CL)
#include <opencv2/opencv.hpp>
#endif
#ifdef BUILD_CL
#include <opencv2/core/ocl.hpp> // for 3.2
#endif

#ifdef BUILD_CUDA
template<typename T> using vector = std::vector<T>;
#include <opencv2/core/cuda.hpp> // for 3.2
#endif

#ifdef BUILD_X5
#ifdef FEATURE_VERSION	
extern "C" {
#include <libimp.h> // for X5
}
#endif
#endif

namespace tinycv {

template<class C> using ContainerPtr = std::shared_ptr<C>;
#if defined(BUILD_CPU) || defined(BUILD_CUDA) || defined(BUILD_X5) || defined(BUILD_CL)
typedef std::shared_ptr<std::vector<cv::Mat>> MatsPtr;
#endif

#ifdef BUILD_CPU
template<typename T>
class Cpu
{
    typedef cv::Mat Image;
	public:
		Cpu(const Shape& dimension, const T initialization=0.0, const bool random=false) {
			_matrix = Image(dimension[0], dimension[1], 
				CV_MAKETYPE(cv::DataType<T>::type, (3)), 
                (random==true ? cv::Scalar::all(std::rand()) : cv::Scalar::all(1.0))) * initialization;
			_matrix.convertTo(_matrix, CV_MAKETYPE(cv::DataType<T>::type, (static_cast<uint>(dimension[2]))));
		}
		Cpu(const Shape& dimension, const PixelPtr<T> init) {
			_matrix = Image(dimension[0], dimension[1], 
				CV_MAKETYPE(cv::DataType<T>::type, (static_cast<uint>(dimension[2]))), cv::Scalar((*init)[0], (*init)[1], (*init)[2]));
		}
		Cpu(const ContainerPtr<Cpu<T>> cpu, bool deepcopy) {
			Image source = cpu->matrix();
			_matrix = deepcopy==true ? source.clone() : source;
		}
		Cpu(const std::string& name, const Adjust<T>& adjust) {
			_matrix = cv::imread(name, CV_MAKETYPE(cv::DataType<T>::type, (static_cast<uint>(adjust[0]))));
			_matrix.convertTo(_matrix, CV_MAKETYPE(cv::DataType<T>::type, (static_cast<uint>(adjust[0]))), adjust[1]);
		}
        Cpu(const void* matrix) {
            _matrix = ((Image*)matrix)->clone();
			_matrix.convertTo(_matrix, CV_MAKETYPE(cv::DataType<T>::type, (3)), 1.0/255.0);
        }
        Cpu() {}
		const void save(const std::string& name, const Adjust<T>& adjust, const bool textlog=true) {
			Image matrix = _matrix.clone();
			matrix.convertTo(matrix, CV_MAKETYPE(cv::DataType<T>::type, (static_cast<uint>(adjust[0]))), adjust[1]);
			cv::imwrite(name, matrix);
		}
		const void reshape(const Shape& dimension) {
			_matrix = _matrix.reshape(dimension[2], dimension[0]);
		}
		const void resize(const Adjust<T>& adjust, const bool rate) {
			cv::resize(_matrix, _matrix, cv::Size(rate==true ? adjust[1] * _matrix.cols : adjust[1],
                        rate==true ? adjust[0] * _matrix.rows : adjust[0]), cv::INTER_CUBIC);
        }
		const void repmat(const Shape& dimension) {
			const int ny = _matrix.rows * dimension[0];
			const int nx = _matrix.cols * dimension[1];
			Image matrix = Image(ny, nx, CV_MAKETYPE(cv::DataType<T>::type, (3)), cv::Scalar::all(0.0));
			matrix.convertTo(matrix, CV_MAKETYPE(cv::DataType<T>::type, (static_cast<uint>(_matrix.channels()))));
			for (int i = 0; i < ny; i+=_matrix.rows) {
				for (int j = 0; j < nx; j+=_matrix.cols) {
					_matrix.copyTo(matrix(cv::Rect(j, i, _matrix.cols, _matrix.rows)));
				}
			}
			_matrix = matrix;
		}
        const void transpose(const Adjust<T>& adjust) {
            const int size = adjust.size();
            MatsPtr mats = split(size);
            std::vector<Image> channels(size);
            for (int i = 0; i < size; i++) {
                channels[i] = (*mats)[static_cast<uint>(adjust[i])];
            }   
            cv::merge(channels, _matrix);
        }   
		const void cumsum(const uint axis) {
			Image matrix = Image(_matrix.rows, _matrix.cols, CV_MAKETYPE(cv::DataType<T>::type, (3)), cv::Scalar::all(0.0));
			matrix.convertTo(matrix, CV_MAKETYPE(cv::DataType<T>::type, (static_cast<uint>(_matrix.channels()))));
			_matrix = axis == 0 ? cumsumy(matrix) : cumsumx(matrix);
		}
		inline Image cumsumx(Image matrix) {
			_matrix.col(0).copyTo(matrix.col(0));
			for (int i = 1; i < _matrix.cols; i++) {
				Image sum = matrix.col(i - 1) + _matrix.col(i);
				sum.copyTo(matrix.col(i));
			}
			return matrix;
		}
		inline Image cumsumy(Image matrix) {
			_matrix.row(0).copyTo(matrix.row(0));
			for (int i = 1; i < _matrix.rows; i++) {
				Image sum = matrix.row(i - 1) + _matrix.row(i);
				sum.copyTo(matrix.row(i));
			}
			return matrix;
		}
		const void range(const Shape& scope) {	
			cv::Rect rect(scope[1], scope[0], scope[3], scope[2]);
			_matrix = _matrix(rect);
		}
		const void argsort(const uint axis, const bool reverse) {
			cv::sortIdx(_matrix, _matrix,
				(axis == 0 ? CV_SORT_EVERY_ROW : CV_SORT_EVERY_COLUMN) | (reverse==true ? CV_SORT_DESCENDING : CV_SORT_ASCENDING));
		}
		template<typename R, int D>
		const PixelPtr<R> at(const uint y, const uint x) const {
			PixelPtr<R> result = tinycv::make_shared<Pixel<R>>(D);
			const cv::Vec<R, D>* pixel = _matrix.ptr<cv::Vec<R, D>>(y);
			for (uint c = 0; c < D; c++) {
				(*result)[c] = pixel[x][c];
			}
			return result;
		}
		const void substitution(const Pixel<T>& pixel) {
			Image image = Image(1, 1, _matrix.type(), const_cast<T*>(pixel.data()));
			image.copyTo(_matrix(cv::Rect(0, 0, image.cols, image.rows)));	
		}
		const void substitution(const ContainerPtr<Cpu<T>> right) {
			Image image = right->matrix();
			image.copyTo(_matrix(cv::Rect(0, 0, image.cols, image.rows)));	
		}
		virtual const void addition(const ContainerPtr<Cpu<T>> right) {
			_matrix = _matrix + right->matrix();
		}
		virtual const void addition(const T right, const bool reverse) {
			_matrix = reverse==true ? right + _matrix : _matrix + right;
		}
		virtual const void subtraction(const ContainerPtr<Cpu<T>> right) {
			_matrix = _matrix - right->matrix();
		}
		virtual const void subtraction(const T right, const bool reverse) {
			_matrix = reverse==true ? right - _matrix : _matrix - right;
		}
		virtual const void multiplication(const ContainerPtr<Cpu<T>> right) {
			_matrix = _matrix.mul(right->matrix());
		}
		virtual const void multiplication(const T right, const bool reverse) {
			_matrix = reverse==true ? right * _matrix : _matrix * right;
		}
		virtual const void division(const ContainerPtr<Cpu<T>> right) {
			_matrix = _matrix / right->matrix();
		}
		virtual const void division(const T right, const bool reverse) {
			_matrix = reverse==true ? right / _matrix : _matrix / right;
		}
		virtual const void absolute(const ContainerPtr<Cpu<T>> right) {
			_matrix = cv::abs(right->matrix());
		}
		virtual const void grayscale(const ContainerPtr<Cpu<T>> right) {
			cv::cvtColor(right->matrix(), _matrix, CV_BGR2GRAY);
			cv::cvtColor(_matrix, _matrix, CV_GRAY2BGR);
		}	
        virtual const void equalize(const ContainerPtr<Cpu<T>> right) {
			right->matrix().convertTo(_matrix, CV_MAKETYPE(cv::DataType<uchar>::type, (3)), 255.0);
			MatsPtr channels = tinycv::make_shared<std::vector<cv::Mat>>(3);
			cv::split(_matrix, (*channels));
            for (int i = 0; i < 3; i++) {
                cv::equalizeHist((*channels)[i], (*channels)[i]);
            }
            cv::merge(*channels, _matrix);
			_matrix.convertTo(_matrix, CV_MAKETYPE(cv::DataType<T>::type, (3)), 1.0/255.0);
        }
		const T min() const {
            PixelPtr<T> result = mins();
            return static_cast<T>(*std::min_element(result->begin(), result->end()));
		}
		const T max() const {
            PixelPtr<T> result = maxs();
            return static_cast<T>(*std::max_element(result->begin(), result->end()));
		}
        virtual const void min(const Shape& shape) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(shape[1], shape[0]));
			cv::cvtColor(_matrix, _matrix, CV_BGR2GRAY);
            cv::erode(_matrix, _matrix, kernel, cv::Point(-1, -1), 1);
        }
        virtual const void max(const Shape& shape) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(shape[1], shape[0]));
			cv::cvtColor(_matrix, _matrix, CV_BGR2GRAY);
            cv::dilate(_matrix, _matrix, kernel, cv::Point(-1, -1), 1);
        }
		const PixelPtr<T> mins() const {
			const uint channel = shape()[2];
			PixelPtr<T> result = tinycv::make_shared<Pixel<T>>(channel);
			MatsPtr mats = split(channel);
			for (uint i = 0; i < channel; i++) {
				double min = 0.0;
				cv::minMaxLoc((*mats)[i], &min, 0, 0, 0);
				(*result)[i] = static_cast<T>(min);
			}					
			return result;
		}
		const PixelPtr<T> maxs() const {
			const uint channel = shape()[2];
			PixelPtr<T> result = tinycv::make_shared<Pixel<T>>(channel);
			MatsPtr mats = split(channel);
			for (uint i = 0; i < channel; i++) {
				double max = 0.0;
				cv::minMaxLoc((*mats)[i], 0, &max, 0, 0);
				(*result)[i] = static_cast<T>(max);
			}					
			return result;
		}
		inline MatsPtr split(const uint channel) const {
			MatsPtr channels = tinycv::make_shared<std::vector<cv::Mat>>(channel);
			cv::split(_matrix, (*channels));
			return channels;	
		}
		virtual Image matrix() const { return _matrix; }
		virtual const void* matrixp() const { return (void*)(&_matrix); }
		virtual const Shape shape() const {
			return {
				static_cast<uint>(_matrix.rows),			
				static_cast<uint>(_matrix.cols),
				static_cast<uint>(_matrix.channels())
			};			
		}
        static const void initialize(const bool forced=false) {
            throw std::runtime_error("In the CPU version, there is no need to initialize.");
        }
        static const void terminate(const bool forced=false) {
            throw std::runtime_error("In the CPU version, there is no need to terminate.");
        }

	protected:
		Image _matrix;
};
#endif

#ifdef BUILD_CUDA
template<typename T>
class Cuda : public Cpu<T> 
{
    typedef cv::Mat Image;
	typedef cv::cuda::GpuMat Resource;
	typedef cv::cuda::Stream Stream;
	public:
		Cuda(const Shape& dimension, const T initialization=0.0, const bool random=false) 
            : Cpu<T>(dimension, initialization, random) {}
		Cuda(const Shape& dimension, const PixelPtr<T> init) : Cpu<T>(dimension, init) {}
		Cuda(const ContainerPtr<Cuda<T>> cuda, bool deepcopy) : Cpu<T>(cuda, deepcopy) {}
		Cuda(const std::string& name, const Adjust<T>& adjust) : Cpu<T>(name, adjust) {}
        Cuda(const void* matrix) : Cpu<T>(matrix) {}
#ifdef FEATURE_VERSION	
		virtual const void addition(const ContainerPtr<Cuda<T>> right) {
			cv::cuda::add(_resource, right->resource(), _resource);
		}
		virtual const void addition(const T right, const bool reverse) {
			if (reverse) {
				cv::cuda::add(right, _resource, _resource);
			} else {
				cv::cuda::add(_resource, right, _resource);
			}
		}
		virtual const void subtraction(const ContainerPtr<Cuda<T>> right) {
			cv::cuda::subtract(_resource, right->resource(), _resource);
		}
		virtual const void subtraction(const T right, const bool reverse) {
			if (reverse) {
				cv::cuda::subtract(right, _resource, _resource);
			} else {
				cv::cuda::subtract(_resource, right, _resource);
			}
		}
		virtual const void multiplication(const ContainerPtr<Cuda<T>> right) {
			cv::cuda::multiply(_resource, right->resource(), _resource);
		}
		virtual const void multiplication(const T right, const bool reverse) {
			if (reverse) {
				cv::cuda::multiply(right, _resource, _resource);
			} else {
				cv::cuda::multiply(_resource, right, _resource);
			}
		}
		virtual const void division(const ContainerPtr<Cuda<T>> right) {
			cv::cuda::divide(_resource, right->resource(), _resource);
		}
		virtual const void division(const T right, const bool reverse) {
			if (reverse) {
				cv::cuda::divide(right, _resource, _resource);
			} else {
				cv::cuda::divide(_resource, right, _resource);
			}
		}
		virtual const void absolute(const ContainerPtr<Cuda<T>> right) {
			cv::cuda::abs(right->matrix(), C::_matrix);
		}
		virtual const void grayscale(const ContainerPtr<Cuda<T>> right) {
			cv::cuda::cvtColor(right->matrix(), C::_matrix, CV_BGR2GRAY);
			cv::cuda::cvtColor(C::_matrix, C::_matrix, CV_GRAY2BGR); 
		}
#endif
        virtual const void min(const Shape& shape) {
            Image matrix = increase_channel(Cpu<T>::_matrix);
            _resource.upload(matrix);
            cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createBoxMinFilter(_resource.type(), cv::Size(shape[1], shape[0]));
            filter->apply(_resource, _resource);
            _resource.download(matrix);
            Cpu<T>::_matrix = decrease_channel(matrix);
        }
        virtual const void max(const Shape& shape) {
            Image matrix = increase_channel(Cpu<T>::_matrix);
            _resource.upload(matrix);
            cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createBoxMaxFilter(_resource.type(), cv::Size(shape[1], shape[0]));
            filter->apply(_resource, _resource);
            _resource.download(matrix);
            Cpu<T>::_matrix = decrease_channel(matrix);
        }
        inline Image increase_channel(Image& source) {
			source.convertTo(source, CV_MAKETYPE(cv::DataType<uchar>::type, (3)), 255.0);
			MatsPtr channels = tinycv::make_shared<std::vector<cv::Mat>>(3);
			cv::split(source, (*channels));
            channels->push_back(Image(source.rows, source.cols, CV_MAKETYPE(cv::DataType<uchar>::type, (1)), cv::Scalar::all(255.0)));
            Image destribution;
            cv::merge(*channels, destribution);
            return destribution;
        }
        inline Image decrease_channel(Image& source) {
			source.convertTo(source, CV_MAKETYPE(cv::DataType<T>::type, (static_cast<uint>(4))), 1.0/255.0);
			MatsPtr channels = tinycv::make_shared<std::vector<cv::Mat>>(4);
			cv::split(source, (*channels));
            channels->pop_back();
            Image destribution;
            cv::merge(*channels, destribution);
            return destribution;
        }
		Resource resource() const { return _resource; }

        static const void initialize(const bool forced=false) {
            throw std::runtime_error("In the CUDA version, there is no need to initialize.");
        }
        static const void terminate(const bool forced=false) {
            throw std::runtime_error("In the CUDA version, there is no need to terminate.");
        }
 
	protected:
        Resource _resource;
        Stream _stream;
};
#endif

#ifdef BUILD_X5
template<typename T>
class X5 : public Cpu<T>
{
    typedef cv::Mat Image;
	public:
		X5(const Shape& dimension, const T initialization=0.0, const bool random=false) 
            : Cpu<T>(dimension, initialization, random) {}
		X5(const Shape& dimension, const PixelPtr<T> init) : Cpu<T>(dimension, init) {}
		X5(const ContainerPtr<X5<T>> x5, bool deepcopy) : Cpu<T>(x5, deepcopy) {}
		X5(const std::string& name, const Adjust<T>& adjust) : Cpu<T>(name, adjust) {}
        X5(const void* matrix) : Cpu<T>(matrix) {}
        virtual const void min(const Shape& shape) {
#ifdef FEATURE_VERSION	
            IMPLIB_ImageFrameSize size;
            size.ylng = _matrix.rows;
            size.xlng = _matrix.cols;
            IMPLIB_IMGID img_src = implib_AllocImg(size);
            IMPLIB_IMGID img_dst = implib_AllocImg(size);
            transfer_matrix(img_src, size);
            int32_t result = implib_IP_MinFLT8(img_src, img_dst);
            if (result != 0) {
                throw std::runtime_error("Error of implib_IP_MinFLT8 is occured, number: " + std::string(result));
            }
            transfer_image(img_dst, size); 
#endif
        }
        virtual const void max(const Shape& shape) {
#ifdef FEATURE_VERSION	
            IMPLIB_ImageFrameSize size;
            size.ylng = _matrix.rows;
            size.xlng = _matrix.cols;
            IMPLIB_IMGID img_src = implib_AllocImg(size);
            IMPLIB_IMGID img_dst = implib_AllocImg(size);
            transfer_matrix(img_src, size);
            int32_t result = implib_IP_MaxFLT8(img_src, img_dst);
            if (result != 0) {
                throw std::runtime_error("Error of implib_IP_MinFLT8 is occured, number: " + std::string(result));
            }
            transfer_image(img_dst, size); 
#endif
        }
#ifdef FEATURE_VERSION	
        inline const void transfer_matrix(IMPLIB_IMGID source, IMPLIB_ImageFrameSize size) {
			cv::cvtColor(Cpu<T>::_matrix, Cpu<T>::_matrix, CV_BGR2GRAY);
			_matrix.convertTo(_matrix, CV_MAKETYPE(cv::DataType<uchar>::type, (1)), 255.0);
            for (int y = 0; y < size.ylng; y++) {
			    const cv::Vec<uchar, 1>* pixel = Cpu<T>::_matrix.ptr<cv::Vec<uchar, 1>>(y);
                for (int x = 0; x < size.xlng; x++) {
                    source[y][x] = pixel[x][0]; // /* or */ *(source + (y * size.ylng) + x) = pixel[x][0];
                }
            }        
        }
        inline const void transfer_image(IMPLIB_IMGID target, IMPLIB_ImageFrameSize size) {
            for (int y = 0; y < size.ylng; y++) {
			    const cv::Vec<uchar, 1>* pixel = Cpu<T>::_matrix.ptr<cv::Vec<uchar, 1>>(y);
                for (int x = 0; x < size.xlng; x++) {
                    pixel[x][0] = source[y][x]; // /* or */ *(source + (y * size.ylng) + x) = pixel[x][0];
                }
            }        
			Cpu<T>::_matrix.convertTo(Cpu<T>::_matrix, CV_MAKETYPE(cv::DataType<T>::type, (1)), 1.0/255.0);
			cv::cvtColor(Cpu<T>::_matrix, Cpu<T>::_matrix, CV_GRAY2BGR); // to three channels
        }
#endif
 
        static const void initialize(const bool forced=false) {
#ifdef FEATURE_VERSION	
            int32_t result = implib_Open();
            if (result != 0) {
                throw std::runtime_error("Error of implib_Open is occured, number: " + std::string(result));
            }
            result = implib_SetIPDataType(IMPLIB_UNSIGN_DATA); 
            if (result != 0) {
                throw std::runtime_error("Error of implib_SetIPDataType is occured, number: " + std::string(result));
            }
#endif
        }
        static const void terminate(const bool forced=false) {
#ifdef FEATURE_VERSION	
            int32_t result = implib_FreeAllImg();
            if (result != 0) {
                throw std::runtime_error("Erorr of implib_FreeAllImg is occured, number: " + std::string(result));
            }
            result = implib_Close();
            if (result != 0) {
                throw std::runtime_error("Erorr of implib_Close is occured, number: " + std::string(result));
            }
#endif
        }
};
#endif

#ifdef BUILD_CL
template<typename T>
class Cl : public Cpu<T>
{
    typedef cv::Mat Image;
	public:
		Cl(const Shape& dimension, const T initialization=0.0, const bool random=false) 
            : Cpu<T>(dimension, initialization, random) {}
		Cl(const Shape& dimension, const PixelPtr<T> init) : Cpu<T>(dimension, init) {}
		Cl(const ContainerPtr<Cl<T>> cl, bool deepcopy) : Cpu<T>(cl, deepcopy) {}
		Cl(const std::string& name, const Adjust<T>& adjust) : Cpu<T>(name, adjust) {}
        Cl(const void* matrix) : Cpu<T>(matrix) {}
        virtual const void min(const Shape& shape) {
			cv::cvtColor(Cpu<T>::_matrix, Cpu<T>::_matrix, CV_BGR2GRAY);
            cv::UMat erode = Cpu<T>::_matrix.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(shape[1], shape[0]));
            cv::erode(erode, erode, kernel, cv::Point(-1, -1), 1);
        }
        virtual const void max(const Shape& shape) {
			cv::cvtColor(Cpu<T>::_matrix, Cpu<T>::_matrix, CV_BGR2GRAY);
            cv::UMat dilate = Cpu<T>::_matrix.getUMat(cv::ACCESS_READ, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(shape[1], shape[0]));
            cv::dilate(dilate, dilate, kernel, cv::Point(-1, -1), 1);
        }
        static const void initialize(const bool forced=false) {
            throw std::runtime_error("In the OpenCL version, there is no need to initialize.");
        }
        static const void terminate(const bool forced=false) {
            throw std::runtime_error("In the OpenCL version, there is no need to terminate.");
        }
};
#endif


template<class C, typename T>
class MatrixImpl
{
	public:
		MatrixImpl(const Shape& dimension, const T initialization=0.0, const bool random=false) {
			_container = tinycv::make_shared<C>(dimension, initialization);
		}
		MatrixImpl(const Shape& dimension, const PixelPtr<T> initialization) {
			_container = tinycv::make_shared<C>(dimension, initialization);
		}
		MatrixImpl(const MatrixImplPtr<C, T> impl, bool deepcopy=true) {
			_container = tinycv::make_shared<C>(impl->container(), deepcopy);
		}
		MatrixImpl(const std::string& name, const Adjust<T>& adjust) {
			_container = tinycv::make_shared<C>(name, adjust);
		}
        MatrixImpl(const void* matrix) {
            _container = tinycv::make_shared<C>(matrix);
        }
		const void save(const std::string& name, const Adjust<T>& adjust, const bool textlog=false) {
			_container->save(name, adjust);
		}
		const void reshape(const Shape& dimension) {
			_container->reshape(dimension);
		}
		const void resize(const Adjust<T>& adjust, const bool rate) {
            _container->resize(adjust, rate);
        }
		const void repmat(const Shape& dimension) {
			_container->repmat(dimension);
		}
        const void transpose(const Adjust<T>& adjust) {
            _container->transpose(adjust);
        }
		const void cumsum(const uint axis) {
			_container->cumsum(axis);
		}
		const void range(const Shape& scope) {
			_container->range(scope);
		}
		const void argsort(const uint axis, const bool reverse) {	
			_container->argsort(axis, reverse);
		}
		template<typename R, int D>
		const PixelPtr<R> at(const uint y, const uint x) const {
			return _container->template at<R, D>(y, x);
		}
		const void substitution(const Pixel<T>& pixel) {
			_container->substitution(pixel);
		}
		const void substitution(const MatrixImplPtr<C, T> right) {
			_container->substitution(right->container());
		}
		const void addition(const MatrixImplPtr<C, T> right) {
			_container->addition(right->container());
		}
		const void addition(const T right, const bool reverse=false) {
			_container->addition(right, reverse);
		}
		const void subtraction(const MatrixImplPtr<C, T> right) {
			_container->subtraction(right->container());
		}
		const void subtraction(const T right, const bool reverse=false) {
			_container->subtraction(right, reverse);
		}
		const void multiplication(const MatrixImplPtr<C, T> right) {
			_container->multiplication(right->container());
		}
		const void multiplication(const T right, const bool reverse=false) {
			_container->multiplication(right, reverse);
		}
		const void division(const MatrixImplPtr<C, T> right) {
			_container->division(right->container());
		}
		const void division(const T right, const bool reverse=false) {
			_container->division(right, reverse);
		}
		const void absolute(const MatrixImplPtr<C, T> right) {
			_container->absolute(right->container());
		}
		const void grayscale(const MatrixImplPtr<C, T> right) {
			_container->grayscale(right->container());
		}
		const void equalize(const MatrixImplPtr<C, T> right) {
			_container->equalize(right->container());
		}
		const T min() const {
			return _container->Cpu<T>::min();
		}
		const T max() const {
			return _container->Cpu<T>::max();
		}
		const void min(const Shape& shape) {
			_container->min(shape);
		}
		const void max(const Shape& shape) {
			_container->max(shape);
		}
		const PixelPtr<T> mins() const {
			return _container->mins();
		}
		const PixelPtr<T> maxs() const {
			return _container->maxs();
		}
        const Shape shape() {
            return _container->shape();
        }
        const void* matrix() {
            return _container->matrixp();
        }
		const ContainerPtr<C> container() const {
            return _container;
        }
        static const void initialize(const bool forced=false) {
            C::initialize();
        }
        static const void terminate(const bool forced=false) {
            C::terminate();
        }

	private:
		ContainerPtr<C> _container;
};

template<class C, typename T>
Matrix<C, T>::Matrix(const Shape& dimension, const T initialization, const bool random)
{
	_impl = tinycv::make_shared<MatrixImpl<C, T>>(dimension, initialization, random);
}

template<class C, typename T>
Matrix<C, T>::Matrix(const Shape& dimension, const PixelPtr<T> initialization)
{
	_impl = tinycv::make_shared<MatrixImpl<C, T>>(dimension, initialization);
}

template<class C, typename T>
Matrix<C, T>::Matrix(const Matrix<C, T>& matrix, bool deepcopy)
{
	_impl = tinycv::make_shared<MatrixImpl<C, T>>(matrix.impl(), deepcopy);
}

template<class C, typename T>
Matrix<C, T>::Matrix(const std::string& name, const Adjust<T>& adjust) 
{
	_impl = tinycv::make_shared<MatrixImpl<C, T>>(name, adjust);
}

template<class C, typename T>
Matrix<C, T>::Matrix(const void* matrix)
{
	_impl = tinycv::make_shared<MatrixImpl<C, T>>(matrix);
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::zeros(const Shape& dimension)
{
	return tinycv::make_shared<Matrix<C, T>>(dimension, 0.0);
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::ones(const Shape& dimension)
{
	return tinycv::make_shared<Matrix<C, T>>(dimension, 1.0);
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::rand(const Shape& dimension)
{
	return tinycv::make_shared<Matrix<C, T>>(dimension, 1.0, true);
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::reshape(const Shape& dimension)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*this);
    target->impl()->reshape(dimension);
	return target;	
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::resize(const Adjust<T>& adjust, const bool rate)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*this);
	target->impl()->resize(adjust, rate);
	return target;	
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::transpose(const Adjust<T>& adjust)
{
    MatrixPtr<C, T> target = std::make_shared<Matrix<C, T>>(*this);
    target->impl()->transpose(adjust);
    return target;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::operator()(const Shape& scope)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*this, false);
	target->impl()->range(scope);
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::operator()(const uint y, const uint x)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*this, false);
	target->impl()->range({y, x, 1, 1});
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::operator[](const uint y)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*this, false);
	target->impl()->range({y, 0, 1, 1});
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::fill(const Shape& dimension, const MatrixPtr<C, T> right)
{
	PixelPtr<T> pixel = right->template at<T, 3>(0, 0);
	pixel->resize(dimension[2]);
	return tinycv::make_shared<Matrix<C, T>>(dimension, pixel);
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::cumsum(const uint axis)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*this, false);
	target->impl()->cumsum(axis);
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::repmat(const Shape& dimension)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*this, false);
	target->impl()->repmat(dimension);
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::argsort(const uint axis, const bool reverse)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*this, false);
	target->impl()->argsort(axis, reverse);
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::load(const std::string& name, const Adjust<T>& adjust)
{
	return tinycv::make_shared<Matrix<C, T>>(name, adjust);	
}

template<class C, typename T>
const void Matrix<C, T>::save(const std::string& name, const Adjust<T>& adjust, const bool textlog)
{
	impl()->save(name, adjust);
}

template<class C, typename T> template<typename R, int D>
const PixelPtr<R> Matrix<C, T>::at(const uint y, const uint x) const
{
	return _impl->template at<R, D>(y, x);
}

template<class C, typename T>
const void Matrix<C, T>::operator=(const Pixel<T>& right)
{
	_impl->substitution(right);	
}

template<class C, typename T>
const void Matrix<C, T>::operator=(const MatrixPtr<C, T> right)
{
	_impl->substitution(right->impl());
}

template<class C, typename T>
const T Matrix<C, T>::min(const MatrixPtr<C, T> right)
{
	return right->impl()->min();
}

template<class C, typename T>
const T Matrix<C, T>::max(const MatrixPtr<C, T> right)
{
	return right->impl()->max();
}

template<class C, typename T>
const MatrixPtr<C, T> Matrix<C, T>::min(const MatrixPtr<C, T> right, const Shape& shape)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
	target->impl()->min(shape);
    return target;
}

template<class C, typename T>
const MatrixPtr<C, T> Matrix<C, T>::max(const MatrixPtr<C, T> right, const Shape& shape)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
	target->impl()->max(shape);
    return target;
}

template<class C, typename T>
const MatrixPtr<C, T> Matrix<C, T>::grayscale(const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
	target->impl()->grayscale(right->impl());
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::equalize(const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
	target->impl()->equalize(right->impl());
	return target;
}

template<class C, typename T>
const MatrixPtr<C, T> Matrix<C, T>::mins(const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> result = Matrix<C, T>::zeros({1, 1, 3});
	(*result) = (*right->impl()->mins());
	return result;
}

template<class C, typename T>
const MatrixPtr<C, T> Matrix<C, T>::maxs(const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> result = Matrix<C, T>::zeros({1, 1, 3});
	(*result) = (*right->impl()->maxs());
	return result;
}

template<class C, typename T>
MatrixPtr<C, T> Matrix<C, T>::absolute(const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
	target->impl()->absolute(right->impl());	
	return target;
}

template<class C, typename T>
const Shape Matrix<C, T>::shape() {
    return impl()->shape();
}

template<class C, typename T>
const void* Matrix<C, T>::matrix() {
    return impl()->matrix();
}

template<class C, typename T>
const void Matrix<C, T>::initialize(const bool forced)
{
	MatrixImpl<C, T>::initialize();
}

template<class C, typename T>
const void Matrix<C, T>::terminate(const bool forced)
{
    Matrix<C, T>::terminate();
}

#ifdef BUILD_CPU
template class Cpu<real>;
template class Matrix<Cpu<real>, real>;
template class MatrixImpl<Cpu<real>, real>;
template const PixelPtr<uint> Matrix<Cpu<real>, real>::at<uint, 1>(const uint y, const uint x) const;
template const PixelPtr<real> Matrix<Cpu<real>, real>::at<real, 3>(const uint y, const uint x) const;
template const PixelPtr<real> Matrix<Cpu<real>, real>::at<real, 1>(const uint y, const uint x) const;
#endif

#ifdef BUILD_CUDA
template class Cuda<real>;
template class Matrix<Cuda<real>, real>;
template class MatrixImpl<Cuda<real>, real>;
template const PixelPtr<uint> Matrix<Cuda<real>, real>::at<uint, 1>(const uint y, const uint x) const;
template const PixelPtr<real> Matrix<Cuda<real>, real>::at<real, 3>(const uint y, const uint x) const;
template const PixelPtr<real> Matrix<Cuda<real>, real>::at<real, 1>(const uint y, const uint x) const;
#endif

#ifdef BUILD_X5
template class X5<real>;
template class Matrix<X5<real>, real>;
template class MatrixImpl<X5<real>, real>;
template const PixelPtr<uint> Matrix<X5<real>, real>::at<uint, 1>(const uint y, const uint x) const;
template const PixelPtr<real> Matrix<X5<real>, real>::at<real, 3>(const uint y, const uint x) const;
template const PixelPtr<real> Matrix<X5<real>, real>::at<real, 1>(const uint y, const uint x) const;
#endif

#ifdef BUILD_CL
template class Cl<real>;
template class Matrix<Cl<real>, real>;
template class MatrixImpl<Cl<real>, real>;
template const PixelPtr<uint> Matrix<Cl<real>, real>::at<uint, 1>(const uint y, const uint x) const;
template const PixelPtr<real> Matrix<Cl<real>, real>::at<real, 3>(const uint y, const uint x) const;
template const PixelPtr<real> Matrix<Cl<real>, real>::at<real, 1>(const uint y, const uint x) const;
#endif

template<class C, typename T>
MatrixPtr<C, T> operator+(const T left, const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
    target->impl()->addition(left, true);	
	return target;	
}

template<class C, typename T>
MatrixPtr<C, T> operator-(const T left, const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
	target->impl()->subtraction(left, true);
	return target;	
}

template<class C, typename T>
MatrixPtr<C, T> operator*(const T left, const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
    target->impl()->multiplication(left, true);	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator/(const T left, const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*right);
    target->impl()->multiplication(left, true);	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator+(const MatrixPtr<C, T> left, const T right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*left);
    target->impl()->addition(right);	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator-(const MatrixPtr<C, T> left, const T right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*left);
    target->impl()->subtraction(right);	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator*(const MatrixPtr<C, T> left, const T right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*left);
    target->impl()->multiplication(right);	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator/(const MatrixPtr<C, T> left, const T right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*left);
    target->impl()->division(right);	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator+(const MatrixPtr<C, T> left, const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*left);
    target->impl()->addition(right->impl());	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator-(const MatrixPtr<C, T> left, const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*left);
    target->impl()->subtraction(right->impl());	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator*(const MatrixPtr<C, T> left, const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*left);
    target->impl()->multiplication(right->impl());	
	return target;
}

template<class C, typename T>
MatrixPtr<C, T> operator/(const MatrixPtr<C, T> left, const MatrixPtr<C, T> right)
{
	MatrixPtr<C, T> target = tinycv::make_shared<Matrix<C, T>>(*left);
    target->impl()->division(right->impl());	
	return target;
}

#ifdef BUILD_CPU
template MatrixPtr<Cpu<real>, real> operator+(const real left, const MatrixPtr<Cpu<real>, real> right);
template MatrixPtr<Cpu<real>, real> operator-(const real left, const MatrixPtr<Cpu<real>, real> right);
template MatrixPtr<Cpu<real>, real> operator*(const real left, const MatrixPtr<Cpu<real>, real> right);
template MatrixPtr<Cpu<real>, real> operator/(const real left, const MatrixPtr<Cpu<real>, real> right);
template MatrixPtr<Cpu<real>, real> operator+(const MatrixPtr<Cpu<real>, real> right, const real left);
template MatrixPtr<Cpu<real>, real> operator-(const MatrixPtr<Cpu<real>, real> right, const real left);
template MatrixPtr<Cpu<real>, real> operator*(const MatrixPtr<Cpu<real>, real> right, const real left);
template MatrixPtr<Cpu<real>, real> operator/(const MatrixPtr<Cpu<real>, real> right, const real left);
template MatrixPtr<Cpu<real>, real> operator+(const MatrixPtr<Cpu<real>, real> left, const MatrixPtr<Cpu<real>, real> right);
template MatrixPtr<Cpu<real>, real> operator-(const MatrixPtr<Cpu<real>, real> left, const MatrixPtr<Cpu<real>, real> right);
template MatrixPtr<Cpu<real>, real> operator*(const MatrixPtr<Cpu<real>, real> left, const MatrixPtr<Cpu<real>, real> right);
template MatrixPtr<Cpu<real>, real> operator/(const MatrixPtr<Cpu<real>, real> left, const MatrixPtr<Cpu<real>, real> right);
#endif

#ifdef BUILD_CUDA
template MatrixPtr<Cuda<real>, real> operator+(const real left, const MatrixPtr<Cuda<real>, real> right);
template MatrixPtr<Cuda<real>, real> operator-(const real left, const MatrixPtr<Cuda<real>, real> right);
template MatrixPtr<Cuda<real>, real> operator*(const real left, const MatrixPtr<Cuda<real>, real> right);
template MatrixPtr<Cuda<real>, real> operator/(const real left, const MatrixPtr<Cuda<real>, real> right);
template MatrixPtr<Cuda<real>, real> operator+(const MatrixPtr<Cuda<real>, real> right, const real left);
template MatrixPtr<Cuda<real>, real> operator-(const MatrixPtr<Cuda<real>, real> right, const real left);
template MatrixPtr<Cuda<real>, real> operator*(const MatrixPtr<Cuda<real>, real> right, const real left);
template MatrixPtr<Cuda<real>, real> operator/(const MatrixPtr<Cuda<real>, real> right, const real left);
template MatrixPtr<Cuda<real>, real> operator+(const MatrixPtr<Cuda<real>, real> left, const MatrixPtr<Cuda<real>, real> right);
template MatrixPtr<Cuda<real>, real> operator-(const MatrixPtr<Cuda<real>, real> left, const MatrixPtr<Cuda<real>, real> right);
template MatrixPtr<Cuda<real>, real> operator*(const MatrixPtr<Cuda<real>, real> left, const MatrixPtr<Cuda<real>, real> right);
template MatrixPtr<Cuda<real>, real> operator/(const MatrixPtr<Cuda<real>, real> left, const MatrixPtr<Cuda<real>, real> right);
#endif

#ifdef BUILD_X5
template MatrixPtr<X5<real>, real> operator+(const real left, const MatrixPtr<X5<real>, real> right);
template MatrixPtr<X5<real>, real> operator-(const real left, const MatrixPtr<X5<real>, real> right);
template MatrixPtr<X5<real>, real> operator*(const real left, const MatrixPtr<X5<real>, real> right);
template MatrixPtr<X5<real>, real> operator/(const real left, const MatrixPtr<X5<real>, real> right);
template MatrixPtr<X5<real>, real> operator+(const MatrixPtr<X5<real>, real> right, const real left);
template MatrixPtr<X5<real>, real> operator-(const MatrixPtr<X5<real>, real> right, const real left);
template MatrixPtr<X5<real>, real> operator*(const MatrixPtr<X5<real>, real> right, const real left);
template MatrixPtr<X5<real>, real> operator/(const MatrixPtr<X5<real>, real> right, const real left);
template MatrixPtr<X5<real>, real> operator+(const MatrixPtr<X5<real>, real> left, const MatrixPtr<X5<real>, real> right);
template MatrixPtr<X5<real>, real> operator-(const MatrixPtr<X5<real>, real> left, const MatrixPtr<X5<real>, real> right);
template MatrixPtr<X5<real>, real> operator*(const MatrixPtr<X5<real>, real> left, const MatrixPtr<X5<real>, real> right);
template MatrixPtr<X5<real>, real> operator/(const MatrixPtr<X5<real>, real> left, const MatrixPtr<X5<real>, real> right);
#endif

#ifdef BUILD_CL
template MatrixPtr<Cl<real>, real> operator+(const real left, const MatrixPtr<Cl<real>, real> right);
template MatrixPtr<Cl<real>, real> operator-(const real left, const MatrixPtr<Cl<real>, real> right);
template MatrixPtr<Cl<real>, real> operator*(const real left, const MatrixPtr<Cl<real>, real> right);
template MatrixPtr<Cl<real>, real> operator/(const real left, const MatrixPtr<Cl<real>, real> right);
template MatrixPtr<Cl<real>, real> operator+(const MatrixPtr<Cl<real>, real> right, const real left);
template MatrixPtr<Cl<real>, real> operator-(const MatrixPtr<Cl<real>, real> right, const real left);
template MatrixPtr<Cl<real>, real> operator*(const MatrixPtr<Cl<real>, real> right, const real left);
template MatrixPtr<Cl<real>, real> operator/(const MatrixPtr<Cl<real>, real> right, const real left);
template MatrixPtr<Cl<real>, real> operator+(const MatrixPtr<Cl<real>, real> left, const MatrixPtr<Cl<real>, real> right);
template MatrixPtr<Cl<real>, real> operator-(const MatrixPtr<Cl<real>, real> left, const MatrixPtr<Cl<real>, real> right);
template MatrixPtr<Cl<real>, real> operator*(const MatrixPtr<Cl<real>, real> left, const MatrixPtr<Cl<real>, real> right);
template MatrixPtr<Cl<real>, real> operator/(const MatrixPtr<Cl<real>, real> left, const MatrixPtr<Cl<real>, real> right);
#endif

}; // end of namespace


