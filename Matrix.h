/*!
 * @author Hisashi Ikari
 */

#ifndef TINYCV_BASE_MATRIX_H
#define TINYCV_BASE_MATRIX_H

#include <vector>
#include <string>

#include "Def.h"
#include "Memory.h"

namespace tinycv {

typedef std::vector<uint> Shape;
typedef std::shared_ptr<Shape> ShapePtr;	

template<typename T> using Pixel = std::vector<T>;
template<typename T> using Adjust = std::vector<T>;
template<typename T> using PixelPtr = std::shared_ptr<Pixel<T>>;

template<class C, typename T> class Matrix;
template<class C, typename T> class MatrixImpl;
template<class C, typename T> using MatrixPtr = std::shared_ptr<Matrix<C, T>>;
template<class C, typename T> using MatrixImplPtr = std::shared_ptr<MatrixImpl<C, T>>;

template<class C, typename T>
class Matrix
{
	public:
		Matrix(const Shape& dimension, const T initialization=0.0, const bool random=false);
		Matrix(const Shape& dimension, const PixelPtr<T> initialization);
		Matrix(const std::string& name, const Adjust<T>& adjust={3, 1});
		Matrix(const Matrix<C, T>& matrix, bool deepcopy=true);
		Matrix(const void* matrix);

		static MatrixPtr<C, T> zeros(const Shape& dimension);
		static MatrixPtr<C, T> ones(const Shape& dimension);
		static MatrixPtr<C, T> rand(const Shape& dimension);
		MatrixPtr<C, T> reshape(const Shape& dimension);
        MatrixPtr<C, T> resize(const Adjust<T>& adjust, const bool rate=true);
		MatrixPtr<C, T> repmat(const Shape& dimension);
        MatrixPtr<C, T> transpose(const Adjust<T>& adjust);
		MatrixPtr<C, T> argsort(const uint axis=AXIS_Y, const bool reverse=false);
		MatrixPtr<C, T> cumsum(const uint axis=AXIS_Y);

		static MatrixPtr<C, T> load(const std::string& name, const Adjust<T>& adjust={3, 1});
		const void save(const std::string& name, const Adjust<T>& adjust={3, 1}, const bool textlog=false);

		static const T min(const MatrixPtr<C, T> right);
		static const T max(const MatrixPtr<C, T> right);
		static const MatrixPtr<C, T> min(const MatrixPtr<C, T> right, const Shape& shape);
		static const MatrixPtr<C, T> max(const MatrixPtr<C, T> right, const Shape& shape);
		static const MatrixPtr<C, T> mins(const MatrixPtr<C, T> right);
		static const MatrixPtr<C, T> maxs(const MatrixPtr<C, T> right);
		static MatrixPtr<C, T> fill(const Shape& dimension, const MatrixPtr<C, T> right);
		static MatrixPtr<C, T> absolute(const MatrixPtr<C, T> right);
        static MatrixPtr<C, T> equalize(const MatrixPtr<C, T> right);
		static const MatrixPtr<C, T> grayscale(const MatrixPtr<C, T> right);

        static const void initialize(const bool forced=false);
        static const void terminate(const bool forced=false);
 
		template<typename R, int D>
		const PixelPtr<R> at(const uint y, const uint x) const;
		MatrixPtr<C, T> operator()(const uint y, const uint x);
		MatrixPtr<C, T> operator()(const Shape& shape);
		MatrixPtr<C, T> operator[](const uint y);

		const void operator=(const Pixel<T>& right);
		const void operator=(const MatrixPtr<C, T> right);

        const Shape shape();
        const void* matrix();
		const MatrixImplPtr<C, T> impl() const { return _impl; }

	private:
		MatrixImplPtr<C, T> _impl;
};

#ifdef BUILD_CPU
template<typename T> class Cpu;
#endif
#ifdef BUILD_CUDA
template<typename T> class Cuda;
#endif
#ifdef BUILD_X5
template<typename T> class X5;
#endif
#ifdef BUILD_CL
template<typename T> class Cl;
#endif

}; // end of namespace

#endif


