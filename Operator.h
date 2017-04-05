/*!
 * @author Hisashi Ikari
 */

#ifndef TINYCV_BASE_OPERATOR_H
#define TINYCV_BASE_OPERATOR_H

#include "Memory.h"

namespace tinycv {

template<class C, typename T> class Matrix;
template<class C, typename T> using MatrixPtr = std::shared_ptr<Matrix<C, T>>;

template<class C, typename T>
MatrixPtr<C, T> operator+(const T left, const MatrixPtr<C, T> right);

template<class C, typename T>
MatrixPtr<C, T> operator-(const T left, const MatrixPtr<C, T> right);

template<class C, typename T>
MatrixPtr<C, T> operator*(const T left, const MatrixPtr<C, T> right);

template<class C, typename T>
MatrixPtr<C, T> operator/(const T left, const MatrixPtr<C, T> right);

template<class C, typename T>
MatrixPtr<C, T> operator+(const MatrixPtr<C, T> left, const T right);

template<class C, typename T>
MatrixPtr<C, T> operator-(const MatrixPtr<C, T> left, const T right);

template<class C, typename T>
MatrixPtr<C, T> operator*(const MatrixPtr<C, T> left, const T right);

template<class C, typename T>
MatrixPtr<C, T> operator/(const MatrixPtr<C, T> left, const T right);

template<class C, typename T>
MatrixPtr<C, T> operator+(const MatrixPtr<C, T> left, const MatrixPtr<C, T> right);

template<class C, typename T>
MatrixPtr<C, T> operator-(const MatrixPtr<C, T> left, const MatrixPtr<C, T> right);

template<class C, typename T>
MatrixPtr<C, T> operator*(const MatrixPtr<C, T> left, const MatrixPtr<C, T> right);

template<class C, typename T>
MatrixPtr<C, T> operator/(const MatrixPtr<C, T> left, const MatrixPtr<C, T> right);

}; // end of namespace

#endif

