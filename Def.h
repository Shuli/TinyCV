/*!
 * @author Hisashi Ikari
 */

#ifndef TINYCV_BASE_DEF_H
#define TINYCV_BASE_DEF_H

#include "Config.h"

namespace tinycv {
	typedef unsigned int uint;
	typedef unsigned char uchar;
	typedef float real;

    enum TCV_DIMS {DIMS_Y = 0, DIMS_X = 1, DIMS_CHANNEL = 3};
    enum TCV_RECT {RECT_Y = 0, RECT_X = 1, RECT_HEIGHT = 2, RECT_WIDTH = 3};
    enum TCV_AXIS {AXIS_Y = 0, AXIS_X = 1};
};

#endif
