/*!
 * @author Hisashi Ikari
 */
#ifndef TINYCV_BASE_SCREEN_H
#define TINYCV_BASE_SCREEN_H

#include "Def.h"

#if defined(BUILD_CPU) || defined(BUILD_CUDA) || defined(BUILD_X5) || defined(BUILD_CL)
#include <QGraphicsView>
#include <QtConcurrent>
#endif

#if defined(BUILD_CPU) || defined(BUILD_CUDA) || defined(BUILD_X5) || defined(BUILD_CL)
class QGraphicsViewDehazeImpl;
class QGraphicsViewDehaze : public QGraphicsView
{
    Q_OBJECT
    public slots:
        virtual void advance();

    public:
        explicit QGraphicsViewDehaze(QGraphicsScene *scene, QTimer* timer, QWidget* parent=Q_NULLPTR, 
            const char* type=NULL, const tinycv::real rate=0.25, const int step=10); 
        virtual ~QGraphicsViewDehaze();
        QGraphicsViewDehazeImpl* impl() { return _impl; }

    private:
        QGraphicsViewDehazeImpl* _impl; 
};
#endif

#endif

