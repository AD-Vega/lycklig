/*
 *    lycklig, image processing for lucky imaging.
 *    Copyright (C) 2013, 2014 Andrej Lajovic <andrej.lajovic@ad-vega.si>
 *
 *    Code in this file is based on code taken from the OpenCV library,
 *    Copyright (C) 2000, Intel Corporation
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cookedtemplate.h"

using namespace cv;


cookedXcor::cookedXcor(const Mat& _templ, Size _corrsize, int _ctype) :
  ctype(_ctype), maxDepth(CV_64F), corrsize(_corrsize)
{
    const double blockScale = 4.5;
    const int minBlockSize = 256;
    std::vector<uchar> buf;

    Mat templ = _templ;
    tdepth = templ.depth();
    tcn = templ.channels();
    templsize = templ.size();

    CV_Assert( templ.dims <= 2 );
    CV_Assert( tdepth == CV_32F );

    blocksize.width = cvRound(templ.cols*blockScale);
    blocksize.width = std::max( blocksize.width, minBlockSize - templ.cols + 1 );
    blocksize.width = std::min( blocksize.width, corrsize.width );
    blocksize.height = cvRound(templ.rows*blockScale);
    blocksize.height = std::max( blocksize.height, minBlockSize - templ.rows + 1 );
    blocksize.height = std::min( blocksize.height, corrsize.height );

    dftsize.width = std::max(getOptimalDFTSize(blocksize.width + templ.cols - 1), 2);
    dftsize.height = getOptimalDFTSize(blocksize.height + templ.rows - 1);
    if( dftsize.width <= 0 || dftsize.height <= 0 )
        CV_Error( CV_StsOutOfRange, "the input arrays are too big" );

    // recompute block size
    blocksize.width = dftsize.width - templ.cols + 1;
    blocksize.width = MIN( blocksize.width, corrsize.width );
    blocksize.height = dftsize.height - templ.rows + 1;
    blocksize.height = MIN( blocksize.height, corrsize.height );

    dftTempl.create(dftsize.height*tcn, dftsize.width, maxDepth);

    int bufSize = 0;
    if( tcn > 1 && tdepth != maxDepth )
        bufSize = templ.cols*templ.rows*CV_ELEM_SIZE(tdepth);

    buf.resize(bufSize);

    // compute DFT of each template plane
    for(int k = 0; k < tcn; k++ )
    {
        int yofs = k*dftsize.height;
        Mat src = templ;
        Mat dst(dftTempl, Rect(0, yofs, dftsize.width, dftsize.height));
        Mat dst1(dftTempl, Rect(0, yofs, templ.cols, templ.rows));

        if( tcn > 1 )
        {
            src = tdepth == maxDepth ? dst1 : Mat(templ.size(), tdepth, &buf[0]);
            int pairs[] = {k, 0};
            mixChannels(&templ, 1, &src, 1, pairs, 1);
        }

        if( dst1.data != src.data )
            src.convertTo(dst1, dst1.depth());

        if( dst.cols > templ.cols )
        {
            Mat part(dst, Range(0, templ.rows), Range(templ.cols, dst.cols));
            part = Scalar::all(0);
        }
        dft(dst, dst, 0, templ.rows);
    }
}


void cookedXcor::xcor(const Mat& img, Mat& corr)
{
    Point anchor(0, 0);
    const double delta = 0;
    int borderType = 0;
    std::vector<uchar> buf;

    int depth = img.depth(), cn = img.channels();
    int cdepth = CV_MAT_DEPTH(ctype), ccn = CV_MAT_CN(ctype);

    CV_Assert( img.dims <= 2 && corr.dims <= 2 );
    CV_Assert(depth == CV_32F);
    CV_Assert(cdepth == CV_32F);
    CV_Assert( corrsize.height <= img.rows + templsize.height - 1 &&
               corrsize.width <= img.cols + templsize.width - 1 );
    CV_Assert( ccn == 1 || delta == 0 );

    corr.create(corrsize, ctype);

    Mat dftImg( dftsize, maxDepth );

    int bufSize = 0;
    if( tcn > 1 && tdepth != maxDepth )
        bufSize = templsize.width*templsize.height*CV_ELEM_SIZE(tdepth);

    if( cn > 1 && depth != maxDepth )
        bufSize = std::max( bufSize, (blocksize.width + templsize.width - 1)*
            (blocksize.height + templsize.height - 1)*CV_ELEM_SIZE(depth));

    if( (ccn > 1 || cn > 1) && cdepth != maxDepth )
        bufSize = std::max( bufSize, blocksize.width*blocksize.height*CV_ELEM_SIZE(cdepth));

    buf.resize(bufSize);

    int tileCountX = (corr.cols + blocksize.width - 1)/blocksize.width;
    int tileCountY = (corr.rows + blocksize.height - 1)/blocksize.height;
    int tileCount = tileCountX * tileCountY;

    Size wholeSize = img.size();
    Point roiofs(0,0);
    Mat img0 = img;

    if( !(borderType & BORDER_ISOLATED) )
    {
        img.locateROI(wholeSize, roiofs);
        img0.adjustROI(roiofs.y, wholeSize.height-img.rows-roiofs.y,
                       roiofs.x, wholeSize.width-img.cols-roiofs.x);
    }
    borderType |= BORDER_ISOLATED;

    // calculate correlation by blocks
    for(int i = 0; i < tileCount; i++ )
    {
        int x = (i%tileCountX)*blocksize.width;
        int y = (i/tileCountX)*blocksize.height;

        Size bsz(std::min(blocksize.width, corr.cols - x),
                 std::min(blocksize.height, corr.rows - y));
        Size dsz(bsz.width + templsize.width - 1, bsz.height + templsize.height - 1);
        int x0 = x - anchor.x + roiofs.x, y0 = y - anchor.y + roiofs.y;
        int x1 = std::max(0, x0), y1 = std::max(0, y0);
        int x2 = std::min(img0.cols, x0 + dsz.width);
        int y2 = std::min(img0.rows, y0 + dsz.height);
        Mat src0(img0, Range(y1, y2), Range(x1, x2));
        Mat dst(dftImg, Rect(0, 0, dsz.width, dsz.height));
        Mat dst1(dftImg, Rect(x1-x0, y1-y0, x2-x1, y2-y1));
        Mat cdst(corr, Rect(x, y, bsz.width, bsz.height));

        for(int k = 0; k < cn; k++ )
        {
            Mat src = src0;
            dftImg = Scalar::all(0);

            if( cn > 1 )
            {
                src = depth == maxDepth ? dst1 : Mat(y2-y1, x2-x1, depth, &buf[0]);
                int pairs[] = {k, 0};
                mixChannels(&src0, 1, &src, 1, pairs, 1);
            }

            if( dst1.data != src.data )
                src.convertTo(dst1, dst1.depth());

            if( x2 - x1 < dsz.width || y2 - y1 < dsz.height )
                copyMakeBorder(dst1, dst, y1-y0, dst.rows-dst1.rows-(y1-y0),
                               x1-x0, dst.cols-dst1.cols-(x1-x0), borderType);

            dft( dftImg, dftImg, 0, dsz.height );
            Mat dftTempl1(dftTempl, Rect(0, tcn > 1 ? k*dftsize.height : 0,
                                         dftsize.width, dftsize.height));
            mulSpectrums(dftImg, dftTempl1, dftImg, 0, true);
            dft( dftImg, dftImg, DFT_INVERSE + DFT_SCALE, bsz.height );

            src = dftImg(Rect(0, 0, bsz.width, bsz.height));

            if( ccn > 1 )
            {
                if( cdepth != maxDepth )
                {
                    Mat plane(bsz, cdepth, &buf[0]);
                    src.convertTo(plane, cdepth, 1, delta);
                    src = plane;
                }
                int pairs[] = {0, k};
                mixChannels(&src, 1, &cdst, 1, pairs, 1);
            }
            else
            {
                if( k == 0 )
                    src.convertTo(cdst, cdepth, 1, delta);
                else
                {
                    if( maxDepth != cdepth )
                    {
                        Mat plane(bsz, cdepth, &buf[0]);
                        src.convertTo(plane, cdepth);
                        src = plane;
                    }
                    add(src, cdst, cdst);
                }
            }
        }
    }
}


cookedTemplate::cookedTemplate(InputArray _templ, Size searchSize)
{
  Mat templ = _templ.getMat();
  CV_Assert(templ.cols <= searchSize.width &&
            templ.rows <= searchSize.height);

  CV_Assert(templ.depth() == CV_32F);

  templType = templ.type();
  corrSize = Size(searchSize.width - templ.cols + 1, searchSize.height - templ.rows + 1);
  cxc = cookedXcor(templ, corrSize, CV_32F);
}


void cookedTemplate::match(InputArray _img, OutputArray _result)
{
  Mat img = _img.getMat();
  CV_Assert(img.depth() == CV_32F);
  CV_Assert(img.type() == templType);

  _result.create(corrSize, CV_32F);
  Mat result = _result.getMat();
  cxc.xcor(img, result);
}
