/*
 *    lycklig, image processing for lucky imaging.
 *    Copyright (C) 2013, 2014 Andrej Lajovic <andrej.lajovic@ad-vega.si>
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

#include "globalregistrator.h"

using namespace cv;

globalRegistrator::globalRegistrator(const Mat& reference, const int maxmove) {
  refImgWithBorder = Mat::zeros(reference.rows + 2*maxmove, reference.cols + 2*maxmove, CV_32F);
  Rect imageRect = Rect(maxmove, maxmove, reference.cols, reference.rows);
  reference.copyTo(refImgWithBorder(imageRect));
  refImageArea = Mat::zeros(reference.rows + 2*maxmove, reference.cols + 2*maxmove, CV_32F);
  refImageArea(imageRect) = Mat::ones(reference.rows, reference.cols, CV_32F);
  searchMask = Mat::ones(reference.rows, reference.cols, CV_32F);
  matchTemplate(refImgWithBorder.mul(refImgWithBorder), searchMask, areasq, CV_TM_CCORR);
  originShift = Point(maxmove, maxmove);
}


Point globalRegistrator::findShift(const Mat& img)
{
  matchTemplate(refImageArea, img.mul(img), imgsq, CV_TM_CCORR);
  matchTemplate(refImgWithBorder, img, cor, CV_TM_CCORR);
  match = areasq - cor.mul(cor).mul(1/imgsq);
  Point minpoint;
  minMaxLoc(match, NULL, NULL, &minpoint);
  return -(minpoint - originShift);
}


void globalRegistrator::getGlobalShifts(const registrationParams& params,
                                        registrationContext& context,
                                        const Mat& refimg,
                                        const bool showProgress) {
  int progress = 0;
  if (showProgress)
    std::fprintf(stderr, "0/%ld", context.images.size());
  #pragma omp parallel
  {
    grayReader reader;
    globalRegistrator globalReg(refimg, params.prereg_maxmove);
    #pragma omp for schedule(dynamic)
    for (int ifile = 0; ifile < (signed)context.images.size(); ifile++) {
      auto& image = context.images.at(ifile);
      Mat img(reader.read(image.filename));
      image.globalShift = globalReg.findShift(img);

      if (showProgress) {
        #pragma omp critical
        std::fprintf(stderr, "\r\033[K%d/%ld", ++progress, context.images.size());
      }
    }
  }
  if (showProgress)
    std::fprintf(stderr, "\n");

  context.crop = Rect(-context.images.at(0).globalShift, refimg.size());
  for (auto& image : context.images) {
        context.crop &= Rect(-image.globalShift, refimg.size());
  }
}
