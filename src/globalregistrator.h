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

#ifndef GLOBALREGISTRATOR_H
#define GLOBALREGISTRATOR_H

#include "imageops.h"
#include "registrationparams.h"
#include "registrationcontext.h"

// globalRegistrator is a tool that initially takes a template image and can
// then sequentially register any number of images against that template. The
// class should not be used directly because of its non-thread-safeness, but
// rather via the static method getGlobalShifts() that performs a parallelized
// registration of multiple images.
class globalRegistrator {
public:
  globalRegistrator(const cv::Mat& reference, const int maxmove);

private:
  // Not thread safe.
  // Each thread needs to have its own instance of globalRegistrator in order
  // to call this method. See getGlobalShifts().
  void findShift(inputImage& image, const cv::Mat& pixels);

  cv::Mat1f refImgWithBorder;
  cv::Mat1f refImageArea;
  cv::Mat1f searchMask;
  cv::Mat1f areasq;
  cv::Mat1f imgsq;
  cv::Mat1f cor;
  cv::Mat1f match;
  cv::Point originShift;

public:
  // Parallelized static method that registers a set of images and stores
  // the results into the registration context.
  static void getGlobalShifts(const registrationParams& params,
                              registrationContext& context,
                              const cv::Mat& refimg,
                              bool showProgress);
};

#endif // GLOBALREGISTRATOR_H
