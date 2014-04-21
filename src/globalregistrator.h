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

class globalRegistrator;

class globalRegistration {
public:
  bool valid = false;
  std::vector<cv::Point> shifts;
  cv::Rect crop;

  void calculateOptimalCrop(const cv::Size& size);
};


class globalRegistrator {
public:
  globalRegistrator(const cv::Mat& reference, const int maxmove);

  // Parallelized static method that registers a set of images.
  static globalRegistration getGlobalShifts(const cv::Mat& refimg,
                                            const registrationParams& params,
                                            bool showProgress);

private:
  // Not thread safe.
  // Each thread needs to have its own instance of globalRegistrator in order
  // to call this method.
  cv::Point findShift(const cv::Mat& img);

friend globalRegistration getGlobalShifts(const cv::Mat& refimg,
                                          const registrationParams& params,
                                          bool showProgress);

private:
  cv::Mat1f refImgWithBorder;
  cv::Mat1f refImageArea;
  cv::Mat1f searchMask;
  cv::Mat1f areasq;
  cv::Mat1f imgsq;
  cv::Mat1f cor;
  cv::Mat1f match;
  cv::Point originShift;
};

#endif // GLOBALREGISTRATOR_H
