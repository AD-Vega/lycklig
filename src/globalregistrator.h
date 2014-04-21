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

class globalRegistrator {
public:
  globalRegistrator(const cv::Mat& reference, const int maxmove);

  // NOT THREAD SAFE!
  cv::Point findShift(const cv::Mat& img);

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

std::vector<cv::Point> getGlobalShifts(const cv::Mat& refimg,
                                       const registrationParams& params,
                                       bool showProgress);

cv::Rect optimalCrop(std::vector<cv::Point> shifts, cv::Size size);

#endif // GLOBALREGISTRATOR_H
