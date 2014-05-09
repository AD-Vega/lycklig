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

#ifndef REGISTRATIONCONTEXT_H
#define REGISTRATIONCONTEXT_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include "imagepatch.h"

class inputImage {
public:
  inputImage(std::string filename_) :
    filename(filename_), globalShift(0, 0)
    {};

  std::string filename;
  cv::Point globalShift;
};

class registrationContext {
public:
  std::vector<inputImage> images;
  cv::Rect crop = cv::Rect(0, 0, 0, 0);
  cv::Mat refimg;
  std::vector<imagePatch> patches;
};

#endif // REGISTRATIONCONTEXT_H
