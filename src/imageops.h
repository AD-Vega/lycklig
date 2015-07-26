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

#ifndef IMAGEOPS_H
#define IMAGEOPS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "registrationparams.h"
#include "registrationcontext.h"

class grayReader {
public:
  // NOT THREAD SAFE!
  cv::Mat read(const std::string& file);

private:
  cv::Mat1f imggray;
};

cv::Mat magickImread(const std::string& filename);

void writeTestImage(const std::string& path);

void sRGB2linearRGB(cv::Mat& img);

void linearRGB2sRGB(cv::Mat& img);

void divideChannelsByMask(cv::Mat& image, cv::Mat& mask);

cv::Mat meanimg(const registrationContext& context,
                const bool showProgress = false);

cv::Mat normalizeTo16Bits(const cv::Mat& inputImg);

class imageSumLookup
{
public:
  imageSumLookup() = default;
  imageSumLookup(const cv::Mat& img);
  float lookup(const cv::Rect rect) const;

private:
  cv::Mat table;
};

#endif // IMAGEOPS_H
