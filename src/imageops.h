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
#include "imagepatch.h"

cv::Mat meanimg(const std::vector<std::string>& files, bool showProgress = false);

std::vector<imagePatch> selectPointsHex(const cv::Mat img,
                                        const unsigned int boxsize,
                                        const unsigned int xydiff,
                                        const double val_threshold,
                                        const double surf_threshold);

cv::Mat drawPoints(const cv::Mat& img, const std::vector<imagePatch>& patches);

std::vector<cv::Rect> createSearchAreas(const std::vector<imagePatch>& patches,
                                        const cv::Size& imagesize,
                                        const int maxmove);

cv::Mat1f findShifts(const cv::Mat& img,
                     const std::vector<imagePatch>& patches,
                     const std::vector<cv::Rect>& areas);

#endif // IMAGEOPS_H
