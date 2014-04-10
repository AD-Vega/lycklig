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
#include <Magick++.h>
#include "imagepatch.h"
#include "registrationparams.h"
#include "rbfwarper.h"


class grayReader {
public:
  // NOT THREAD SAFE!
  cv::Mat read(std::string file);

private:
  cv::Mat imgcolor;
  cv::Mat1f imggray;
};

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
  cv::Mat1f cor;
  cv::Mat1f weight;
  cv::Mat1f match;
  cv::Point originShift;
};

cv::Mat magickImread(const std::string& filename);

cv::Mat meanimg(const std::vector<std::string>& files,
            cv::Rect crop,
            std::vector<cv::Point> shifts,
            bool showProgress = false);

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

std::vector<cv::Point> getGlobalShifts(const std::vector<std::string>& files,
                                       const cv::Mat& refimg,
                                       unsigned int maxmove,
                                       bool showProgress = false);

cv::Rect optimalCrop(std::vector<cv::Point> shifts, cv::Size size);

cv::Mat3f lucky(registrationParams params,
                cv::Mat refimg,
                cv::Rect crop,
                std::vector<cv::Point> globalShifts,
                std::vector<imagePatch> patches,
                std::vector<cv::Rect> areas,
                rbfWarper rbf);

cv::Mat3w normalizeTo16Bits(const cv::Mat& inputImg);

#endif // IMAGEOPS_H
