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

cv::Mat magickImread(const std::string& filename);

cv::Mat meanimg(const std::vector<std::string>& files,
            cv::Rect crop,
            std::vector<cv::Point> shifts,
            bool showProgress = false);

std::vector<imagePatch> selectPointsHex(const cv::Mat img,
                                        const registrationParams& params);

std::vector<imagePatch> filterPatchesByQuality(const std::vector<imagePatch> patches,
                                               const cv::Mat& refimg);

cv::Mat drawPoints(const cv::Mat& img, const std::vector<imagePatch>& patches);

cv::Mat1f findShifts(const cv::Mat& img,
                     const std::vector<imagePatch>& patches,
                     const std::vector<cv::Rect>& areas);

cv::Mat3f lucky(registrationParams params,
                cv::Mat refimg,
                cv::Rect crop,
                std::vector<cv::Point> globalShifts,
                std::vector<imagePatch> patches,
                rbfWarper rbf,
                bool showProgress = false);

cv::Mat3w normalizeTo16Bits(const cv::Mat& inputImg);

#endif // IMAGEOPS_H
