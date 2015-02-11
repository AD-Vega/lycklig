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

#ifndef RBFWARPER_H
#define RBFWARPER_H

#include <utility>
#include "imagepatch.h"

class rbfWarper {
public:
  rbfWarper(const patchCollection& patches,
            const cv::Size inputImageSize,
            const cv::Rect& targetRect,
            const float sigma,
            const int supersampling = 1);

  std::pair<cv::Mat, cv::Mat>
    warp(const cv::Mat& image,
         const cv::Point& globalShift,
         const cv::Mat1f& shifts) const;

private:
  void gauss1d(float* ptr, const cv::Range& range, const float sigma) const;
  void prepareBases();

private:
  const patchCollection& patches;
  const cv::Point targetOrigin;
  const cv::Size imagesize;
  const float sigma;
  const int supersampling;
  const cv::Mat1f normalizationMask;
  cv::Mat1f coeffs;
  cv::Mat1f xshiftbase;
  cv::Mat1f yshiftbase;
  cv::Rect basesRect;
  cv::Mat gaussianKernel;
};

#endif // RBFWARPER_H
