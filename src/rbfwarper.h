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

#include "imagepatch.h"

class rbfWarper {
public:
  rbfWarper(const std::vector<imagePatch>& patches,
	    const cv::Size& imagesize,
	    const float sigma);
  cv::Mat warp(const cv::Mat& image, const cv::Mat1f& shifts);

private:
  void gauss1d(float* ptr, const cv::Range& range, const float sigma);
  void prepareBases(const std::vector<imagePatch>& patches,
                    const cv::Size& imagesize,
                    const float sigma);
  void prepareCoeffs(const std::vector<imagePatch>& patches);

private:
  std::vector<cv::Mat1f> bases;
  cv::Mat1f coeffs;
  cv::Mat1f xshiftbase;
  cv::Mat1f yshiftbase;
};

#endif // RBFWARPER_H
