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

#ifndef LUCKY_H
#define LUCKY_H

#include <opencv2/core/core.hpp>
#include "registrationparams.h"
#include "registrationcontext.h"
#include "rbfwarper.h"

class patchMatcher {
public:
  cv::Mat1f match(const cv::Mat1f& img,
                  const imagePatch& patch,
                  const float multiplier);

private:
  cv::Mat1f areasq;
  cv::Mat1f cor;
};


class quadraticFit {
public:
  quadraticFit(const cv::Mat& data, const cv::Point& point);
  cv::Point2f minimum() const;
  // The smaller of the two eigenvalues.
  float smallerEig() const;
  // The larger of the two eigenvalues.
  float largerEig() const;
  // Eigenvector corresponding to the smaller eigenvalue.
  cv::Point2f smallerEigVec() const;
  // Eigenvector corresponding to the larger eigenvalue.
  cv::Point2f largerEigVec() const;

private:
  // A matrix of x^2, x*y and y^2 for the quadratic fit.
  static cv::Mat1f fitx;
  // Location of the minimum of the best matching quadratic function.
  cv::Mat x0y0;
  // The Hessian (divided by two).
  cv::Mat H;
  cv::Mat eigenvalues;
  cv::Mat eigenvectors;
};


patchCollection selectPointsHex(const registrationParams& params,
                                const registrationContext& context);

patchCollection filterPatchesByQuality(const patchCollection& patches,
                                       const cv::Mat& refimg);

cv::Mat drawPoints(const cv::Mat& img, const patchCollection& patches);

cv::Mat lucky(const registrationParams& params,
              registrationContext& context,
              const bool showProgress = false);

#endif // LUCKY_H
