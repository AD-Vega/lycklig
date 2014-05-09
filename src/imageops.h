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
#include "registrationcontext.h"
#include "rbfwarper.h"

class globalRegistration;

class grayReader {
public:
  // NOT THREAD SAFE!
  cv::Mat read(const std::string& file);

private:
  cv::Mat imgcolor;
  cv::Mat1f imggray;
};

class patchMatcher {
public:
  cv::Mat1f match(const cv::Mat1f& img,
                  const imagePatch& patch,
                  const float multiplier);

private:
  cv::Mat1f mask;
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

cv::Mat magickImread(const std::string& filename);

void sRGB2linearRGB(cv::Mat& img);

void linearRGB2sRGB(cv::Mat& img);

cv::Mat meanimg(const registrationParams& params,
                const registrationContext& context,
                const bool showProgress = false);

std::vector<imagePatch> selectPointsHex(const cv::Mat& img,
                                        const registrationParams& params);

std::vector<imagePatch> filterPatchesByQuality(const std::vector<imagePatch>& patches,
                                               const cv::Mat& refimg);

cv::Mat drawPoints(const cv::Mat& img, const std::vector<imagePatch>& patches);

cv::Mat lucky(const registrationParams& params,
              const registrationContext& context,
              const cv::Mat& refimg,
              const bool showProgress = false);

cv::Mat normalizeTo16Bits(const cv::Mat& inputImg);

#endif // IMAGEOPS_H
