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

#include <opencv2/imgproc/imgproc.hpp>
#include <limits>
#include "rbfwarper.h"

using namespace cv;


rbfWarper::rbfWarper(const patchCollection& patches_,
                     const cv::Size inputImageSize,
                     const Rect& targetRect,
                     const float sigma_,
                     const int supersampling_):
  patches(patches_), targetOrigin(targetRect.tl()),
  imagesize(targetRect.size()*supersampling_),
  sigma(sigma_*supersampling_), supersampling(supersampling_),
  normalizationMask(Mat::ones(inputImageSize, CV_32F)),
  coeffs(patches.size(), patches.size()),
  xshiftbase(imagesize), yshiftbase(imagesize)
{
  for (int y = 0; y < imagesize.height; y++) {
    for (int x = 0; x < imagesize.width; x++) {
      xshiftbase.at<float>(y, x) =
        (2*(float)x - supersampling + 1)/(2*supersampling) + targetOrigin.x;
      yshiftbase.at<float>(y, x) =
        (2*(float)y - supersampling + 1)/(2*supersampling) + targetOrigin.y;
    }
  }

  if (!patches.empty())
    prepareBases();
}


void rbfWarper::gauss1d(float* ptr, const Range& range, const float sigma) const {
  const float sigmasq = sigma*sigma;
  for (int x = range.start; x <= range.end; x++)
    *(ptr++) = exp(-0.5*(x*x)/sigmasq);
}


void rbfWarper::prepareBases() {
  Point2f targetF = Point2f(targetOrigin.x, targetOrigin.y);

  // We will create one large plane with a 2D Gaussian and then cut out
  // portions of it to create the needed set of basis functions.
  // First, we determine how larget the "mother" plane must be.
  const float inf = std::numeric_limits<float>::infinity();
  Point2f tl(inf, inf);
  Point2f br(-inf, -inf);
  for (auto& patch : patches) {
    const float x = patch.center().x;
    const float y = patch.center().y;
    if (x < tl.x)
      tl.x = x;
    if (x > br.x)
      br.x = x;
    if (y < tl.y)
      tl.y = y;
    if (y > br.y)
      br.y = y;
  }
  tl = (tl - targetF) * supersampling;
  br = (br - targetF) * supersampling;
  int minx = floor(tl.x);
  int miny = floor(tl.y);
  int maxx = ceil(br.x);
  int maxy = ceil(br.y);

  // Determine the size of the rectangle that contains all the basis function
  // centers and the destination image.
  Rect allCenters(minx, miny, maxx + 1, maxy + 1);
  Rect targetRect(targetOrigin, imagesize);
  basesRect = allCenters | targetRect;

  // Create the 1D Gaussian kernel.
  // 5 sigma ought to be enough for everybody :-)
  const int halfKernelSize = 5 * sigma;
  Range gaussRange(-halfKernelSize, halfKernelSize);
  gaussianKernel = Mat(2 * halfKernelSize+1, 1, CV_32F);
  gauss1d(gaussianKernel.ptr<float>(0), gaussRange, sigma);

  const float sigmasq = pow(sigma, 2);

  // In this loop, we prepare the matrix that will be inverted to solve the
  // equations for the basis coefficients.
  for (int i = 0; i < (signed)patches.size(); i++) {
    Point baseCenter = (patches.at(i).center() - targetF) * supersampling;

    // Due to the way which the basis functions are constructed, the origin
    // of the Gaussian functions always lies exactly in the center of a pixel.
    // This location does not necessarily coincide with the center of the
    // respective registration point. The entries in the diagonal of the
    // matrix are therefore not exactly equal to one (although they should
    // end up being quite close).
    Point2f baseCenterF(baseCenter.x, baseCenter.y);
    Point2f patchCenter = (patches.at(i).center() - targetF) * supersampling;
    Point2f centerDiff = patchCenter - baseCenterF;
    float centerDistSq = pow(centerDiff.x, 2) + pow(centerDiff.y, 2);
    coeffs.at<float>(i, i) = exp(-0.5*centerDistSq/sigmasq);

    // Determine the off-diagonal coefficients for the current basis function.
    for (int j = i+1; j < (signed)patches.size(); j++) {
      Point2f diff = patches.at(j).center() - patches.at(i).center();
      diff *= supersampling;
      float distanceSq = pow(diff.x, 2) + pow(diff.y, 2);
      coeffs.at<float>(i, j) = coeffs.at<float>(j, i) =
         exp(-0.5*distanceSq/sigmasq);
    }
  }
  // Invert the matrix. The resulting matrix is then ready to be multiplied
  // by a vector of lucky imaging shifts to yield the corresponding basis
  // function weights.
  coeffs = coeffs.inv(DECOMP_CHOLESKY);
}


std::pair<Mat, Mat>
rbfWarper::warp(const Mat& image,
                const Point& globalShift,
                const Mat1f& shifts) const {
  Mat xField, yField;

  if (!shifts.empty()) {
    Mat1f weights(coeffs * shifts);
    Mat xshiftPoints = Mat::zeros(basesRect.size(), CV_32F);
    Mat yshiftPoints = Mat::zeros(basesRect.size(), CV_32F);

    for (int i = 0; i < (signed)patches.size(); i++) {
      Point baseCenter = patches.at(i).center() * supersampling;
      baseCenter -= basesRect.tl();
      xshiftPoints.at<float>(baseCenter) = weights.at<float>(i, 0);
      yshiftPoints.at<float>(baseCenter) = weights.at<float>(i, 1);
    }

    Mat xshift(basesRect.size(), CV_32F);
    Mat yshift(basesRect.size(), CV_32F);
    sepFilter2D(xshiftPoints, xshift, -1, gaussianKernel, gaussianKernel,
                Point(-1,-1), 0, BORDER_CONSTANT);
    sepFilter2D(yshiftPoints, yshift, -1, gaussianKernel, gaussianKernel,
                Point(-1,-1), 0, BORDER_CONSTANT);

    xField = xshift(Rect(targetOrigin, imagesize)) + xshiftbase + globalShift.x;
    yField = yshift(Rect(targetOrigin, imagesize)) + yshiftbase + globalShift.y;
  }
  else {
    xField = xshiftbase + globalShift.x;
    yField = yshiftbase + globalShift.y;
  }

  Mat imremap, normremap;
  remap(image, imremap, xField, yField,
        CV_INTER_LINEAR, BORDER_CONSTANT, 0);
  remap(normalizationMask, normremap, xField, yField,
        CV_INTER_LINEAR, BORDER_CONSTANT, 0);
  return std::pair<Mat, Mat>(imremap, normremap);
}
