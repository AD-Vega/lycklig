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
#include "rbfwarper.h"

using namespace cv;

rbfWarper::rbfWarper(const vector<imagePatch>& patches_,
                     const Size& imagesize_,
                     const float sigma_,
                     const int supersampling_):
  patches(patches_), imagesize(imagesize_*supersampling_),
  sigma(sigma_*supersampling_), supersampling(supersampling_),
  bases(patches.size()), coeffs(patches.size(), patches.size()),
  xshiftbase(imagesize), yshiftbase(imagesize) {

  prepareBases();
  prepareCoeffs();
  for (int y = 0; y < imagesize.height; y++) {
    for (int x = 0; x < imagesize.width; x++) {
      xshiftbase.at<float>(y, x) = (2*(float)x - supersampling + 1)/(2*supersampling);
      yshiftbase.at<float>(y, x) = (2*(float)y - supersampling + 1)/(2*supersampling);
    }
  }
}


void rbfWarper::gauss1d(float* ptr, const Range& range, const float sigma) {
  const float sigmasq = sigma*sigma;
  for (int x = range.start; x < range.end; x++)
    *(ptr++) = exp(-0.5*(x*x)/sigmasq);
}


void rbfWarper::prepareBases() {
  Mat1f row(1, 2*imagesize.width-1);
  gauss1d(row.ptr<float>(0), Range(-imagesize.width+1, imagesize.width), sigma);
  Mat1f col(2*imagesize.height-1, 1);
  gauss1d(col.ptr<float>(0), Range(-imagesize.height+1, imagesize.height), sigma);
  Mat1f bigGauss(col*row);
  Point gaussCenter(imagesize.width, imagesize.height);

  for (int i = 0; i < (signed)patches.size(); i++) {
    Point baseCenter(patches.at(i).xcenter()*supersampling,
                     patches.at(i).ycenter()*supersampling);
    bases.at(i) = bigGauss(Rect(gaussCenter-baseCenter, imagesize));
  }
}


void rbfWarper::prepareCoeffs() {
  for (int i = 0; i < (signed)bases.size(); i++) {
    Point baseCenter(patches.at(i).xcenter()*supersampling,
                     patches.at(i).ycenter()*supersampling);
    coeffs.at<float>(i, i) = 1;
    for (int j = i+1; j < (signed)bases.size(); j++) {
      coeffs.at<float>(i, j) = coeffs.at<float>(j, i) = 
        bases.at(j).at<float>(baseCenter);
    }
  }
  coeffs = coeffs.inv(DECOMP_CHOLESKY);
}


Mat rbfWarper::warp(const Mat& image, const Mat1f& shifts) const {
  Mat1f weights(coeffs * shifts);
  Mat1f xshift(xshiftbase.clone());
  Mat1f yshift(yshiftbase.clone());
  for (int i = 0; i < (signed)bases.size(); i++) {
    xshift += bases.at(i) * weights.at<float>(i, 0);
    yshift += bases.at(i) * weights.at<float>(i, 1);
  }
  Mat imremap;
  remap(image, imremap, xshift, yshift, CV_INTER_LINEAR, BORDER_REFLECT);
  return imremap;
}
