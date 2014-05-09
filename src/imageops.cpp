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

#include <limits>
#include "imageops.h"
#include "globalregistrator.h"

using namespace cv;

Mat magickImread(const std::string& filename)
{
  Magick::Image image;
  image.read(filename);
  cv::Mat output(image.rows(), image.columns(), CV_32FC3);
  image.write(0, 0, image.columns(), image.rows(),
                 "BGR", Magick::FloatPixel, output.data);
  sRGB2linearRGB(output);
  return output;
}


void sRGB2linearRGB(Mat& img) {
  int rows = img.rows;
  int cols = img.cols;
  if (img.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for(int row = 0; row < rows; row++)
  {
    float* ptr = img.ptr<float>(row);
    for (int i = 0; i < 3*cols; i++) {
      *ptr = (*ptr <= 0.04045 ? *ptr/12.92 : pow((*ptr + 0.055)/1.055, 2.4));
      ptr++;
    }
  }
}


void linearRGB2sRGB(Mat& img) {
  int rows = img.rows;
  int cols = img.cols;
  if (img.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for(int row = 0; row < rows; row++)
  {
    float* ptr = img.ptr<float>(row);
    for (int i = 0; i < 3*cols; i++) {
      *ptr = (*ptr <= 0.0031308 ? *ptr*12.92 : 1.055*pow(*ptr, 1/2.4) - 0.055);
      ptr++;
    }
  }
}


Mat grayReader::read(const string& file) {
  magickImread(file.c_str()).convertTo(imgcolor, CV_32F);
  cvtColor(imgcolor, imggray, CV_BGR2GRAY);
  return imggray;
}

Mat meanimg(const registrationParams& params,
            const registrationContext& context,
            const bool showProgress) {
  const auto& images = context.images;
  const bool globalRegValid = (context.crop.width > 0 && context.crop.height > 0);

  Mat sample = magickImread(images.at(0).filename);
  Rect imgRect(Point(0, 0), sample.size());

  Mat imgmean;
  if (params.crop && globalRegValid)
    imgmean = Mat::zeros(context.crop.size(), CV_MAKETYPE(CV_32F, sample.channels()));
  else
    imgmean = Mat::zeros(sample.size(), CV_MAKETYPE(CV_32F, sample.channels()));

  int progress = 0;
  if (showProgress)
    std::fprintf(stderr, "0/%ld", images.size());
  #pragma omp parallel
  {
    Mat localsum(imgmean.clone());
    #pragma omp barrier
    #pragma omp for
    for (int i = 0; i < (signed)images.size(); i++) {
      auto image = images.at(i);
      Mat data = magickImread(image.filename);

      if (globalRegValid) {
        if (params.crop)
          accumulate(data(context.crop + image.globalShift), localsum);
        else {
          Rect sourceRoi = (imgRect + image.globalShift) & imgRect;
          Rect destRoi = sourceRoi - image.globalShift;
          accumulate(data(sourceRoi), localsum(destRoi));
        }
      }
      else
        accumulate(data, localsum);

      if (showProgress) {
        #pragma omp critical
        std::fprintf(stderr, "\r\033[K%d/%ld", ++progress, images.size());
      }
    }
    #pragma omp critical
    accumulate(localsum, imgmean);
  }
  if (showProgress)
    std::fprintf(stderr, "\n");
  imgmean /= images.size();
  return imgmean;
}

std::vector<imagePatch> selectPointsHex(const registrationParams& params,
                                        const registrationContext& context) {
  const auto& refimg = context.refimg;
  std::vector<imagePatch> patches;
  Rect imgrect(Point(0, 0), refimg.size());
  const int boxsize = params.boxsize;

  // We set maximum displacement to maxmove+1: the 1px border is used as a
  // "safety zone" (detecting maximum displacement in at least one direction
  // usually indicates that the local minimum is probably outside the
  // search area) and also to allow for the estimation of the local
  // curvature of fit around the minimum point.
  const int maxmb = params.maxmove + 1;

  // Points are arranged in a hexagonal grid. Each point is chosen sufficiently
  // far from the borders so that the search area (maxb in both directions)
  // is fully contained in the reference image.
  const int xydiff = boxsize/2;
  int yspacing = ceil(xydiff*sqrt(0.75));
  const int xshift = xydiff/2;
  int period = 0;
  for (int y = maxmb; y <= refimg.rows - boxsize - maxmb; y += yspacing, period++) {
    for (int x = maxmb + (period % 2 ? xshift : 0);
         x <= refimg.cols - boxsize - maxmb;
         x += xydiff) {
      Mat roi(refimg, Rect(x, y, boxsize, boxsize));
      Rect searchArea(Point(x-maxmb, y-maxmb), Point(x+boxsize+maxmb, y+boxsize+maxmb));
      imagePatch p(roi, x, y, searchArea);
      patches.push_back(p);
    }
  }
  return patches;
}


Mat1f patchMatcher::match(const Mat1f& img,
                          const imagePatch& patch,
                          const float multiplier)
{
  if (mask.size() != patch.image.size())
    mask = Mat::ones(patch.image.rows, patch.image.cols, CV_32F);
  Mat1f roi(img, patch.searchArea);
  patch.cookedMask.match(roi.mul(roi), areasq);
  patch.cookedTmpl.match(roi, cor);
  Mat1f match = areasq - 2*multiplier*cor + pow(multiplier, 2)*patch.sqsum;
  return match;
}


Mat1f quadraticFit::fitx = [] {
  Mat1f fitx(9, 6, CV_32F);
  int row = 0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      fitx.at<float>(row, 0) = 1;
      fitx.at<float>(row, 1) = x;
      fitx.at<float>(row, 2) = y;
      fitx.at<float>(row, 3) = x*x;
      fitx.at<float>(row, 4) = x*y;
      fitx.at<float>(row, 5) = y*y;
      row++;
    }
  }
  return fitx;
}();


quadraticFit::quadraticFit(const Mat& data, const Point& point) {
  // Local neighbourhood of the central point.
  Mat aroundMinimum(3, 3, CV_32F);
  // Same, but shaped as a column vector (for fitting).
  Mat amAsVector(aroundMinimum.reshape(0, 9));

  // Extract the data and perform fitting.
  Rect matchLocal3x3(point - Point(1, 1), Size(3, 3));
  data(matchLocal3x3).copyTo(aroundMinimum);
  Mat coeffs;
  solve(fitx, amAsVector, coeffs, DECOMP_SVD);

  H = (Mat_<float>(3,3) <<
    coeffs.at<float>(0), 0, 0,
    coeffs.at<float>(1), coeffs.at<float>(3), 0,
    coeffs.at<float>(2), coeffs.at<float>(4), coeffs.at<float>(5));
  H = (H + H.t())/2;
  solve(H(Range(1,3), Range(1,3)), -H(Range(1,3), Range(0,1)), x0y0);
  Mat S = Mat::eye(3, 3, CV_32F);
  S.at<float>(1, 0) = x0y0.at<float>(0);
  S.at<float>(2, 0) = x0y0.at<float>(1);
  H = S.t() * H * S;
  eigen(H(Range(1,3), Range(1,3)), eigenvalues, eigenvectors);
}


Point2f quadraticFit::minimum() const {
  return Point2f(x0y0.at<float>(0), x0y0.at<float>(1));
}

float quadraticFit::largerEig() const {
  return eigenvalues.at<float>(0);
}

float quadraticFit::smallerEig() const {
  return eigenvalues.at<float>(1);
}

Point2f quadraticFit::largerEigVec() const
{
  return Point2f(eigenvectors.at<float>(0, 0), eigenvectors.at<float>(1, 0));
}

Point2f quadraticFit::smallerEigVec() const
{
  return Point2f(eigenvectors.at<float>(0, 1), eigenvectors.at<float>(1, 1));
}


// Patch quality estimation
//
// Patch quality is assessed as follows: each patch is matched against its
// own search area on the reference image. Local curvature around the central
// (best matching - by definition) point is then estimated by fitting a 2D
// quadratic polynomial to a 3x3 pixel neighbourhood of the point, yielding a
// Hessian matrix. The smaller of the two eigenvalues of this Hessian
// represents the worst-case (smallest) change in match value that one can get
// by moving one pixel away from the central point. If the match field contains
// more than one point for which the match value is below this eigenvalue,
// the patch quality is deemed insufficient and the patch is rejected.
//
std::vector<imagePatch> filterPatchesByQuality(const std::vector<imagePatch>& patches,
                                               const Mat& refimg) {
  // Patches that are good enough will be returned in this vector.
  std::vector<imagePatch> newPatches;

  patchMatcher matcher;
  for (auto& patch : patches) {
    // perform the matching
    Mat1f match = matcher.match(refimg, patch, 1.0);

    // Find the local neighbourhood of the central point and fit a 2D
    // quadratic polynomial to it.
    Point matchCenter(patch.matchShiftx(), patch.matchShiftx());
    quadraticFit qf(match, matchCenter);
    const float lowEig = qf.smallerEig();

    // No point in dealing with eigenvalues smaller than epsilon. We also
    // reject negative eigenvalues with this test.
    if (lowEig >= std::numeric_limits<float>::epsilon()) {
      // Tunable parameter for possible future use.
      const float eigMult = 1.0;
      int overThreshold = countNonZero(match < lowEig*eigMult);
      // Note that in some pathological cases, overThreshold can actually end
      // up being zero. We don't want to mess with those anyway, so we only
      // accept the patch if overThreshold is exactly one.
      if (overThreshold == 1)
        newPatches.push_back(patch);
    }
  }
  return newPatches;
}


Mat drawPoints(const Mat& img, const std::vector<imagePatch>& patches) {
  Mat out = img.clone();
  for (auto& patch : patches) {
     circle(out, Point(patch.xcenter(), patch.ycenter()), 2, Scalar(0, 0, 255));
     rectangle(out, Rect(patch.x, patch.y, patch.image.cols, patch.image.rows), Scalar(0, 255, 0));
  }
  return out;
}


Mat1f findShifts(const Mat& img,
                 const std::vector<imagePatch>& patches,
                 const float multiplier,
                 patchMatcher& matcher) {
  Mat1f shifts(patches.size(), 2);
  int patchNr = 0;
  for (auto& patch : patches) {
    Mat1f match = matcher.match(img, patch, multiplier);
    Point coarseMin;
    minMaxLoc(match, NULL, NULL, &coarseMin);
    Point2f subPixelMin(0,0);
    // Check whether the match was located in the outer 1px buffer zone
    // (i.e., whether it has exceeded the given maxmove). This usually
    // indicates an extremely questionable match and we rather leave
    // the shift at (0,0) for this point.
    if (coarseMin.x != 0 && coarseMin.y != 0 &&
      coarseMin.x != match.cols - 1 && coarseMin.y != match.rows - 1) {
      // The coarse estimate seems OK; do subpixel correction now.
      subPixelMin = coarseMin;
      quadraticFit qf(match, coarseMin);
      Point2f subShift = qf.minimum();
      if (abs(subShift.x) > 0.5 || abs(subShift.y) > 0.5) {
        // Subpixel correction larger than 0.5 px indicates poor fit. Project
        // out the direction corresponding to the smaller eigenvalue and see
        // if that helps.
        subShift = subShift.dot(qf.largerEigVec()) * qf.largerEigVec();
        // Give up if the shift is still larger than 0.5 px.
        if (abs(subShift.x) > 0.5 || abs(subShift.y) > 0.5)
          subShift = Point2f(0, 0);
      }
      subPixelMin += subShift;
      // The shift is reported relative to the top left corner in the
      // image. Change it so that it refers to the center.
      subPixelMin -= Point2f(patch.matchShiftx(), patch.matchShifty());
    }
    shifts.at<float>(patchNr, 0) = subPixelMin.x;
    shifts.at<float>(patchNr, 1) = subPixelMin.y;
    patchNr++;
  }
  return shifts;
}

#include <iostream>

Mat lucky(const registrationParams& params,
          const registrationContext& context,
          const bool showProgress) {
  const auto& refimg = context.refimg;
  rbfWarper rbf(context.patches, refimg.size(), params.boxsize/4, params.supersampling);
  const float refimgsq = sum(refimg.mul(refimg))[0];
  const bool globalRegValid = (context.crop.width > 0 && context.crop.height > 0);

  Mat finalsum;
  if (params.crop && globalRegValid)
    finalsum = Mat::zeros(context.crop.size() * params.supersampling, CV_32FC3);
  else
    finalsum = Mat::zeros(refimg.size() * params.supersampling, CV_32FC3);

  int progress = 0;
  if (showProgress)
    std::fprintf(stderr, "0/%ld", context.images.size());
  #pragma omp parallel
  {
    Mat localsum(Mat::zeros(refimg.size() * params.supersampling, CV_32FC3));
    patchMatcher matcher;
    #pragma omp for schedule(dynamic)
    for (int ifile = 0; ifile < (signed)context.images.size(); ifile++) {
      auto image = context.images.at(ifile);
      Mat imgcolor;
      magickImread(image.filename).convertTo(imgcolor, CV_32F);
      if (globalRegValid) {
        if (params.crop)
          imgcolor = imgcolor(context.crop + image.globalShift);
        else {
          Mat tmp = Mat::zeros(imgcolor.size(), imgcolor.type());
          Rect imgRect(Point(0, 0), imgcolor.size());
          Rect sourceRoi = (imgRect + image.globalShift) & imgRect;
          Rect destRoi = sourceRoi - image.globalShift;
          imgcolor(sourceRoi).copyTo(tmp(destRoi));
          imgcolor = tmp;
        }
      }
      Mat1f img;
      cvtColor(imgcolor, img, CV_BGR2GRAY);
      const float multiplier = sum(img.mul(refimg))[0] / refimgsq;
      Mat1f shifts(findShifts(img, context.patches, multiplier, matcher));
      Mat imremap(rbf.warp(imgcolor, shifts));
      localsum += imremap;

      if (showProgress) {
        #pragma omp critical
        std::fprintf(stderr, "\r\033[K%d/%ld", ++progress, context.images.size());
      }
    }
    #pragma omp critical
    finalsum += localsum;
  }
  if (showProgress)
    std::fprintf(stderr, "\n");

  return finalsum;
}


Mat normalizeTo16Bits(const Mat& inputImg) {
  Mat img = inputImg.clone();
  double minval, maxval;
  minMaxLoc(img, &minval, &maxval);
  img = (img - minval)/(maxval - minval);
  linearRGB2sRGB(img);
  img *= ((1<<16)-1);
  Mat imgout;
  img.convertTo(imgout, CV_16UC3);
  return imgout;
}
