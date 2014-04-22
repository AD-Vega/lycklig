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
  return output;
}

Mat grayReader::read(const string& file) {
  magickImread(file.c_str()).convertTo(imgcolor, CV_32F);
  cvtColor(imgcolor, imggray, CV_BGR2GRAY);
  return imggray;
}

Mat meanimg(const std::vector<std::string>& files,
            const globalRegistration& globalReg,
            const bool showProgress) {
  Mat sample = magickImread(files.at(0));
  Rect crop(Point(0, 0), sample.size());
  if (globalReg.valid)
    crop = globalReg.crop;
  Mat imgmean(Mat::zeros(crop.size(), CV_MAKETYPE(CV_32F, sample.channels())));
  int progress = 0;
  if (showProgress)
    std::fprintf(stderr, "0/%ld", files.size());
  #pragma omp parallel
  {
    Mat localsum(imgmean.clone());
    #pragma omp barrier
    #pragma omp for
    for (int i = 0; i < (signed)files.size(); i++) {
      Mat img = magickImread(files.at(i));

      if (globalReg.valid)
        accumulate(img(crop + globalReg.shifts.at(i)), localsum);
      else
        accumulate(img, localsum);

      if (showProgress) {
        #pragma omp critical
        std::fprintf(stderr, "\r\033[K%d/%ld", ++progress, files.size());
      }
    }
    #pragma omp critical
    accumulate(localsum, imgmean);
  }
  if (showProgress)
    std::fprintf(stderr, "\n");
  imgmean /= files.size();
  return imgmean;
}


std::vector<imagePatch> selectPointsHex(const Mat& img,
                                        const registrationParams& params) {
  std::vector<imagePatch> patches;
  Rect imgrect(Point(0, 0), img.size());
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
  for (int y = maxmb; y <= img.rows - boxsize - maxmb; y += yspacing, period++) {
    for (int x = maxmb + (period % 2 ? xshift : 0);
         x <= img.cols - boxsize - maxmb;
         x += xydiff) {
      Mat roi(img, Rect(x, y, boxsize, boxsize));
      Mat1f roif;
      roi.convertTo(roif, CV_32F);
      Rect searchArea(Point(x-maxmb, y-maxmb), Point(x+boxsize+maxmb, y+boxsize+maxmb));
      imagePatch p(x, y, roif, searchArea);
      patches.push_back(p);
    }
  }
  return patches;
}


Mat1f patchMatcher::match(Mat1f img, imagePatch patch)
{
  if (mask.size() != patch.image.size())
    mask = Mat::ones(patch.image.rows, patch.image.cols, CV_32F);
  Mat1f roi(img, patch.searchArea);
  matchTemplate(roi.mul(roi), mask, areasq, CV_TM_CCORR);
  matchTemplate(roi, patch.image, cor, CV_TM_CCORR);
  Mat1f match = areasq - (cor.mul(cor) / patch.sqsum);
  return match;
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

  // A matrix of x^2, x*y and y^2 that will be needed for the quadratic fit.
  Mat fitx(9, 3, CV_32F);
  int row = 0;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      fitx.at<float>(row, 0) = x*x;
      fitx.at<float>(row, 1) = x*y;
      fitx.at<float>(row, 2) = y*y;
      row++;
    }
  }

  // Local neighbourhood of the central point.
  Mat aroundMinimum(3, 3, CV_32F);
  // Same, but shaped as a column vector (for fitting).
  Mat amAsVector = aroundMinimum.reshape(0, 9);
  // Fit coefficients.
  Mat coeffs;

  patchMatcher matcher;
  for (auto& patch : patches) {
    // perform the matching
    Mat1f match = matcher.match(refimg, patch);

    // Find the local neighbourhood of the central point and fit a 2D
    // quadratic polynomial to it.
    Point matchCenter(patch.x - patch.searchArea.x,
                      patch.y - patch.searchArea.y);
    Rect matchLocal3x3(matchCenter - Point(1, 1), Size(3, 3));
    match(matchLocal3x3).copyTo(aroundMinimum);
    solve(fitx, amAsVector, coeffs, DECOMP_SVD);

    // Calculate the smaller of the eigenvalues. Since this is only a 2x2
    // Hessian, the analytical solution is used.
    const float kxx = coeffs.at<float>(0);
    const float kxy = 0.5*coeffs.at<float>(1);
    const float kyy = coeffs.at<float>(2);
    const float lowEig = 0.5*(kxx + kyy - sqrt(pow(kxx - kyy, 2) + 4*pow(kxy, 2)));

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
                 patchMatcher& matcher) {
  Mat1f shifts(patches.size(), 2);
  for (int i = 0; i < (signed)patches.size(); i++) {
    Mat1f match = matcher.match(img, patches.at(i));
    Point minpoint;
    minMaxLoc(match, NULL, NULL, &minpoint);
    if (minpoint.x == 0 || minpoint.y == 0 ||
        minpoint.x == match.cols - 1 || minpoint.y == match.rows - 1) {
      // A match was located in the outer 1px buffer zone. This is shady
      // business. Pretend that we did not see anything.
      minpoint = Point(0, 0);
    }
    else {
      int xshift = patches.at(i).x - patches.at(i).searchArea.x;
      int yshift = patches.at(i).y - patches.at(i).searchArea.y;
      minpoint -= Point(xshift, yshift);
    }
    shifts.at<float>(i, 0) = minpoint.x;
    shifts.at<float>(i, 1) = minpoint.y;
  }
  return shifts;
}


Mat3f lucky(const registrationParams& params,
            const Mat& refimg,
            const globalRegistration& globalReg,
            const std::vector<imagePatch>& patches,
            const rbfWarper& rbf,
            const bool showProgress) {
  Mat3f finalsum(Mat3f::zeros(refimg.size()));
  int progress = 0;
  if (showProgress)
    std::fprintf(stderr, "0/%ld", params.files.size());
  #pragma omp parallel
  {
    Mat3f localsum(Mat3f::zeros(refimg.size()));
    patchMatcher matcher;
    #pragma omp for schedule(dynamic)
    for (int ifile = 0; ifile < (signed)params.files.size(); ifile++) {
      Mat imgcolor;
      magickImread(params.files.at(ifile).c_str()).convertTo(imgcolor, CV_32F);
      if (params.prereg)
        imgcolor = imgcolor(globalReg.crop + globalReg.shifts.at(ifile));
      Mat1f img;
      cvtColor(imgcolor, img, CV_BGR2GRAY);
      Mat1f shifts(findShifts(img, patches, matcher));
      Mat imremap(rbf.warp(imgcolor, shifts));
      localsum += imremap;

      if (showProgress) {
        #pragma omp critical
        std::fprintf(stderr, "\r\033[K%d/%ld", ++progress, params.files.size());
      }
    }
    #pragma omp critical
    finalsum += localsum;
  }
  if (showProgress)
    std::fprintf(stderr, "\n");

  return finalsum;
}


Mat3w normalizeTo16Bits(const Mat& inputImg) {
  Mat img = inputImg.clone();
  double minval, maxval;
  minMaxLoc(img, &minval, &maxval);
  img = (img - minval)/(maxval - minval) * ((1<<16)-1);
  Mat3w imgout;
  img.convertTo(imgout, CV_16UC3);
  return imgout;
}
