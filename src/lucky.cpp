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

#include <iostream>
#include <algorithm>
#include <tuple>
#include "imageops.h"
#include "lucky.h"

using namespace cv;

patchCollection selectPointsHex(const registrationParams& params,
                                const registrationContext& context,
                                const cv::Rect patchCreationArea)
{
  const auto& refimg = context.refimg();
  patchCollection patches;
  patches.patchCreationArea = patchCreationArea;
  int originx = patchCreationArea.x;
  int originy = patchCreationArea.y;
  const int boxsize = context.boxsize();

  // We set maximum displacement to maxmove+1: the 1px border is used as a
  // "safety zone" (detecting maximum displacement in at least one direction
  // usually indicates that the local minimum is probably outside the
  // search area) and also to allow for the estimation of the local
  // curvature of fit around the minimum point.
  const int maxmb = params.maxmove + 1;

  // Points are arranged in a hexagonal grid. Each point is chosen sufficiently
  // far from the borders so that the search area (maxb in both directions)
  // is fully contained within patchCreationArea.
  const int xydiff = boxsize/2;
  int yspacing = ceil(xydiff*sqrt(0.75));
  const int xshift = xydiff/2;
  int period = 0;
  for (int y = 0;
       y <= patchCreationArea.height - boxsize;
       y += yspacing, period++)
    {
    for (int x = (period % 2 ? xshift : 0);
         x <= patchCreationArea.width - boxsize;
         x += xydiff) {
      Rect relativeSearchArea(Point(x-maxmb, y-maxmb), Point(x+boxsize+maxmb, y+boxsize+maxmb));
      imagePatch p(refimg, originx + x, originy + y, boxsize,
                   relativeSearchArea + patchCreationArea.tl());
      patches.push_back(p);
    }
  }
  return patches;
}


Mat1f patchMatcher::match(const Mat1f& img,
                          const Rect imgRect,
                          const Rect validRect,
                          const imagePatch& patch,
                          const float multiplier)
{
  Mat1f roi(img, patch.searchArea - imgRect.tl());
  patch.cookedMask.match(roi.mul(roi), roisq);
  patch.cookedTmpl.match(roi, cor);

  if (patch.searchAreaWithin(validRect))
  {
    // Search area is completely within the image. This is easy.
    return roisq - 2*multiplier*cor + pow(multiplier, 2)*patch.sqsum;
  }
  else {
    // Search area is only partially within the image. We need some more
    // computations to handle this.
    if (imgValidMask.size() != patch.searchArea.size())
      imgValidMask = Mat::zeros(patch.searchArea.size(), CV_32F);
    else
      imgValidMask.setTo(0);

    imgValidMask((patch.searchArea & validRect) - patch.searchArea.tl()) = 1;
    patch.cookedSquare.match(imgValidMask, patchsq);
    patch.cookedMask.match(imgValidMask, normalization);

    Mat1f unn_match = roisq - 2*multiplier*cor + pow(multiplier, 2)*patchsq;
    return unn_match.mul(1/normalization);
  }
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
patchCollection filterPatchesByQuality(const patchCollection& patches,
                                       const Mat& refimg) {
  // Patches that are good enough will be returned in this vector.
  patchCollection newPatches;
  newPatches.patchCreationArea = patches.patchCreationArea;

  Rect refimgRect(Point(0, 0), refimg.size());
  Rect totalArea = patches.searchAreaForImage(refimgRect);
  int top = -std::min(totalArea.y, 0);
  int left = -std::min(totalArea.x, 0);
  int bottom = std::max(totalArea.br().y - refimg.rows, 0);
  int right = std::max(totalArea.br().x - refimg.cols, 0);
  Mat paddedRefimg;
  copyMakeBorder(refimg, paddedRefimg, top, bottom, left, right, BORDER_CONSTANT, 0);
  Rect paddedRect = Rect(Point(-left, -top), paddedRefimg.size());
  refimgRect += Point(left, top);

  patchMatcher matcher;
  for (auto& patch : patches) {
    // perform the matching
    Mat1f match = matcher.match(paddedRefimg, paddedRect, refimgRect, patch, 1.0);

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


Mat drawPoints(const Mat& img, const patchCollection& patches) {
  Mat out = img.clone();
  for (auto& patch : patches) {
     circle(out, patch.center(), 2, Scalar(0, 0, 255));
     rectangle(out, Rect(patch.x, patch.y, patch.image.cols, patch.image.rows), Scalar(0, 255, 0));
  }
  return out;
}


Mat1f findShifts(const Mat& img,
                 const Rect imgRect,
                 const Rect validRect,
                 const patchCollection& patches,
                 const float multiplier,
                 patchMatcher& matcher) {
  Mat1f shifts(patches.size(), 2);
  int patchNr = -1;

  for (auto& patch : patches) {
    patchNr++;
    if (!patch.searchAreaOverlaps(validRect)) {
      shifts.at<float>(patchNr, 0) = 0;
      shifts.at<float>(patchNr, 1) = 0;
      continue;
    }

    Mat1f match = matcher.match(img, imgRect, validRect, patch, multiplier);
    Point coarseMin;
    Point2f subPixelMin(0,0);
    minMaxLoc(match, NULL, NULL, &coarseMin);

    // Check whether the match was located in the outer 1px buffer zone
    // (i.e., whether it has exceeded the given maxmove). This usually
    // indicates an extremely questionable match and we rather leave
    // the shift at (0,0) for this point. We also reject really pathological
    // cases (like a whole matrix of NaNs) which are indicated by minMaxLoc
    // reporting the minimum at (-1,-1).
    if (coarseMin.x > 0 && coarseMin.y > 0 &&
      coarseMin.x < match.cols - 1 && coarseMin.y < match.rows - 1) {
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
  }
  return shifts;
}


// Lucky imaging + stacking.
//
// These are, in principle, two separate operations. However, to minimize the
// number of needed image reads (and conversions), they are performed in a
// single parallelized loop. Parts of the loop specific to lucky imaging or
// stacking are in conditionals to allow the user to request only one operation
// to be performed.
//
Mat lucky(const registrationParams& params,
          registrationContext& context,
          const bool showProgress)
{
  Rect outputRectangle = Rect(Point(0, 0), context.imagesize());
  if (params.crop && context.commonRectangle.valid())
    outputRectangle = context.commonRectangle();

  const Mat& refimg = context.refimg();

  imageSumLookup refsqLookup;
  std::vector<Mat1f> allShifts;
  if (params.stage_lucky) {
    // Shifts will be computed during this run.
    allShifts.resize(context.images().size());
    refsqLookup = imageSumLookup(refimg.mul(refimg));
  }
  else if (params.stage_stack && context.shifts.valid()) {
    // Use shifts from a state file, if they are available.
    allShifts = context.shifts();
  }

  // STACKING: initialization
  Mat finalsum, normalization;
  rbfWarper* rbf = nullptr;
  if (params.stage_stack) {
    Mat sample = magickImread(context.images().at(0).filename);
    finalsum = Mat::zeros(outputRectangle.size() * params.supersampling,
                          CV_MAKETYPE(CV_32F, sample.channels()));
    normalization = Mat::zeros(finalsum.size(), CV_32F);

    // This could take quite some time if there is a lot of registration
    // points. Inform the user about what is going on.
    std::cerr << "Initializing the RBF warper (could take some time)... ";
    rbf = new rbfWarper(context.patches(), context.imagesize(), outputRectangle,
                 context.boxsize()/4, params.supersampling);
    std::cerr << "done\n";
  }

  int progress = 0;
  if (showProgress)
    std::fprintf(stderr, "0/%ld", context.images().size());
  #pragma omp parallel
  {
    // LUCKY IMAGING: local initialization
    patchMatcher matcher;
    // STACKING: local initialization
    Mat localsum;
    Mat localNormalization;
    if (params.stage_stack) {
      localsum = Mat::zeros(finalsum.size(), finalsum.type());
      localNormalization = Mat::zeros(normalization.size(), CV_32F);
    }

    // PARALLELIZED LOOP
    #pragma omp for schedule(dynamic)
    for (int ifile = 0; ifile < (signed)context.images().size(); ifile++) {
      // common step: load an image
      const auto& image = context.images().at(ifile);
      Mat inputImage = magickImread(image.filename);

      // LUCKY IMAGING: main operation
      if (params.stage_lucky) {
        Mat1f img;
        if (inputImage.channels() > 1)
          cvtColor(inputImage, img, COLOR_BGR2GRAY);
        else
          img = inputImage;

        // Image rectangle, expressed in coordinate systems of image itself
        // and the reference image.
        Rect img_coordImg(Point(0, 0), img.size());
        Rect img_coordRefimg = img_coordImg - image.globalShift;

        // Overlap between img and refimg, again according to both coordinate
        // systems.
        Rect overlap_coordRefimg = context.refimgRectangle() & img_coordRefimg;
        Rect overlap_coordImg = overlap_coordRefimg + image.globalShift;

        // Isolate the common portions of img and refimg.
        Mat imgOverlap(img, overlap_coordImg);
        Mat refimgOverlap(refimg, overlap_coordRefimg);

        // Calculate optimal multiplier for img vs. refimg.
        const float multiplier = sum(imgOverlap.mul(refimgOverlap))[0] /
                                 refsqLookup.lookup(overlap_coordRefimg);

        // Extract the part of image needed for matching and possibly pad it.
        Rect totalArea = context.patches().searchAreaForImage(img_coordRefimg);
        Rect searchOverlap = totalArea & img_coordRefimg;
        Mat imgSearchRoi(img, searchOverlap + image.globalShift);
        if (searchOverlap == totalArea)
          img = imgSearchRoi;
        else {
          Mat paddedImg = Mat::zeros(totalArea.size(), img.type());
          Rect destinationRoi = searchOverlap - totalArea.tl();
          imgSearchRoi.copyTo(paddedImg(destinationRoi));
          img = paddedImg;
        }

        // Find lucky imaging shifts.
        allShifts.at(ifile) =
          findShifts(img, totalArea, searchOverlap, context.patches(),
                     multiplier, matcher);
      }

      // STACKING: main operation
      if (params.stage_stack) {
        const Mat1f shifts(params.stage_lucky || context.shifts.valid() ?
                           allShifts.at(ifile) : Mat());
        Mat warpedImg, warpedNormalization;
        std::tie(warpedImg, warpedNormalization) =
          rbf->warp(inputImage, image.globalShift, shifts);
        localsum += warpedImg;
        localNormalization += warpedNormalization;
      }

      // progress indication
      if (showProgress) {
        #pragma omp critical
        std::fprintf(stderr, "\r\033[K%d/%ld", ++progress, context.images().size());
      }
    } // end of loop

    // STACKING: final sum
    if (params.stage_stack) {
      #pragma omp critical
      {
        finalsum += localsum;
        normalization += localNormalization;
      }
    }
  } // end of parallel section
  if (showProgress)
    std::fprintf(stderr, "\n");

  // LUCKY IMAGING: pass the results to registrationContext
  if (params.stage_lucky)
    context.shifts(allShifts);

  if (params.stage_stack) {
    divideChannelsByMask(finalsum, normalization);
    delete rbf;
  }

  // This is only going to return something meaningful if we performed
  // stacking; otherwise, an empty image will be returned.
  return finalsum;
}

