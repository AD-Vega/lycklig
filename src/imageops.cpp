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

#include "imageops.h"

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

Mat grayReader::read(string file) {
  magickImread(file.c_str()).convertTo(imgcolor, CV_32F);
  cvtColor(imgcolor, imggray, CV_BGR2GRAY);
  return imggray;
}

Mat meanimg(const std::vector<std::string>& files,
            Rect crop,
            vector<Point> shifts,
            bool showProgress) {
  Mat sample = magickImread(files.at(0));
  bool doCrop;
  if (crop.size() == Size(0, 0)) {
    doCrop = false;
    crop = Rect(Point(0, 0), sample.size());
  }
  else {
    doCrop = true;
  }
  Mat imgmean(Mat::zeros(crop.size(), CV_MAKETYPE(CV_32F, sample.channels())));
  int progress = 0;
  if (showProgress)
    fprintf(stderr, "0/%ld", files.size());
  #pragma omp parallel
  {
    Mat localsum(imgmean.clone());
    #pragma omp barrier
    #pragma omp for
    for (int i = 0; i < (signed)files.size(); i++) {
      Mat img = magickImread(files.at(i));

      if (doCrop)
        accumulate(img(crop + shifts.at(i)), localsum);
      else
        accumulate(img, localsum);

      if (showProgress) {
        #pragma omp critical
        fprintf(stderr, "\r\033[K%d/%ld", ++progress, files.size());
      }
    }
    #pragma omp critical
    accumulate(localsum, imgmean);
  }
  if (showProgress)
    fprintf(stderr, "\n");
  imgmean /= files.size();
  return imgmean;
}


std::vector<imagePatch> selectPointsHex(const Mat img,
                                        const unsigned int boxsize,
                                        const unsigned int maxmove) {
  std::vector<imagePatch> patches;
  Rect imgrect(Point(0, 0), img.size());
  const int xydiff = boxsize/2;
  int yspacing = ceil(xydiff*sqrt(0.75));
  const int xshift = xydiff/2;
  int period = 0;
  for (int y = 0; y <= img.rows - (signed)boxsize; y += yspacing, period++) {
    for (int xbase = 0; xbase <= img.cols - (signed)boxsize; xbase += xydiff) {
      int x = (period % 2 ? xbase + xshift : xbase);
      if (x > img.rows - (signed)boxsize)
        break;
      Mat roi(img, Rect(x, y, boxsize, boxsize));
      Mat1f roif;
      roi.convertTo(roif, CV_32F);
      Rect fullSearchArea(Point(x-maxmove, y-maxmove), Point(x+boxsize+maxmove, y+boxsize+maxmove));
      imagePatch p(x, y, roif, fullSearchArea & imgrect);
      patches.push_back(p);
    }
  }
  return patches;
}


std::vector<imagePatch> filterPatchesByQuality(const std::vector<imagePatch> patches,
                                               const double val_threshold,
                                               const double surf_threshold) {
  std::vector<imagePatch> newPatches;
  for (auto& patch : patches) {
    double maxval;
    minMaxLoc(patch.image, NULL, &maxval);
    int overThreshold = countNonZero(patch.image > val_threshold * maxval);
    if (overThreshold > surf_threshold * patch.image.size().area())
      newPatches.push_back(patch);
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
                 const std::vector<imagePatch>& patches) {
  Mat1f shifts(patches.size(), 2);
  for (int i = 0; i < (signed)patches.size(); i++) {
    Mat1f roi(img, patches.at(i).searchArea);
    Mat1f patch(patches.at(i).image);
    Mat1f mask = Mat::ones(patch.rows, patch.cols, CV_32F);
    Mat1f areasq;
    matchTemplate(roi.mul(roi), mask, areasq, CV_TM_CCORR);
    Mat1f cor;
    matchTemplate(roi, patch, cor, CV_TM_CCORR);
    Mat1f match = areasq - (cor.mul(cor) / patches.at(i).sqsum);
    Point minpoint;
    minMaxLoc(match, NULL, NULL, &minpoint);
    int xshift = patches.at(i).x - patches.at(i).searchArea.x;
    int yshift = patches.at(i).y - patches.at(i).searchArea.y;
    minpoint -= Point(xshift, yshift);
    shifts.at<float>(i, 0) = minpoint.x;
    shifts.at<float>(i, 1) = minpoint.y;
  }
  return shifts;
}


globalRegistrator::globalRegistrator(const Mat& reference, const int maxmove) {
  refImgWithBorder = Mat::zeros(reference.rows + 2*maxmove, reference.cols + 2*maxmove, CV_32F);
  Rect imageRect = Rect(maxmove, maxmove, reference.cols, reference.rows);
  reference.copyTo(refImgWithBorder(imageRect));
  refImageArea = Mat::zeros(reference.rows + 2*maxmove, reference.cols + 2*maxmove, CV_32F);
  refImageArea(imageRect) = Mat::ones(reference.rows, reference.cols, CV_32F);
  searchMask = Mat::ones(reference.rows, reference.cols, CV_32F);
  matchTemplate(refImgWithBorder.mul(refImgWithBorder), searchMask, areasq, CV_TM_CCORR);
  originShift = Point(maxmove, maxmove);
}


Point globalRegistrator::findShift(const Mat& img)
{
  matchTemplate(refImageArea, img.mul(img), imgsq, CV_TM_CCORR);
  matchTemplate(refImgWithBorder, img, cor, CV_TM_CCORR);
  match = areasq - cor.mul(cor).mul(1/imgsq);
  Point minpoint;
  minMaxLoc(match, NULL, NULL, &minpoint);
  return -(minpoint - originShift);
}


std::vector<Point> getGlobalShifts(const std::vector<std::string>& files,
                                   const Mat& refimg,
                                   unsigned int maxmove,
                                   bool showProgress) {
  std::vector<Point> shifts(files.size());
  int progress = 0;
  if (showProgress)
    fprintf(stderr, "0/%ld", files.size());
  #pragma omp parallel
  {
    grayReader reader;
    globalRegistrator globalReg(refimg, maxmove);
    #pragma omp for schedule(dynamic)
    for (int ifile = 0; ifile < (signed)files.size(); ifile++) {
      Mat img(reader.read(files.at(ifile)));
      Point shift = globalReg.findShift(img);
      #pragma omp critical
      shifts.at(ifile) = shift;

      if (showProgress) {
        #pragma omp critical
        fprintf(stderr, "\r\033[K%d/%ld", ++progress, files.size());
      }
    }
  }
  if (showProgress)
    fprintf(stderr, "\n");

  return shifts;
}

Rect optimalCrop(std::vector<Point> shifts, Size size) {
  Rect crop(shifts.at(0), size);
  Rect origin(shifts.at(0), size);
  for (auto& shift : shifts) {
        crop &= Rect(shift, size);
        origin |= Rect(shift, size);
  }
  crop -= crop.tl() + origin.tl();
  return crop;
}


Mat3f lucky(registrationParams params,
            Mat refimg,
            Rect crop,
            std::vector<Point> globalShifts,
            std::vector<imagePatch> patches,
            rbfWarper rbf,
            bool showProgress) {
  Mat3f finalsum(Mat3f::zeros(refimg.size()));
  int progress = 0;
  if (showProgress)
    fprintf(stderr, "0/%ld", params.files.size());
  #pragma omp parallel
  {
    Mat3f localsum(Mat3f::zeros(refimg.size()));
    #pragma omp for schedule(dynamic)
    for (int ifile = 0; ifile < (signed)params.files.size(); ifile++) {
      Mat imgcolor;
      magickImread(params.files.at(ifile).c_str()).convertTo(imgcolor, CV_32F);
      if (params.prereg)
        imgcolor = imgcolor(crop + globalShifts.at(ifile));
      Mat1f img;
      cvtColor(imgcolor, img, CV_BGR2GRAY);
      Mat1f shifts(findShifts(img, patches));
      Mat imremap(rbf.warp(imgcolor, shifts));
      localsum += imremap;

      if (showProgress) {
        #pragma omp critical
        fprintf(stderr, "\r\033[K%d/%ld", ++progress, params.files.size());
      }
    }
    #pragma omp critical
    finalsum += localsum;
  }
  if (showProgress)
    fprintf(stderr, "\n");

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
