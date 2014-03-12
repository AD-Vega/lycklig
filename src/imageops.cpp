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

Mat meanimg(const std::vector<std::string>& files, bool showProgress) {
  Mat sample = imread(files.at(0));
  Mat imgmean(Mat::zeros(sample.size(), CV_MAKETYPE(CV_32F, sample.channels())));
  int progress = 0;
  #pragma omp parallel
  {
    Mat localsum(imgmean.clone());
    #pragma omp barrier
    #pragma omp for
    for (int i = 0; i < (signed)files.size(); i++) {
      if (showProgress) {
        #pragma omp critical
        fprintf(stderr, "\r\033[K%d/%ld", ++progress, files.size());
      }
      accumulate(imread(files.at(i)), localsum);
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
                                        const unsigned int xydiff,
                                        const double val_threshold,
                                        const double surf_threshold) {
  std::vector<imagePatch> patches;
  int yspacing = ceil(xydiff*sqrt(0.75));
  int xshift = xydiff/2;
  int period = 0;
  for (int y = 0; y <= img.rows - (signed)boxsize; y += yspacing, period++) {
    for (int xbase = 0; xbase <= img.cols - (signed)boxsize; xbase += xydiff) {
      int x = (period % 2 ? xbase + xshift : xbase);
      if (x > img.rows - (signed)boxsize)
        break;
      Mat roi(img, Rect(x, y, boxsize, boxsize));
      double maxval;
      minMaxLoc(roi, NULL, &maxval);
      int overThreshold = countNonZero(roi > val_threshold * maxval);
      if (overThreshold > surf_threshold * boxsize * boxsize) {
        Mat1f roif;
        roi.convertTo(roif, CV_32F);
        imagePatch p(x, y, roif);
        patches.push_back(p);
      }
    }
  }
  return patches;
}


Mat drawPoints(const Mat& img, const std::vector<imagePatch>& patches) {
  Mat out = img.clone();
  for (auto i = patches.begin(); i != patches.end(); i++) {
     circle(out, Point(i->xcenter(), i->ycenter()), 2, Scalar(0, 0, 255));
     rectangle(out, Rect(i->x, i->y, i->image.cols, i->image.rows), Scalar(0, 255, 0));
  }
  return out;
}


std::vector<Rect> createSearchAreas(const std::vector<imagePatch>& patches,
                                    const Size& imagesize,
                                    const int maxmove) {
  std::vector<Rect> areas(patches.size());
  Rect imgrect(Point(0, 0), imagesize);
  for (int i = 0; i < (signed)patches.size(); i++) {
    int x = patches.at(i).x;
    int y = patches.at(i).y;
    int width = patches.at(i).image.cols;
    int height = patches.at(i).image.rows;
    Rect r(Point(x-maxmove, y-maxmove), Point(x+width+maxmove, y+height+maxmove));
    areas.at(i) = r & imgrect;
  }
  return areas;
}


Mat1f findShifts(const Mat& img,
                 const std::vector<imagePatch>& patches,
                 const std::vector<Rect>& areas) {
  Mat1f shifts(patches.size(), 2);
  for (int i = 0; i < (signed)patches.size(); i++) {
    Mat1f roi(img, areas.at(i));
    Mat1f patch(patches.at(i).image);
    Mat1f mask = Mat::ones(patch.rows, patch.cols, CV_32F);
    Mat1f areasq;
    matchTemplate(roi.mul(roi), mask, areasq, CV_TM_CCORR);
    Mat1f cor;
    matchTemplate(roi, patch, cor, CV_TM_CCORR);
    Mat1f match = areasq - (cor.mul(cor) / patches.at(i).sqsum);
    Point minpoint;
    minMaxLoc(match, NULL, NULL, &minpoint);
    int xshift = patches.at(i).x - areas.at(i).x;
    int yshift = patches.at(i).y - areas.at(i).y;
    minpoint -= Point(xshift, yshift);
    shifts.at<float>(i, 0) = minpoint.x;
    shifts.at<float>(i, 1) = minpoint.y;
  }
  return shifts;
}
