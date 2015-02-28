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

#ifndef IMAGEPATCH_H
#define IMAGEPATCH_H

#include <opencv2/core/core.hpp>
#include "cookedtemplate.h"

class imagePatchPosition {
public:
  imagePatchPosition(int xpos, int ypos, cv::Rect search);
  imagePatchPosition(const cv::FileNode& node);
  void write(cv::FileStorage& fs) const;
  bool searchAreaWithin(const cv::Rect rect) const;
  bool searchAreaOverlaps(const cv::Rect rect) const;

  unsigned int x;
  unsigned int y;
  cv::Rect searchArea;
};

void write(cv::FileStorage& fs,
           const std::string&,
           const imagePatchPosition& patch);


class imagePatch : public imagePatchPosition {
public:
  imagePatch(cv::Mat img, imagePatchPosition position, int boxsize);
  imagePatch(cv::Mat img, int xpos, int ypos, int boxsize, cv::Rect search);

  inline cv::Point2f center() const
    { return cv::Point2f(x + (image.cols-1)/2.0, y + (image.rows-1)/2.0); }
  inline int matchShiftx() const { return x - searchArea.x; }
  inline int matchShifty() const { return y - searchArea.y; }

  cv::Mat image;
  double sqsum;
  cookedTemplate cookedTmpl;
  cookedTemplate cookedMask;
  cookedTemplate cookedSquare;
};


class patchCollection : public std::vector<imagePatch> {
public:
  // This method returns a rectangle that contains all the search areas that
  // are applicable to the image of a given size and position (imageRect).
  // The returned rectangle is never smaller than the image itself.
   cv::Rect searchAreaForImage(const cv::Rect imageRect) const;

   // Area of the reference image on which the patches were collected.
   cv::Rect patchCreationArea = cv::Rect(0, 0, 0, 0);
};

#endif // IMAGEPATCH_H
