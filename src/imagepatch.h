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

class imagePatch {
public:
  inline imagePatch(int xpos, int ypos, cv::Mat img, cv::Rect search) :
    x(xpos), y(ypos), image(img), searchArea(search),
    sqsum(sum(img.mul(img))[0]) {}
  inline int xcenter() const { return x + image.cols/2; }
  inline int ycenter() const { return y + image.rows/2; }

  unsigned int x;
  unsigned int y;
  cv::Mat image;
  cv::Rect searchArea;
  double sqsum;
};

#endif // IMAGEPATCH_H
