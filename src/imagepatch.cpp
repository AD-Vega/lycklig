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

#include "imagepatch.h"

imagePatchPosition::imagePatchPosition(int xpos, int ypos, cv::Rect search) :
  x(xpos), y(ypos), searchArea(search) {}


imagePatchPosition::imagePatchPosition(const cv::FileNode& node) {
  int xpos, ypos;
  node["x"] >> xpos;
  node["y"] >> ypos;
  x = xpos;
  y = ypos;
  node["searchArea"] >> searchArea;
}


void imagePatchPosition::write(cv::FileStorage& fs) const {
  fs << "{"
     << "x" << static_cast<int>(x)
     << "y" << static_cast<int>(y)
     << "searchArea" << searchArea
     << "}";
}


void write(cv::FileStorage& fs,
           const std::string&,
           const imagePatchPosition& patch) {
  patch.write(fs);
}


bool imagePatchPosition::searchAreaWithin(const cv::Rect rect) const
{
  return rect.contains(searchArea.tl()) &&
         rect.contains(searchArea.br() - cv::Point(1, 1));
}


bool imagePatchPosition::searchAreaOverlaps(const cv::Rect rect) const
{
  return rect.contains(searchArea.tl()) ||
         rect.contains(searchArea.br() - cv::Point(1, 1));
}


imagePatch::imagePatch(cv::Mat img, imagePatchPosition position, int boxsize) :
  imagePatchPosition(position),
  image(img(cv::Rect((int)position.x, (int)position.y, boxsize, boxsize))),
  sqsum(sum(image.mul(image))[0]), cookedTmpl(image, position.searchArea.size()),
  cookedMask(cv::Mat::ones(image.size(), CV_32F), position.searchArea.size()) {}


imagePatch::imagePatch(cv::Mat img, int xpos, int ypos, int boxsize, cv::Rect search) :
  imagePatch(img, imagePatchPosition(xpos, ypos, search), boxsize) {}


cv::Rect patchCollection::searchAreaForImage(const cv::Rect imageRect) const
{
  cv::Rect totalRect(imageRect);

  for (const auto& patch : *this)
  {
    if (patch.searchAreaOverlaps(imageRect))
      totalRect |= patch.searchArea;
  }
  return totalRect;
}
