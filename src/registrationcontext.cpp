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

#include "registrationcontext.h"

inputImage::inputImage(std::string filename_) :
  filename(filename_), globalShift(0, 0)
{}


inputImage::inputImage(const cv::FileNode& node) {
  node["filename"] >> filename;
  node["globalShift"] >> globalShift;
}


void inputImage::write(cv::FileStorage& fs) const {
  fs << "{"
     << "filename" << filename
     << "globalShift" << globalShift
     << "}";
}


void write(cv::FileStorage& fs, const std::string&, const inputImage& image) {
  image.write(fs);
}


registrationContext::registrationContext(const cv::FileStorage& fs) {
  fs["boxsize"] >> boxsize;
  fs["crop"] >> crop;
  fs["refimg"] >> refimg;

  for (const auto& i : fs["images"])
    images.push_back(inputImage(i));

  for (const auto& i : fs["patches"])
    patches.push_back(imagePatch(refimg, imagePatchPosition(i), boxsize));
}


void write(cv::FileStorage& fs,
           const std::string&,
           const std::vector<inputImage>& images) {
  fs << "[";
  for (auto& image : images)
    fs << image;
  fs << "]";
}

void write(cv::FileStorage& fs,
           const std::string&,
           const std::vector<imagePatch>& patches) {
  fs << "[";
  for (auto& patch : patches)
    fs << patch;
  fs << "]";
}

void registrationContext::write(cv::FileStorage& fs) const {
  fs << "boxsize" << boxsize
     << "images" << images
     << "crop" << crop
     << "patches" << patches
     << "refimg" << refimg;
}

void write(cv::FileStorage& fs,
           const std::string&,
           const registrationContext& context) {
  context.write(fs);
}
