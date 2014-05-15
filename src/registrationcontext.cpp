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
  if (fs["boxsize"].isInt()) {
    fs["boxsize"] >> priv_boxsize;
    boxsize_valid = true;
  }

  if (! fs["crop"].empty()) {
    fs["crop"] >> priv_crop;
    crop_valid = true;
  }

  if (! fs["refimg"].empty()) {
    fs["refimg"] >> priv_refimg;
    refimg_valid = true;
  }

  if (fs["images"].isSeq()) {
    for (const auto& i : fs["images"])
      priv_images.push_back(inputImage(i));
    images_valid = true;
  }

  if (fs["patches"].isSeq()) {
    for (const auto& i : fs["patches"])
      priv_patches.push_back(imagePatch(priv_refimg, imagePatchPosition(i), priv_boxsize));
    patches_valid = true;
  }
}

void registrationContext::boxsize(int new_boxsize) {
  priv_boxsize = new_boxsize;
  boxsize_valid = true;
}

void registrationContext::images(std::vector<inputImage>& new_images) {
  priv_images = new_images;
  images_valid = true;
}

void registrationContext::crop(cv::Rect new_crop) {
  priv_crop = new_crop;
  crop_valid = true;
}

void registrationContext::refimg(cv::Mat& new_refimg) {
  priv_refimg = new_refimg;
  refimg_valid = true;
}

void registrationContext::patches(std::vector<imagePatch>& new_patches) {
  priv_patches = new_patches;
  patches_valid = true;
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
  if (boxsize_valid)
    fs << "boxsize" << priv_boxsize;
  if (images_valid)
    fs << "images" << priv_images;
  if (crop_valid)
    fs << "crop" << priv_crop;
  if (patches_valid)
    fs << "patches" << priv_patches;
  if (refimg_valid)
    fs << "refimg" << priv_refimg;
}

void write(cv::FileStorage& fs,
           const std::string&,
           const registrationContext& context) {
  context.write(fs);
}
