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
  if (!fs["imagesize"].empty()) {
    fs["imagesize"] >> priv_imagesize;
    imagesize_valid = true;
  }

  if (fs["boxsize"].isInt()) {
    fs["boxsize"] >> priv_boxsize;
    boxsize_valid = true;
  }

  if (! fs["commonRectangle"].empty()) {
    fs["commonRectangle"] >> priv_commonRectangle;
    commonRectangle_valid = true;
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

  if (fs["shifts"].isSeq()) {
    for (const auto& i : fs["shifts"]) {
      cv::Mat1f shifts;
      i >> shifts;
      priv_shifts.push_back(shifts);
    }
    shifts_valid = true;
  }
}

void registrationContext::imagesize(cv::Size new_imagesize) {
  priv_imagesize= new_imagesize;
  imagesize_valid = true;
}

void registrationContext::boxsize(int new_boxsize) {
  priv_boxsize = new_boxsize;
  boxsize_valid = true;
}

void registrationContext::images(std::vector<inputImage>& new_images) {
  priv_images = new_images;
  images_valid = true;
}

void registrationContext::commonRectangle(cv::Rect new_commonRectangle) {
  priv_commonRectangle = new_commonRectangle;
  commonRectangle_valid = true;
}

void registrationContext::refimg(cv::Mat& new_refimg) {
  priv_refimg = new_refimg;
  refimg_valid = true;
}

void registrationContext::patches(patchCollection& new_patches) {
  priv_patches = new_patches;
  patches_valid = true;
}

void registrationContext::shifts(std::vector< cv::Mat1f >& new_shifts)
{
  priv_shifts = new_shifts;
  shifts_valid = true;
}

void registrationContext::clearRefimgEtc() {
  if (refimg_valid)
    std::cerr << "  Invalidating current reference image\n";
  priv_refimg = cv::Mat();
  refimg_valid = false;
  clearPatchesEtc();
}

void registrationContext::clearPatchesEtc() {
  if (patches_valid)
    std::cerr << "  Invalidating existing registration points\n";
  priv_boxsize = 0;
  boxsize_valid = false;
  priv_patches.clear();
  patches_valid = false;
  clearShiftsEtc();
}

void registrationContext::clearShiftsEtc() {
  if (shifts_valid)
    std::cerr << "  Invalidating existing lucky imaging shifts\n";
  priv_shifts.clear();
  shifts_valid = false;
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
           const patchCollection& patches) {
  fs << "[";
  for (auto& patch : patches)
    fs << patch;
  fs << "]";
}

void registrationContext::write(cv::FileStorage& fs) const {
  if (imagesize_valid)
    fs << "imagesize" << priv_imagesize;
  if (boxsize_valid)
    fs << "boxsize" << priv_boxsize;
  if (images_valid)
    fs << "images" << priv_images;
  if (commonRectangle_valid)
    fs << "commonRectangle" << priv_commonRectangle;
  if (patches_valid)
    fs << "patches" << priv_patches;
  if (refimg_valid)
    fs << "refimg" << priv_refimg;
  if (shifts_valid)
    fs << "shifts" << priv_shifts;
}

void write(cv::FileStorage& fs,
           const std::string&,
           const registrationContext& context) {
  context.write(fs);
}

void registrationContext::printReport() const {
  if (images_valid)
    std::cerr << "  * " << images().size() << " images ("
      << imagesize().width << "x" << imagesize().height << ")\n";
  if (commonRectangle_valid)
    std::cerr << "  * global registration data\n";
  if (refimg_valid)
    std::cerr << "  * reference image\n";
  if (patches_valid)
    std::cerr << "  * " << patches().size() << " registration points (boxsize "
      << boxsize() << ")\n";
  if (shifts_valid)
    std::cerr << "  * lucky imaging shifts\n";
}
