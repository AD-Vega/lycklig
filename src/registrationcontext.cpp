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
  filename(filename_), globalShift(0, 0), globalMultiplier(1)
{}


inputImage::inputImage(const cv::FileNode& node) {
  node["filename"] >> filename;
  node["globalShift"] >> globalShift;
  node["globalMultiplier"] >> globalMultiplier;
}


void inputImage::write(cv::FileStorage& fs) const {
  fs << "{"
     << "filename" << filename
     << "globalShift" << globalShift
     << "globalMultiplier" << globalMultiplier
     << "}";
}


void write(cv::FileStorage& fs, const cv::String&, const inputImage& image) {
  image.write(fs);
}


registrationContext::registrationContext(const cv::FileStorage& fs) {
  if (!fs["imagesize"].empty()) {
    cv::Size new_imagesize;
    fs["imagesize"] >> new_imagesize;
    imagesize(new_imagesize);
  }

  if (fs["boxsize"].isInt()) {
    int new_boxsize;
    fs["boxsize"] >> new_boxsize;
    boxsize(new_boxsize);
  }

  if (! fs["commonRectangle"].empty()) {
    cv::Rect new_commonRectangle;
    fs["commonRectangle"] >> new_commonRectangle;
    commonRectangle(new_commonRectangle);
  }

  if (! fs["refimg"].empty()) {
    cv::Mat new_refimg;
    fs["refimg"] >> new_refimg;
    refimg(new_refimg);
  }

  if (fs["images"].isSeq()) {
    std::vector<inputImage> new_images;
    for (const auto& i : fs["images"])
      new_images.push_back(inputImage(i));
    images(new_images);
  }

  if (fs["patches"].isSeq() && ! fs["patchCreationArea"].empty()) {
    patchCollection new_patches;
    for (const auto& i : fs["patches"])
      new_patches.push_back(imagePatch(refimg(), imagePatchPosition(i), boxsize()));
    fs["patchCreationArea"] >> new_patches.patchCreationArea;
    patches(new_patches);
  }

  if (fs["shifts"].isSeq()) {
    std::vector<cv::Mat1f> new_shifts;
    for (const auto& i : fs["shifts"]) {
      cv::Mat1f shifts;
      i >> shifts;
      new_shifts.push_back(shifts);
    }
    shifts(new_shifts);
  }
}

void registrationContext::clearRefimgEtc() {
  if (refimg.valid())
    std::cerr << "  Invalidating current reference image\n";
  refimg.invalidate();
  clearPatchesEtc();
}

void registrationContext::clearPatchesEtc() {
  if (patches.valid())
    std::cerr << "  Invalidating existing registration points\n";
  boxsize.invalidate();
  patches.invalidate();
  clearShiftsEtc();
}

void registrationContext::clearShiftsEtc() {
  if (shifts.valid())
    std::cerr << "  Invalidating existing dedistortion shifts\n";
  shifts.invalidate();
}


void write(cv::FileStorage& fs,
           const cv::String&,
           const std::vector<inputImage>& images) {
  fs << "[";
  for (auto& image : images)
    fs << image;
  fs << "]";
}

void write(cv::FileStorage& fs,
           const cv::String&,
           const patchCollection& patches) {
  fs << "[";
  for (auto& patch : patches)
    fs << patch;
  fs << "]";
  fs << "patchCreationArea" << patches.patchCreationArea;
}

void registrationContext::write(cv::FileStorage& fs) const {
  if (imagesize.valid())
    fs << "imagesize" << imagesize();
  if (boxsize.valid())
    fs << "boxsize" << boxsize();
  if (images.valid())
    fs << "images" << images();
  if (commonRectangle.valid())
    fs << "commonRectangle" << commonRectangle();
  if (patches.valid())
    fs << "patches" << patches();
  if (refimg.valid())
    fs << "refimg" << refimg();
  if (shifts.valid())
    fs << "shifts" << shifts();
}

void write(cv::FileStorage& fs,
           const cv::String&,
           const registrationContext& context) {
  context.write(fs);
}

void registrationContext::printReport() const {
  if (images.valid())
    std::cerr << "  * " << images().size() << " images ("
      << imagesize().width << "x" << imagesize().height << ")\n";
  if (commonRectangle.valid())
    std::cerr << "  * global registration data\n";
  if (refimg.valid())
    std::cerr << "  * reference image\n";
  if (patches.valid())
    std::cerr << "  * " << patches().size() << " registration points (boxsize "
      << boxsize() << ")\n";
  if (shifts.valid())
    std::cerr << "  * dedistortion shifts\n";
}
