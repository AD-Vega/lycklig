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

#ifndef REGISTRATIONCONTEXT_H
#define REGISTRATIONCONTEXT_H

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include "imagepatch.h"

class inputImage {
public:
  inputImage(std::string filename_);
  inputImage(const cv::FileNode& node);
  void write(cv::FileStorage& fs) const;

  std::string filename;
  cv::Point globalShift;
};

void write(cv::FileStorage& fs, const std::string&, const inputImage& image);


class registrationContext {
public:
  registrationContext() = default;
  registrationContext(const cv::FileStorage& fs);
  void write(cv::FileStorage& fs) const;
  void printReport() const;

  // accessors
  inline cv::Size imagesize() const { return priv_imagesize; }
  inline int boxsize() const { return priv_boxsize; }
  inline std::vector<inputImage>& images() { return priv_images; }
  inline const std::vector<inputImage>& images() const { return priv_images; }
  inline cv::Rect commonRectangle() const { return priv_commonRectangle; }
  inline const cv::Mat& refimg() const { return priv_refimg; }
  inline const std::vector<imagePatch>& patches() const { return priv_patches; }
  inline const std::vector<cv::Mat1f>& shifts() const { return priv_shifts; }

  // modificators
  void imagesize(cv::Size new_size);
  void boxsize(int new_boxsize);
  void images(std::vector<inputImage>& new_images);
  void commonRectangle(cv::Rect new_commonRectangle);
  void refimg(cv::Mat& new_refimg);
  void patches(std::vector<imagePatch>& new_patches);
  void shifts(std::vector<cv::Mat1f>& new_shifts);

  void clearRefimgEtc();
  void clearPatchesEtc();
  void clearShiftsEtc();

  // checks
  inline bool imagesizeValid() const { return imagesize_valid; }
  inline bool boxsizeValid() const { return boxsize_valid; }
  inline bool imagesValid() const { return images_valid; }
  inline bool commonRectangleValid() const { return commonRectangle_valid; }
  inline bool refimgValid() const { return refimg_valid; }
  inline bool patchesValid() const { return patches_valid; }
  inline bool shiftsValid() const { return shifts_valid; }

private:
  cv::Size priv_imagesize;
  bool imagesize_valid = false;

  int priv_boxsize = 0;
  bool boxsize_valid = false;

  std::vector<inputImage> priv_images;
  bool images_valid = false;

  cv::Rect priv_commonRectangle = cv::Rect(0, 0, 0, 0);
  bool commonRectangle_valid = false;

  cv::Mat priv_refimg;
  bool refimg_valid = false;

  std::vector<imagePatch> priv_patches;
  bool patches_valid = false;

  std::vector<cv::Mat1f> priv_shifts;
  bool shifts_valid = false;
};

void write(cv::FileStorage& fs,
           const std::string&,
           const registrationContext& context);

#endif // REGISTRATIONCONTEXT_H
