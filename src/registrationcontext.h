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
  float globalMultiplier;
};

void write(cv::FileStorage& fs, const cv::String&, const inputImage& image);


// This class acts as a proxy that tracks the validity of the contained
// object.
template <typename T>
class managed {
public:
  managed() = default;
  managed(const T& initialValue) : value(initialValue), isValid(true) {}

  inline const T& operator()() const { return value; }
  inline T& operator()() { return value; }
  inline bool valid() const { return isValid; }

  void operator()(T newValue) {
    value = std::move(newValue);
    isValid = true; }

  void invalidate() {
    value = T();
    isValid = false;
  }

private:
  T value;
  bool isValid = false;
};


class registrationContext {
public:
  registrationContext() = default;
  registrationContext(const cv::FileStorage& fs);
  void write(cv::FileStorage& fs) const;
  void printReport() const;

  managed<cv::Size> imagesize;
  managed<int> boxsize;
  managed<std::vector<inputImage>> images;
  managed<cv::Rect> commonRectangle;
  managed<cv::Mat> refimg;
  managed<patchCollection> patches;
  managed<std::vector<cv::Mat1f>> shifts;

  void clearRefimgEtc();
  void clearPatchesEtc();
  void clearShiftsEtc();

  // convenience methods
  inline cv::Rect refimgRectangle() const
    { return cv::Rect(cv::Point(0, 0), refimg().size()); }
};

void write(cv::FileStorage& fs,
           const cv::String&,
           const registrationContext& context);

#endif // REGISTRATIONCONTEXT_H
