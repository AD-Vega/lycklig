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

#include <limits>
#include "imageops.h"
#include "globalregistrator.h"

using namespace cv;

Mat magickImread(const std::string& filename)
{
  Magick::Image image;
  image.read(filename);
  cv::Mat output(image.rows(), image.columns(), CV_32FC3);
  image.write(0, 0, image.columns(), image.rows(),
                 "BGR", Magick::FloatPixel, output.data);
  sRGB2linearRGB(output);
  return output;
}


void sRGB2linearRGB(Mat& img) {
  int rows = img.rows;
  int cols = img.cols;
  if (img.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for(int row = 0; row < rows; row++)
  {
    float* ptr = img.ptr<float>(row);
    for (int i = 0; i < 3*cols; i++) {
      *ptr = (*ptr <= 0.04045 ? *ptr/12.92 : pow((*ptr + 0.055)/1.055, 2.4));
      ptr++;
    }
  }
}


void linearRGB2sRGB(Mat& img) {
  int rows = img.rows;
  int cols = img.cols;
  if (img.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for(int row = 0; row < rows; row++)
  {
    float* ptr = img.ptr<float>(row);
    for (int i = 0; i < 3*cols; i++) {
      *ptr = (*ptr <= 0.0031308 ? *ptr*12.92 : 1.055*pow(*ptr, 1/2.4) - 0.055);
      ptr++;
    }
  }
}


Mat grayReader::read(const string& file) {
  cvtColor(magickImread(file.c_str()), imggray, CV_BGR2GRAY);
  return imggray;
}


Mat meanimg(const registrationParams& params,
            const registrationContext& context,
            const bool showProgress) {
  const auto& images = context.images();

  Mat sample = magickImread(images.at(0).filename);
  Rect imgRect(Point(0, 0), sample.size());
  Mat imgmean = Mat::zeros(sample.size(), CV_MAKETYPE(CV_32F, sample.channels()));

  int progress = 0;
  if (showProgress)
    std::fprintf(stderr, "0/%ld", images.size());
  #pragma omp parallel
  {
    Mat localsum(imgmean.clone());
    #pragma omp barrier
    #pragma omp for
    for (int i = 0; i < (signed)images.size(); i++) {
      auto image = images.at(i);
      Mat data = magickImread(image.filename);

      if (image.globalShift != Point(0, 0)) {
        // TODO: normalization mask
        Rect sourceRoi = (imgRect + image.globalShift) & imgRect;
        Rect destRoi = sourceRoi - image.globalShift;
        accumulate(data(sourceRoi), localsum(destRoi));
      }
      else
        accumulate(data, localsum);

      if (showProgress) {
        #pragma omp critical
        std::fprintf(stderr, "\r\033[K%d/%ld", ++progress, images.size());
      }
    }
    #pragma omp critical
    accumulate(localsum, imgmean);
  }
  if (showProgress)
    std::fprintf(stderr, "\n");
  imgmean /= images.size();
  return imgmean;
}


Mat normalizeTo16Bits(const Mat& inputImg) {
  Mat img = inputImg.clone();
  double minval, maxval;
  minMaxLoc(img, &minval, &maxval);
  img = (img - minval)/(maxval - minval);
  linearRGB2sRGB(img);
  img *= ((1<<16)-1);
  Mat imgout;
  img.convertTo(imgout, CV_16UC3);
  return imgout;
}


imageSumLookup::imageSumLookup(const Mat& img) :
  table(img.size() + cv::Size(1,1), img.type())
{
  table.row(0) = Scalar(0);
  table.col(0) = Scalar(0);
  img.copyTo(table(cv::Rect(Point(1, 1), img.size())));

  for(int row = 2; row < table.rows; row++)
  {
    float* prevptr = table.ptr<float>(row-1);
    float* ptr = table.ptr<float>(row);
    for (int x = 0; x < table.cols; x++)
    {
      *ptr += *prevptr;
      prevptr++;
      ptr++;
    }
  }

  for(int row = 1; row < table.rows; row++)
  {
    float* ptr = table.ptr<float>(row);
    float* nextptr = ptr + 1;
    for (int x = 1; x < table.cols; x++)
    {
      *nextptr += *ptr;
      ptr++;
      nextptr++;
    }
  }
}


float imageSumLookup::lookup(const Rect rect) const
{
  return
      table.at<float>(rect.x + rect.width, rect.y + rect.height)
    + table.at<float>(rect.x, rect.y)
    - table.at<float>(rect.x + rect.width, rect.y)
    - table.at<float>(rect.x, rect.y + rect.height);
}
