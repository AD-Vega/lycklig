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

#include <cstdio>
#include <iostream>
#include <limits>
#include <Magick++.h>
#include <boost/filesystem.hpp>
#include "imageops.h"
#include "globalregistrator.h"

using namespace cv;

Mat magickImread(const std::string& filename)
{
  Magick::Image image;
  image.read(filename);
  cv::Mat output;

  if (image.colorSpace() == Magick::GRAYColorspace) {
    output = cv::Mat(image.rows(), image.columns(), CV_32F);
    image.write(0, 0, image.columns(), image.rows(),
                "I", Magick::FloatPixel, output.data);
  }
  else {
    output = cv::Mat(image.rows(), image.columns(), CV_32FC3);
    image.write(0, 0, image.columns(), image.rows(),
                  "BGR", Magick::FloatPixel, output.data);
  }
  sRGB2linearRGB(output);
  return output;
}


void magickImwrite16U(const std::string& filename, const Mat& cvImage) {
  std::string map;
  switch (cvImage.channels()) {
    case 1:
      map = "I"; break;
    case 3:
      map = "BGR"; break;
    default:
      std::cerr << "Don't know how to write images that have neither 1 nor 3 channels.\n";
      return;
  }
  Magick::Image image(cvImage.cols, cvImage.rows,
                      map, Magick::ShortPixel, cvImage.data);
  image.write(filename);
}


// Generates a test filename by prepending the true filename with the program
// name and a string of random hex characters.
std::string generateTestFilename(const std::string& origPath) {
  namespace bf = boost::filesystem;
  bf::path path(origPath);
  bf::path dir = path.parent_path();
  std::string filename = path.filename().string();
  std::string randomPrefix = bf::unique_path("lycklig-%%%%%%%%-").string();
  bf::path  newPath = dir / bf::path(randomPrefix + filename);
  return newPath.string();
}


void writeTestImage(const std::string& path) {
  Magick::Image img(Magick::Geometry(1, 1), Magick::ColorGray(1.0));
  std::string testfile(generateTestFilename(path));
  img.write(testfile);
  std::remove(testfile.c_str());
}


void sRGB2linearRGB(Mat& img) {
  int rows = img.rows;
  int cols = img.cols;
  int channels = img.channels();
  if (img.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for(int row = 0; row < rows; row++)
  {
    float* ptr = img.ptr<float>(row);
    for (int i = 0; i < cols*channels; i++) {
      *ptr = (*ptr <= 0.04045 ? *ptr/12.92 : pow((*ptr + 0.055)/1.055, 2.4));
      ptr++;
    }
  }
}


void linearRGB2sRGB(Mat& img) {
  int rows = img.rows;
  int cols = img.cols;
  int channels = img.channels();
  if (img.isContinuous()) {
    cols *= rows;
    rows = 1;
  }
  for(int row = 0; row < rows; row++)
  {
    float* ptr = img.ptr<float>(row);
    for (int i = 0; i < cols*channels; i++) {
      *ptr = (*ptr <= 0.0031308 ? *ptr*12.92 : 1.055*pow(*ptr, 1/2.4) - 0.055);
      ptr++;
    }
  }
}


Mat grayReader::read(const std::string& file) {
  Mat img = magickImread(file.c_str());
  if (img.channels() > 1) {
    // colour image: convert to gray
    cvtColor(img, imggray, CV_BGR2GRAY);
    return imggray;
  }
  else {
    // already gray
    return img;
  }
}


void divideChannelsByMask(Mat& image, Mat& mask)
{
  int rows = image.rows;
  int cols = image.cols;
  int channels = image.channels();

  if(image.isContinuous() && mask.isContinuous())
  {
      cols *= rows;
      rows = 1;
  }
  for (int row = 0; row < rows; row++)
  {
    float* imgptr = image.ptr<float>(row);
    float* normptr = mask.ptr<float>(row);
    for (int pos = 0; pos < cols*channels; pos++)
      imgptr[pos] /= normptr[pos/channels];
  }
}


Mat meanimg(const registrationContext& context,
            const bool showProgress) {
  const auto& images = context.images();

  Mat sample = magickImread(images.at(0).filename);
  Rect imgRect(Point(0, 0), sample.size());
  Mat imgmean = Mat::zeros(sample.size(), CV_MAKETYPE(CV_32F, sample.channels()));
  Mat normalizationMask = Mat::zeros(sample.size(), CV_32F);

  int progress = 0;
  if (showProgress)
    std::fprintf(stderr, "0/%ld", images.size());
  #pragma omp parallel
  {
    Mat localsum(imgmean.clone());
    Mat localNormMask(normalizationMask.clone());
    #pragma omp barrier
    #pragma omp for
    for (int i = 0; i < (signed)images.size(); i++) {
      auto image = images.at(i);
      Mat data = magickImread(image.filename);

      Rect sourceRoi = (imgRect + image.globalShift) & imgRect;
      Rect destRoi = sourceRoi - image.globalShift;
      accumulate(data(sourceRoi), localsum(destRoi));
      localNormMask(destRoi) += image.globalMultiplier;

      if (showProgress) {
        #pragma omp critical
        std::fprintf(stderr, "\r\033[K%d/%ld", ++progress, images.size());
      }
    }
    #pragma omp critical
    accumulate(localsum, imgmean);
    accumulate(localNormMask, normalizationMask);
  }
  if (showProgress)
    std::fprintf(stderr, "\n");

  divideChannelsByMask(imgmean, normalizationMask);
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
  img.convertTo(imgout, CV_MAKETYPE(CV_16U, img.channels()));
  return imgout;
}


imageSumLookup::imageSumLookup(const Mat& img) :
  table(img.size() + cv::Size(1, 1), img.type())
{
  table.row(0) = Scalar(0);
  table.col(0) = Scalar(0);
  if (img.size() == Size(0, 0))
    return;

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
      table.at<float>(rect.y + rect.height, rect.x + rect.width)
    + table.at<float>(rect.y, rect.x)
    - table.at<float>(rect.y + rect.height, rect.x)
    - table.at<float>(rect.y, rect.x + rect.width);
}
