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

#ifndef COOKEDTEMPLATE_H
#define COOKEDTEMPLATE_H

#include <opencv2/core/core.hpp>

class cookedXcor
{
public:
  cookedXcor() {};
  cookedXcor(const cv::Mat& _templ, cv::Size corrsize, int ctype);
  void xcor(const cv::Mat& img, cv::Mat& corr) const;

private:
  int ctype;
  int maxDepth;
  int tdepth;
  int tcn;
  cv::Size corrsize;
  cv::Size templsize;
  cv::Size blocksize;
  cv::Size dftsize;
  cv::Mat dftTempl;
};


class cookedTemplate
{
public:
  cookedTemplate(cv::InputArray _templ, cv::Size searchSize);
  void match(cv::InputArray _img, cv::OutputArray _result) const;

private:
  int templType;
  cv::Size corrSize;
  cookedXcor cxc;
};

#endif // COOKEDTEMPLATE_H
